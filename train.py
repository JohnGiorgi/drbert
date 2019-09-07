import argparse
import glob
import logging
import os
import random
from itertools import zip_longest

import numpy as np
import torch
from pytorch_transformers import (AdamW, AutoConfig, AutoTokenizer,
                                  WarmupLinearSchedule)
from pytorch_transformers.modeling_utils import WEIGHTS_NAME
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from sklearn.metrics import precision_recall_fscore_support

import data_utils
import eval_utils
from constants import *
from model import BertForJointDeIDAndCohortID

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, deid_dataset, cohort_dataset, model, tokenizer):
    """Coordinates training of a multi-task patient note deidentification and cohort classification
    model.

    Args:
        args (ArgumentParser): Object containing objects parsed from the command line.
        deid_dataset (dict): A dictionary, keyed by partition, containing a TensorDataset for each
            partition of the DeID dataset.
        cohort_dataset (dict): A dictionary, keyed by partition, containing a TensorDataset for each
            partition of the Cohort Identification dataset.
        model (nn.Module): The multi-task patient note de-identification and cohort classification
            model.
        tokenizer (BertTokenizer): A pytorch-transformers tokenizer object.

    Raises:
        ImportError if args.fp16 but Apex is not installed.
    """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    deid_sampler = (RandomSampler(deid_dataset['train']) if args.local_rank == -1
                    else DistributedSampler(deid_dataset['train']))
    deid_dataloader = \
        DataLoader(deid_dataset['train'], sampler=deid_sampler, batch_size=args.train_batch_size)

    cohort_sampler = (RandomSampler(cohort_dataset['train']) if args.local_rank == -1
                      else DistributedSampler(cohort_dataset['train']))
        
    # batch_size hardcoded to 1 because these sentences are grouped by document
    cohort_dataloader = DataLoader(cohort_dataset['train'], sampler=cohort_sampler, batch_size=1)

    deid_total = len(deid_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    cohort_total = len(cohort_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    t_total = ((len(deid_dataloader) + len(cohort_dataloader)) //
               args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(('Please install apex from https://www.github.com/nvidia/apex to use'
                               ' fp16 training.'))
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # Multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(deid_dataloader) + len(cohort_dataloader))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    # logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                (args.train_batch_size * args.gradient_accumulation_steps *
                 torch.distributed.get_world_size() if args.local_rank != -1 else 1)
                )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss = {'deid': 0.0, 'cohort': 0.0}
    logging_loss = {'deid': 0.0, 'cohort': 0.0}

    model.zero_grad()

    train_iterator = trange(int(args.num_train_epochs),
                            desc="Epoch",
                            disable=args.local_rank not in [-1, 0],
                            dynamic_ncols=True)

    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)

    for epoch, _ in enumerate(train_iterator):
        epoch_iterator = tqdm(zip_longest(deid_dataloader, cohort_dataloader),
                              unit="batch", desc="Iteration",
                              total=t_total,
                              disable=args.local_rank not in [-1, 0],
                              dynamic_ncols=True)
        for step, (deid_batch, cohort_batch) in enumerate(epoch_iterator):
            model.train()

            # Dataloader introduces a first dimension of size one
            if cohort_batch is not None:
                cohort_batch[0] = cohort_batch[0].squeeze(0)
                cohort_batch[1] = cohort_batch[1].squeeze(0)
                cohort_batch[2] = cohort_batch[2].squeeze(0)

            # Training order of two tasks is chosen at random every batch
            '''
            batch_pair = [batch for batch in [(deid_batch, 'deid'), (cohort_batch, 'cohort')]
                          if batch[0] is not None]

            batch_pair = [batch_pair[0]]
            random.shuffle(batch_pair)
            '''

            batch_pair = [(deid_batch, 'deid')]
            
            for batch, task in batch_pair:
                if batch is None:
                    break

                batch = tuple(t.to(args.device) for t in batch)

                # TODO (Nick, Gary): Make sure both dataloader return objects in this order
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'labels':         batch[2],
                          'task':           task,
                          }

                outputs = model(**inputs)
                loss = outputs[0]  # outputs are always tuple in pytorch-transformers (see doc)

                if args.n_gpu > 1:
                    # mean() to average on multi-gpu parallel (not distributed) training
                    loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                tr_loss[task] += loss.item()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    scheduler.step()  # Update learning rate schedule
                    optimizer.step()
                    model.zero_grad()
                    global_step += 1

                    # Log metrics
                    if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                            deid_results = evaluate(args, model, tokenizer, deid_dataset, task='deid')
                            # cohort_results = evaluate(args, model, tokenizer, cohort_dataset, task='cohort')
                            cohort_results = {}
                            for key, value in deid_results.items():
                                tb_writer.add_scalar('deid_eval_{}'.format(key), value, global_step)
                            for key, value in cohort_results.items():
                                tb_writer.add_scalar('cohort_eval_{}'.format(key), value, global_step)
                        tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar('loss', (tr_loss[task] - logging_loss[task])/args.logging_steps, global_step)
                        logging_loss[task] = tr_loss[task]

                    # Save model checkpoint
                    if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                        output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                        logger.info("Saving model checkpoint to %s", output_dir)

                deid_loss = tr_loss["deid"] / (deid_total * epoch + (step + 1))
                cohort_loss = tr_loss["cohort"] / (cohort_total * epoch + (step + 1))

                postfix = {'deid_loss': f'{deid_loss:.6f}',
                           'cohort_loss': f'{cohort_loss:.6f}',
                           'total_loss': f'{(deid_loss + cohort_loss):.6f}'
                           }
                epoch_iterator.set_postfix(postfix)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    # Compute the average loss for each task
    avg_loss = {key: value / global_step for key, value in tr_loss.items()}

    return global_step, avg_loss


def evaluate(args, model, tokenizer, dataset, task="deid"):
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    eval_dataloader = {}

    # batch_size hardcoded to 1 for cohort task because these sentences are grouped by document
    batch_size = args.eval_batch_size if task == 'deid' else 1

    for partition in dataset:
        sampler = (SequentialSampler(dataset[partition]) if args.local_rank == -1
                   # Note that DistributedSampler samples randomly
                   else DistributedSampler(dataset[partition]))
        eval_dataloader[partition] = \
            DataLoader(dataset[partition], sampler=sampler, batch_size=batch_size)

    # Eval!
    model.eval()
    for partition, dataset in eval_dataloader.items():
        logger.info(f"***** Running {task} evaluation on {partition} *****")
        logger.info("  Num examples = %d", len(eval_dataloader))
        logger.info("  Batch size = %d", args.eval_batch_size)

        labels, predictions, orig_to_tok_map = [], [], []
        for batch in tqdm(dataset, desc="Evaluating"):
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'labels':         batch[2],
                          'task':           task
                          }

                outputs = model(**inputs)
                loss, logits = outputs[:2]

                # Need to accumulate these for evaluation
                labels.append(inputs['labels'])
                predictions.append(logits.argmax(dim=-1))
                if task == 'deid':
                    orig_to_tok_map.append(batch[3])

        # TODO: result need to be a dictionary of keys (corresponding to some metric) and values
        # (corresponding to the score for that metric), e.g. {'f1': 0.76}, in order for the rest of
        # this script to not break
        if task == 'deid':
            results = evaluate_deid(labels, predictions, orig_to_tok_map)
        elif task == 'cohort':
            results = evaluate_cohort(labels, predictions)

        print(results)
        eval_utils.print_evaluation(results)

    return results


def evaluate_deid(labels, predictions, orig_to_tok_map):
    # TODO (John): idx to tag and tag to idx mapping needs to be re-thought.
    idx_to_tag = eval_utils.reverse_dict(DEID_LABELS)

    y_true, y_pred = [], []
    for labs, preds, tok_map in zip(labels, predictions, orig_to_tok_map):
        orig_token_indices = torch.as_tensor(
            [i for i in tok_map if i != TOK_MAP_PAD], device=labels.device, dtype=torch.long
        )

        masked_labels = torch.index_select(labs, -1, orig_token_indices).tolist()
        masked_preds = torch.index_select(preds, -1, orig_token_indices).tolist()

        # Map predictions to tags
        # TODO:
        y_true.extend([idx_to_tag[idx] for idx in masked_labels])
        y_pred.extend([idx_to_tag[idx] for idx in masked_preds])

    scores = eval_utils.precision_recall_f1_support_sequence_labelling(y_true, y_pred)
    print(scores)

    return scores


def evaluate_cohort(labels, predictions):
    """Coordinates evaluation for the cohort identification task.
    Args:
        labels (list): A list of Tensors containing the gold labels for each example.
        predictions (list): A list of Tensors containing the predicted labels for each example.

    Returns:
    """
    assert len(labels) == len(predictions)
    
    #P, R, F1, support
    # disease_dict = {value:[[],[]] for (key,value) in COHORT_DISEASE_CONSTANTS.items()}
    # score_dict = {key: [] for (key,value) in COHORT_DISEASE_CONSTANTS.items()}
    
    # for prediction, label in zip(predictions, labels):
    #     for i in range(len(prediction)): 
    #        disease_dict[i][0].extend(label[i]) 
    #        disease_dict[i][1].extend(prediction[i])

    # for disease in disease_dict: 
    #     precision, recall, f1, support = precision_recall_fscore_support(disease[0], disease[1])

    # # return disease_dict


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--dataset_folder", default=None, type=str, required=True,
                        help="De-id and co-hort identification data directory")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")

    # Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")

    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=128, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal evaluation.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path)
    config = AutoConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)

    # Process the data
    deid_dataset = data_utils.prepare_deid_dataset(args, tokenizer)
    cohort_dataset = data_utils.prepare_cohort_dataset(args, tokenizer)

    # TODO: Come up with a much better scheme for this
    config.__dict__['num_deid_labels'] = len(DEID_LABELS)
    config.__dict__['num_cohort_disease'] = len(COHORT_LABEL_CONSTANTS)
    config.__dict__['num_cohort_classes'] = len(COHORT_DISEASE_LIST)
    config.__dict__['cohort_ffnn_size'] = 512
    config.__dict__['max_batch_size'] = args.train_batch_size

    model = BertForJointDeIDAndCohortID.from_pretrained(
        pretrained_model_name_or_path=args.model_name_or_path,
        config=config
    )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        global_step, tr_loss = train(args, deid_dataset, cohort_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Save the trained model and the tokenizer
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = BertForJointDeIDAndCohortID.from_pretrained(
            pretrained_model_name_or_path=args.model_name_or_path,
            config=config
        )
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)


    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            # Reduce model loading logs
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            # TODO Change AutoModel to whatever we have named our BERT model
            model = BertForJointDeIDAndCohortID.from_pretrained(
                pretrained_model_name_or_path=args.model_name_or_path,
                config=config
            )
            model.to(args.device)

            # Evaluate
            result = evaluate(args, model, tokenizer, prefix=global_step)

            result = dict((k + ('_{}'.format(global_step) if global_step else ''), v) for k, v in result.items())
            results.update(result)

    logger.info("Results: {}".format(results))

    return results


if __name__ == "__main__":
    main()
