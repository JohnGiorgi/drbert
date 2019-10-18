# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import trange
from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers.modeling_utils import WEIGHTS_NAME

from tensorboardX import SummaryWriter

from .constants import COHORT_DISEASE_LIST
from .constants import COHORT_INTUITIVE_LABEL_CONSTANTS
from .constants import COHORT_TEXTUAL_LABEL_CONSTANTS
from .constants import DEID_LABELS
from .eval import evaluate
from .model import DrBERT
from .utils import data_utils
from .utils import eval_utils
from .utils import train_utils

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, deid_dataset, cohort_dataset, model, tokenizer):
    """Coordinates training of DrBERT.

    Args:
        args (ArgumentParser): Object containing objects parsed from the command line.
        deid_dataset (dict): A dictionary, keyed by partition, containing a TensorDataset for each
            partition of the DeID dataset.
        cohort_dataset (dict): A dictionary, keyed by partition, containing a TensorDataset for each
            partition of the Cohort Identification dataset.
        model (nn.Module): The DrBERT model to train.
        tokenizer (BertTokenizer): A transformers tokenizer object.

    Raises:
        ImportError if args.fp16 but Apex is not installed.
    """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    # TODO (John): Preperation of dataloaders is hardcoded. Need to move to JSON configurable.
    # Prepare dataloaders
    deid_sampler = (RandomSampler(deid_dataset['train']) if args.local_rank == -1
                    else DistributedSampler(deid_dataset['train']))
    deid_dataloader = \
        DataLoader(deid_dataset['train'], sampler=deid_sampler, batch_size=args.train_batch_size)

    cohort_sampler = (RandomSampler(cohort_dataset['train']) if args.local_rank == -1
                      else DistributedSampler(cohort_dataset['train']))
    # batch_size hardcoded to 1 because these sentences are grouped by document
    cohort_dataloader = DataLoader(cohort_dataset['train'], sampler=cohort_sampler, batch_size=1)

    # TODO (John): Hardcoded, but an example of what the JSON will look like
    tasks = [
        {'name': 'deid', 'task': 'sequence_labelling', 'iterator': deid_dataloader},
        {'name': 'cohort', 'task': 'document classification', 'iterator': cohort_dataloader},
    ]

    # TODO (John): These two lines will change when we switch to torchtext
    # The number of batches / examples the sum of the number of batches / examples from all tasks
    num_examples = sum([len(task['iterator'].dataset) for task in tasks])
    num_batches = sum([len(task['iterator']) for task in tasks])

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // num_batches
    else:
        t_total = num_batches * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    optimizer, scheduler = train_utils.prepare_optimizer_and_scheduler(args, model, t_total)

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
    logger.info("  Num Examples = %d", num_examples)
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                (args.train_batch_size * torch.distributed.get_world_size() if args.local_rank != -1 else 1)
                )
    logger.info("  Total optimization steps = %d", t_total)

    task_step = {task['name']: 0 for task in tasks}
    tr_loss = {task['name']: 0.0 for task in tasks}
    logging_loss = {task['name']: 0.0 for task in tasks}

    model.zero_grad()

    train_iterator = trange(args.num_train_epochs,
                            desc="Epoch",
                            disable=args.local_rank not in [-1, 0],
                            dynamic_ncols=True)

    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)

    for _, _ in enumerate(train_iterator):
        '''
        epoch_iterator = tqdm(zip_longest(task['iterator'] for task in tasks),
                              unit="batch", desc="Iteration",
                              total=num_batches,
                              disable=args.local_rank not in [-1, 0],
                              dynamic_ncols=True)
        '''
        task_ids = list(range(len(tasks)))
        epoch_iterator = trange(num_batches,
                                unit="batch",
                                desc="Iteration",
                                disable=args.local_rank not in [-1, 0],
                                dynamic_ncols=True)
        for step, _ in enumerate(epoch_iterator):
            model.train()

            # Training order of two tasks is chosen at random every batch.
            while True:
                task_id = random.choice(task_ids)
                try:
                    name, task, iterator = tasks[task_id].values()
                    batch = next(iter(iterator))
                    break
                except StopIteration:
                    # Remove task_id from candidates once we have exuasted that tasks iterator
                    del task_ids[task_id]

            # Dataloader introduces a first dimension of size one
            if task == 'document_classification':
                batch[0] = batch[0].squeeze(0)
                batch[1] = batch[1].squeeze(0)
                batch[2] = batch[2].squeeze(0)

            batch = tuple(t.to(args.device) for t in batch)

            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'labels':         batch[2],
                      'task':           task,
                      }

            outputs = model(**inputs)
            loss = outputs[0]  # outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                # mean() to average on multi-gpu parallel (not distributed) training
                loss = loss.mean()

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss[name] += loss.item()

            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            task_step[name] += 1

            # Log metrics
            if args.local_rank in [-1, 0] and args.logging_steps > 0 and step % args.logging_steps == 0:
                # Only evaluate when single GPU otherwise metrics may not average well
                if args.local_rank == -1 and args.evaluate_during_training:
                    results = {}
                    for task in tasks:
                        name, task, iterator = task.values()
                        results[name] = evaluate(args, model, iterator.dataset, task)

                    # HACK (John): This is a temporary. Need a more principled API for
                    # saving results to disk.
                    eval_utils.save_eval_to_disk(args, step, **results)

                    ''' TODO
                    for key, value in deid_results.items():
                        tb_writer.add_scalar('deid_eval_{}'.format(key), value, global_step)
                    for key, value in cohort_results.items():
                        tb_writer.add_scalar('cohort_eval_{}'.format(key), value, global_step)
                    '''
                tb_writer.add_scalar('lr', scheduler.get_lr()[0], step)
                tb_writer.add_scalar(f'{name}_loss', (tr_loss[name] - logging_loss[name]) / args.logging_steps, task_step[name])
                logging_loss[name] = tr_loss[name]

            # Save model checkpoint
            if args.local_rank in [-1, 0] and args.save_steps > 0 and step % args.save_steps == 0:
                output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                logger.info("Saving model checkpoint to %s", output_dir)

            postfix = \
                {f'{name}_loss': f'{(tr_loss[name] / task_step[name]):.6f}' for name in tr_loss}
            postfix['total_loss'] = sum(postfix.values())

            epoch_iterator.set_postfix(postfix)

            del batch, inputs, outputs

            if args.max_steps > 0 and step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    # Compute the average loss for each task
    avg_loss = {name: loss / task_step[name] for name, loss in tr_loss.items()}

    return step, avg_loss


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    # TODO (John): If "technical debt" had a dictionary entry they would show this as example
    parser.add_argument("--cohort_type", default='textual', choices=['textual', 'intuitive'], type=str, help="what do you think this is")

    parser.add_argument("--dataset_folder", default=None, type=str, required=True,
                        help="De-id and co-hort identification data directory.")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help=("The output directory where the model checkpoints and predictions"
                              " will be written."))

    # Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name.")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name.")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help=("The maximum total input sequence length after WordPiece"
                              " tokenization. Sequences longer than this will be truncated, and"
                              " sequences shorter than this will be padded."))
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Total batch size to use while training.")
    parser.add_argument("--eval_batch_size", default=128, type=int,
                        help="Total batch size to use while evaluating.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=5, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup", default=0, type=float,
                        help=("Proportion of training steps to perform a linear warmup of the"
                              " learning rate."))
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal evaluation.")

    parser.add_argument('--logging_steps', type=int, default=1000,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=1000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for initialization.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--server_ip', type=str, default='',
                        help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='',
                        help="Can be used for distant debugging.")
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
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case
    )

    # Process the data
    deid_dataset, deid_class_weights = data_utils.prepare_deid_dataset(args, tokenizer)
    cohort_dataset = data_utils.prepare_cohort_dataset(args, tokenizer)

    # Here, we add any additional configs to the Pytorch Transformers config file. These will be
    # saved to a `output_dir/config.json` file when we call model.save_pretrained(output_dir)
    config.__dict__['num_deid_labels'] = len(DEID_LABELS)
    # Num of labels is dependent on whether we are running the textual or intuitive analysis
    if args.cohort_type:
        config.__dict__['num_cohort_labels'] = \
            len(COHORT_DISEASE_LIST), len(COHORT_TEXTUAL_LABEL_CONSTANTS)
    else:
        config.__dict__['num_cohort_labels'] = \
            len(COHORT_DISEASE_LIST), len(COHORT_INTUITIVE_LABEL_CONSTANTS)
    config.__dict__['cohort_ffnn_size'] = 512
    config.__dict__['max_batch_size'] = args.train_batch_size
    config.__dict__['deid_class_weights'] = deid_class_weights

    model = DrBERT.from_pretrained(
        pretrained_model_name_or_path=args.model_name_or_path,
        config=config
    )

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

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

        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = DrBERT.from_pretrained(
            pretrained_model_name_or_path=args.model_name_or_path,
            config=config
        )
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)

    # TODO (John): This is a copy paste from the Transformers library.
    # It does not currently work with out setup.
    """
    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = \
                [os.path.dirname(c) for c in
                 sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True))
                 ]
            # Reduce model loading logs
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            # TODO Change AutoModel to whatever we have named our BERT model
            model = DrBERT.from_pretrained(
                pretrained_model_name_or_path=args.model_name_or_path,
                config=config
            )
            model.to(args.device)

            # Evaluate
            result = evaluate(args, model, tokenizer, prefix=global_step)

            result = {(k + ('_{}'.format(global_step) if global_step else ''), v)
                      for k, v in result.items()}
            results.update(result)

    logger.info("Results: {}".format(results))

    return results
    """


if __name__ == "__main__":
    main()
