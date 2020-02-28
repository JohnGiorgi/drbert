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
import argparse
import glob
import json
import logging
import os
import random

import numpy as np
import torch
from tqdm import trange
from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers.modeling_utils import WEIGHTS_NAME

from tensorboardX import SummaryWriter

from .data.util import batch_sizes_to_tuple
from .evaluation.eval import evaluate
from .model import DrBERT
from .modules.proportional_batch_sampler import ProportionalBatchSampler
from .training.util import generate_inputs
from .training.util import prepare_optimizer_and_scheduler
from .training.util import get_iterators_for_task

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, tasks, model, tokenizer):
    """Coordinates training of DrBERT.

    Args:
        args (ArgumentParser): Object containing objects parsed from the command line.
        tasks (list): A list of dictionaries containing information about each task to train.
        model (model.DrBERT): The DrBERT model to train.
        tokenizer (PretrainedTokenizer): A transformers tokenizer object.

    Raises:
        ImportError if args.fp16 but Apex is not installed.
    """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    # TODO (John): Commenting this out until we add back in the per_gpu_* args
    # args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    # The number of batches / examples is the sum of the number of batches / examples from all tasks
    num_examples = sum([len(task['iterators']['train'].dataset.examples) for task in tasks])
    num_batches = sum([len(task['iterators']['train']) for task in tasks])

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // num_batches + 1
    else:
        t_total = num_batches * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    optimizer, scheduler = prepare_optimizer_and_scheduler(args, model, t_total)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            err_msg = ('Please install apex from https://www.github.com/nvidia/apex to use'
                       ' fp16 training.')
            logger.error('ImportError: %s', err_msg)
            raise ImportError(err_msg)
        model, optimizer = amp.initialize(model.to(args.device), optimizer, opt_level=args.fp16_opt_level)

    # Multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train
    logger.info("***** Running training *****")
    logger.info(f"  Num Examples = {num_examples}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Total optimization steps = {t_total}")

    task_step = {task['name']: 0 for task in tasks}
    tr_loss = {task['name']: 0.0 for task in tasks}
    logging_loss = {task['name']: 0.0 for task in tasks}

    model.zero_grad()

    train_iterator = trange(args.num_train_epochs,
                            desc="Epoch",
                            disable=args.local_rank not in [-1, 0],
                            dynamic_ncols=True)

    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)

    for _ in train_iterator:
        epoch_iterator = trange(num_batches,
                                desc="Iteration",
                                unit="batch",
                                disable=args.local_rank not in [-1, 0],
                                dynamic_ncols=True)

        batch_sampler = ProportionalBatchSampler(tasks, 'train')

        for _ in epoch_iterator:
            model.train()

            # The global step is the total number of steps taken for each task
            global_step = sum(list(task_step.values()))

            # Training order of tasks is chosen using proportional sampling
            name, task, batch = batch_sampler.get_batch()

            inputs = generate_inputs(name, task, batch, tokenizer)

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
            log_metrics = (args.local_rank in [-1, 0] and args.logging_steps > 0 and
                           (global_step + 1) % args.logging_steps == 0)
            if log_metrics:
                # Only evaluate when single GPU otherwise metrics may not average well
                if args.local_rank == -1 and args.evaluate_during_training:
                    results = evaluate(tasks, model, tokenizer, partitions=['test'])

                    '''
                    # HACK (John): This is a temporary. Need a more principled API for
                    # saving results to disk.
                    evaluating.util.save_eval_to_disk(args, step, **results)

                    for key, value in deid_results.items():
                        tb_writer.add_scalar('deid_eval_{}'.format(key), value, global_step)
                    for key, value in cohort_results.items():
                        tb_writer.add_scalar('cohort_eval_{}'.format(key), value, global_step)
                    '''
                tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar(
                    f'{name}_loss', (tr_loss[name] - logging_loss[name]) / args.logging_steps, task_step[name]
                )
                logging_loss[name] = tr_loss[name]

            # Save model checkpoint
            save = (args.local_rank in [-1, 0] and args.save_steps > 0 and
                    (global_step + 1) % args.save_steps == 0)
            if save:
                output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step + 1))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                # Take care of distributed/parallel training
                model_to_save = model.module if hasattr(model, 'module') else model
                model_to_save.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                logger.info("Saving model checkpoint to %s", output_dir)

            # Display per task loss in the progress bar (and total loss if > 1 tasks)
            postfix = {
                f'{name}_loss': f'{(tr_loss[name] / task_step[name]):.6f}'
                # arbitrary '0' loss for tasks that haven't begun training yet
                if task_step[name] else 0 for name in tr_loss
            }
            if len(tr_loss) > 1:
                postfix['total_loss'] = \
                    f'{sum([float(loss) for loss in list(postfix.values())]):.6f}'

            epoch_iterator.set_postfix(postfix)

            del batch, inputs, outputs

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    # Compute the average loss for each task
    avg_loss = {name: loss / task_step[name] for name, loss in tr_loss.items()}

    return global_step, avg_loss


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    # TODO (John): Add this functionality back in. Right now, everything is hardcoded to BERT.
    # parser.add_argument("--model_type", default=None, type=str, required=True,
    #                     help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--task_config", default=None, type=str, required=True,
                        help="Path to the task configuration file used to build the DrBERT model.")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help=("The output directory where the model checkpoints and predictions"
                              " will be written."))

    # Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help=("Pretrained Transformers config name or path if not the same as"
                              " model_name."))
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name.")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")

    # TODO (John): These should all override task_config
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

    if (os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train
       and not args.overwrite_output_dir):
        err_msg = (f"Output directory ({args.overwrite_output_dir}) already exists and is not empty"
                   " . Use --overwrite_output_dir to overcome.")
        logger.error('ValueError: %s', err_msg)
        raise ValueError(err_msg)

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging
        # See: https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
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

    model = DrBERT.from_pretrained(
        pretrained_model_name_or_path=args.model_name_or_path,
        config=config
    )

    model.to(args.device)

    # Load the task config
    with open(args.task_config, 'r') as f:
        tasks = json.load(f, object_hook=batch_sizes_to_tuple)

    # Load the datasets and register the classification heads based on the provided JSON config
    for task in tasks:
        task['iterators'] = get_iterators_for_task(task, tokenizer, args.device)

        if task['task'] == 'sts':
            num_labels = 1
        else:
            # TODO (John): Going to cause issues with seq labelling b/c of CLS, PAD and SEP tokens.
            num_labels = len(task['iterators']['train'].dataset.fields['label'].vocab)

        model.register_classification_head(task['name'], task=task['task'], num_labels=num_labels)

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of
    # torch.einsum if args.fp16 is set. Otherwise it'll default to "promote" mode, and we'll get
    # fp32 operations. Note that running `--fp16_opt_level="O2"` will remove the need for this code,
    # but it is still valid.
    if args.fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, 'einsum')
        except ImportError:
            err_msg = ("Please install apex from https://www.github.com/nvidia/apex to use fp16"
                       " training.")
            logger.error('ImportError: %s', err_msg)
            raise ImportError(err_msg)

    # Training
    if args.do_train:
        global_step, tr_loss = train(args, tasks, model, tokenizer)
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
            result = evaluate(model, tokenizer, prefix=global_step)

            result = {(k + ('_{}'.format(global_step) if global_step else ''), v)
                      for k, v in result.items()}
            results.update(result)

    logger.info("Results: {}".format(results))

    return results
    """


if __name__ == "__main__":
    main()
