#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
"""
Fine-tuning a 🤗 Transformers model on summarization.
"""
# You can also adapt this script on your own summarization task. Pointers for this are left as comments.

import argparse
import logging
import math
import os
import random
from pathlib import Path
from tqdm import tqdm, trange
import datasets
import nltk
import numpy as np
import torch
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import statistics
import transformers
from accelerate import Accelerator
from filelock import FileLock
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
    set_seed,
)
from transformers.file_utils import get_full_repo_name, is_offline_mode
from transformers.utils.versions import require_version
from compute_mean_std import compute_for_dict, computer_mean_std_given_list
from load_tasks import load_task_list
from collections import defaultdict

logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to " "padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=1024,
        help="The maximum total input sequence length after "
        "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--eval_truncate",
        type=int,
        default=1000,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--source_prefix",
        type=str,
        default=None,
        help="A prefix to add before every source text " "(useful for T5 models).",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=None, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help="The maximum total sequence length for target text after "
        "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        "during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--val_max_target_length",
        type=int,
        default=None,
        help="The maximum total sequence length for validation "
        "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
        "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
        "param of ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help="Number of beams to use for evaluation. This argument will be "
        "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--repeat_times",
        type=int,
        default=5,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the full texts (for summarization).",
    )
    parser.add_argument(
        "--summary_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the summaries (for summarization).",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--per_device_base_train_batch_size",
        type=int,
        default=5,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument(
        "--forward_distance",
        type=int,
        default=10,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--training_size",
        type=int,
        default=5,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.source_prefix is None and args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # os.environ["LOCAL_RANK"]=str(2)
    accelerator = Accelerator()
    # print('accelerator.state:', accelerator.state)
    #
    # print(os.environ.get("LOCAL_RANK", -1))
    # exit(0)
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()




    '''train on base tasks and save model'''
    all_task_example_path = '/home/tup51337/dataset/Natural-Instructions/TCLNLI_split/all_task_examples_in_CSV/'
    unseen_tasks_pos_path = '/home/tup51337/dataset/Natural-Instructions/TCLNLI_split/all_task_pos_instruction_examples_in_CSV/'
    unseen_tasks_neg_path = '/home/tup51337/dataset/Natural-Instructions/TCLNLI_split/all_task_neg_instruction_examples_in_CSV/'
    all_task_list = load_task_list()
    delta_performance = []
    for _ in range(args.repeat_times):
        base_tasks = random.sample(all_task_list, args.training_size)
        unseen_tasks = [  task_i for task_i in all_task_list if task_i not in base_tasks]
        # print('Base tasks: ', base_tasks)
        '''first prepare a fresh model and tokenizer'''
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
        model = AutoModelForSeq2SeqLM.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config)

        model.resize_token_embeddings(len(tokenizer))
        if model.config.decoder_start_token_id is None:
            raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
        model, optimizer = accelerator.prepare(model, optimizer)

        '''then, start to prepare data'''

        data_files = {}
        data_files["train"] = [all_task_example_path+task_i+'.csv' for task_i in base_tasks] #base_tasks
        # data_files["validation"] = test_file
        raw_datasets = load_dataset("csv", data_files=data_files)
        column_names = raw_datasets["train"].column_names
        text_column = column_names[0]
        summary_column = column_names[1]

        # Temporarily set max_target_length for training.
        max_target_length = args.max_target_length
        padding = "max_length" if args.pad_to_max_length else False

        def preprocess_function(examples):
            '''tokenize, padding'''
            inputs = examples[text_column]
            targets = examples[summary_column]
            '''avoid NoneType in target'''
            targets = [inp  if inp is not None else "none" for inp in targets]
            model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)

            # Setup the tokenizer for targets
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

            # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
            # padding in the loss.
            if padding == "max_length" and args.ignore_pad_token_for_loss:
                labels["input_ids"] = [
                    [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                ]

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        with accelerator.main_process_first():
            tokenized_dataset = raw_datasets.map(
                preprocess_function,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        base_dataset = tokenized_dataset["train"]

        label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if accelerator.use_fp16 else None,
        )

        base_dataloader = DataLoader(base_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_base_train_batch_size)
        base_dataloader = accelerator.prepare(base_dataloader)
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.num_train_epochs*len(base_dataloader),
        )


        logger.info("***** Running base training *****")
        print('Base tasks: ', base_tasks)
        logger.info(f"  Num examples = {len(base_dataset)}")

        # for epoch in range(args.num_train_epochs):
        for epoch in trange(args.num_train_epochs, desc="train_epochs"):
            model.train()
            for step, batch in enumerate(tqdm(base_dataloader, desc="BaseTraining")):
                outputs = model(**batch)
                loss = outputs.loss
                loss = loss / args.gradient_accumulation_steps
                accelerator.backward(loss)
                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

        store_model(accelerator, model, args.output_dir, tokenizer)

        print('\nBase training over, Start evolution part....\n')
        for _ in range(args.repeat_times):


            '''then, start to prepare data, one short sequence, one long sequence'''
            random.shuffle(unseen_tasks)
            # target_task_id = random.randint(40, len(all_task_list)-args.training_size)
            target_task_id = random.randint(0, len(all_task_list)-args.training_size-40)
            print('\ntarget_task_id: ', target_task_id, '\n')
            target_task = unseen_tasks[target_task_id]
            short_task_sequence = unseen_tasks[:target_task_id+1] # include target as the tail task
            long_task_sequence = unseen_tasks[:target_task_id]+ unseen_tasks[target_task_id+1:target_task_id+1+args.forward_distance]+[target_task]
            assert len(long_task_sequence) -len(short_task_sequence) == args.forward_distance





            '''continual learning on task_sequence_for_evolve'''
            pair_performance = []
            for each_unseen_tasks in [short_task_sequence, long_task_sequence]:
                '''first load pretrained model on base tasks'''
                config = AutoConfig.from_pretrained(args.output_dir)
                tokenizer = AutoTokenizer.from_pretrained(args.output_dir, use_fast=not args.use_slow_tokenizer)
                model = AutoModelForSeq2SeqLM.from_pretrained(
                        args.output_dir,
                        from_tf=bool(".ckpt" in args.output_dir),
                        config=config)

                model.resize_token_embeddings(len(tokenizer))
                if model.config.decoder_start_token_id is None:
                    raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
                no_decay = ["bias", "LayerNorm.weight"]
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                        "weight_decay": args.weight_decay,
                    },
                    {
                        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                        "weight_decay": 0.0,
                    },
                ]
                optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
                metric = load_metric("rouge")
                total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
                model, optimizer = accelerator.prepare(model, optimizer)

                target_task_filename = all_task_example_path+target_task+'.csv'
                target_raw_dataset = load_dataset("csv", data_files={'target':target_task_filename})
                column_names = target_raw_dataset["target"].column_names
                def preprocess_function(examples):
                    '''tokenize, padding'''
                    inputs = examples["input"]
                    targets = examples["output"]
                    '''avoid NoneType in target'''
                    targets = [inp  if inp is not None else "none" for inp in targets]
                    model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)

                    # Setup the tokenizer for targets
                    with tokenizer.as_target_tokenizer():
                        labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

                    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
                    # padding in the loss.
                    if padding == "max_length" and args.ignore_pad_token_for_loss:
                        labels["input_ids"] = [
                            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                        ]

                    model_inputs["labels"] = labels["input_ids"]
                    return model_inputs

                with accelerator.main_process_first():
                    tokenized_target_dataset = target_raw_dataset.map(
                        preprocess_function,
                        batched=True,
                        num_proc=args.preprocessing_num_workers,
                        remove_columns=column_names,
                        load_from_cache_file=not args.overwrite_cache,
                        desc="Running tokenizer on target dataset",
                    )
                target_dataset = tokenized_target_dataset["target"]
                if args.eval_truncate and args.eval_truncate< len(target_dataset):
                    target_dataset = target_dataset.select(random.sample(range(0, len(target_dataset)), args.eval_truncate))


                label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
                data_collator = DataCollatorForSeq2Seq(
                    tokenizer,
                    model=model,
                    label_pad_token_id=label_pad_token_id,
                    pad_to_multiple_of=8 if accelerator.use_fp16 else None,
                )

                target_dataloader = DataLoader(target_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
                target_dataloader = accelerator.prepare(target_dataloader)

                for evolve_step, new_task_filename in enumerate(each_unseen_tasks):
                    raw_datasets = load_dataset("csv", data_files={'train':unseen_tasks_pos_path+new_task_filename+'.csv'})
                    column_names = raw_datasets["train"].column_names
                    text_column = column_names[0]
                    summary_column = column_names[1]
                    max_target_length = args.max_target_length
                    padding = "max_length" if args.pad_to_max_length else False
                    with accelerator.main_process_first():
                        tokenized_dataset = raw_datasets.map(
                            preprocess_function,
                            batched=True,
                            num_proc=args.preprocessing_num_workers,
                            remove_columns=column_names,
                            load_from_cache_file=not args.overwrite_cache,
                            desc="Running tokenizer on dataset",
                        )
                    train_dataset = tokenized_dataset["train"]

                    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)
                    train_dataloader = accelerator.prepare(train_dataloader)
                    lr_scheduler = get_scheduler(
                        name=args.lr_scheduler_type,
                        optimizer=optimizer,
                        num_warmup_steps=args.num_warmup_steps,
                        num_training_steps=args.num_train_epochs*len(train_dataloader),
                    )


                    logger.info("***** Running  new task training *****")
                    print('new_task_filename:', new_task_filename)
                    logger.info(f"  Num examples = {len(train_dataset)}")
                    logger.info(f"  Num Epochs = {args.num_train_epochs}")
                    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
                    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
                    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")


                    # for epoch in range(args.num_train_epochs):
                    for epoch in trange(args.num_train_epochs, desc="train_epochs"):
                        model.train()
                        for step, batch in enumerate(tqdm(train_dataloader, desc="NewTaskTraining")):
                            outputs = model(**batch)
                            loss = outputs.loss
                            loss = loss / args.gradient_accumulation_steps
                            accelerator.backward(loss)
                            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                                optimizer.step()
                                lr_scheduler.step()
                                optimizer.zero_grad()
                '''evaluate on target task after this sequence'''
                model.eval()
                if args.val_max_target_length is None:
                    args.val_max_target_length = args.max_target_length

                gen_kwargs = {
                    "max_length": args.val_max_target_length if args is not None else config.max_length,
                    "num_beams": args.num_beams,
                }

                for step, batch in enumerate(tqdm(target_dataloader, desc="Evaluating")):
                    with torch.no_grad():
                        generated_tokens = accelerator.unwrap_model(model).generate(
                            batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            **gen_kwargs,
                        )

                        generated_tokens = accelerator.pad_across_processes(
                            generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                        )
                        labels = batch["labels"]
                        if not args.pad_to_max_length:
                            # If we did not pad to max length, we need to pad the labels too
                            labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

                        generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                        labels = accelerator.gather(labels).cpu().numpy()

                        if args.ignore_pad_token_for_loss:
                            # Replace -100 in the labels as we can't decode them.
                            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                        if isinstance(generated_tokens, tuple):
                            generated_tokens = generated_tokens[0]
                        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

                        metric.add_batch(predictions=decoded_preds, references=decoded_labels)
                result = metric.compute(use_stemmer=True)
                # Extract a few results from ROUGE
                result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
                result = {k: round(v, 4) for k, v in result.items()}
                rouge_L = result["rougeL"]
                pair_performance.append(rouge_L)
            delta_performance.append(pair_performance[1]-pair_performance[0])

    final_result = computer_mean_std_given_list(delta_performance)
    print('Final performance: ', final_result)

def store_model(accele, model, output_dir, tokenizer):
    accele.wait_for_everyone()
    unwrapped_model = accele.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accele.save)
    if accele.is_main_process:
        tokenizer.save_pretrained(output_dir)
    print('Model saved.')

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def computer_mean_std(value_list):
    average = round(sum(value_list)/len(value_list), 2)
    res = round(statistics.pstdev(value_list),2)
    return str(average)+'/'+str(res)


if __name__ == "__main__":
    main()


'''

"sequential finetune on instructions"

CUDA_VISIBLE_DEVICES=0 python -u baseline.seq.finetune.forward.py --model_name_or_path facebook/bart-base --output_dir /home/tup51337/tmp/tmp4 --max_source_length 1024 --per_device_base_train_batch_size=5 --per_device_train_batch_size=2 --per_device_eval_batch_size=24 --num_train_epochs 1 --learning_rate 5e-5 --training_size 1 --eval_truncate 100 --repeat_times 1 --forward_distance 1 


'''
