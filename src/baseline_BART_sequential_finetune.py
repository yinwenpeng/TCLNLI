#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
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
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import random
import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset, load_metric

import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.18.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

# A list of all multilingual tokenizer which require lang attribute.
MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast]


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
            "the model's position embeddings."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    lang: str = field(default=None, metadata={"help": "Language id for summarization."})

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge) on "
            "(a jsonlines or csv file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on " "(a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "The token to force as the first generated token after the decoder_start_token_id."
            "Useful for multilingual models like mBART where the first generated token"
            "needs to be the target language token (Usually it is the target language token)"
        },
    )

    def __post_init__(self):
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


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


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if data_args.source_prefix is None and model_args.model_name_or_path in [
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



    # Set seed before initializing model.
    set_seed(training_args.seed)
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.lang)

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                f"Increasing the model's number of position embedding vectors from {model.config.max_position_embeddings} "
                f"to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has {model.config.max_position_embeddings}"
                f" position encodings. Consider either reducing `--max_source_length` to {model.config.max_position_embeddings} or to automatically "
                "resize the model's position encodings by passing `--resize_position_embeddings`."
            )
    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    def preprocess_function(examples):
        # remove pairs where at least one record is None

        inputs, targets = [], []
        for i in range(len(examples[text_column])):
            if examples[text_column][i] is not None and examples[summary_column][i] is not None:
                inputs.append(examples[text_column][i])
                targets.append(examples[summary_column][i])

        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # Metric
    metric = load_metric("rouge")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    unseen_tasks_path = '/home/tup51337/dataset/Natural-Instructions/test_tasks_instruction_into_examples_csv/'
    unseen_task_sequence = ['QG.csv', 'AG.csv', 'CF.csv', 'IAG.csv', 'MM.csv', 'CF.csv']
    for unseen_task in unseen_task_sequence:
        head_task = unseen_task
        test_file = '/home/tup51337/dataset/Natural-Instructions/test_tasks_csv/'+head_task
        subsequent_task_list = [task_i for task_i in unseen_task_sequence if task_i != head_task]
        for repeat_i in range(10):
            random.shuffle(subsequent_task_list)
            task_sequence_for_evolve = [head_task]+subsequent_task_list
            '''continual learning on task_sequence_for_evolve'''
            for evolve_step, train_task_filename in enumerate(task_sequence_for_evolve):

                data_files = {}
                data_files["train"] = unseen_tasks_path+train_task_filename
                if data_args.validation_file is not None:
                    data_files["validation"] = data_args.validation_file
                data_files["test"] = test_file
                extension = 'csv'
                raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)


                # Preprocessing the datasets.
                # We need to tokenize inputs and targets.
                if training_args.do_train:
                    column_names = raw_datasets["train"].column_names
                elif training_args.do_eval:
                    column_names = raw_datasets["validation"].column_names
                elif training_args.do_predict:
                    column_names = raw_datasets["test"].column_names
                else:
                    logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
                    return

                # Get the column names for input/target.
                dataset_columns = summarization_name_mapping.get(data_args.dataset_name, None)
                if data_args.text_column is None:
                    '''the first column as input text'''
                    text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
                else:
                    text_column = data_args.text_column
                    if text_column not in column_names:
                        raise ValueError(
                            f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
                        )
                if data_args.summary_column is None:
                    '''the second column as output text'''
                    summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
                else:
                    summary_column = data_args.summary_column
                    if summary_column not in column_names:
                        raise ValueError(
                            f"--summary_column' value '{data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
                        )

                # Temporarily set max_target_length for training.
                max_target_length = data_args.max_target_length
                padding = "max_length" if data_args.pad_to_max_length else False

                if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
                    logger.warning(
                        "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
                        f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
                    )

                if training_args.do_train:
                    if "train" not in raw_datasets:
                        raise ValueError("--do_train requires a train dataset")
                    train_dataset = raw_datasets["train"]
                    if data_args.max_train_samples is not None:
                        train_dataset = train_dataset.select(range(data_args.max_train_samples))
                    with training_args.main_process_first(desc="train dataset map pre-processing"):
                        train_dataset = train_dataset.map(
                            preprocess_function,
                            batched=True,
                            num_proc=data_args.preprocessing_num_workers,
                            remove_columns=column_names,
                            load_from_cache_file=not data_args.overwrite_cache,
                            desc="Running tokenizer on train dataset",
                        )

                if training_args.do_eval:
                    max_target_length = data_args.val_max_target_length
                    if "validation" not in raw_datasets:
                        raise ValueError("--do_eval requires a validation dataset")
                    eval_dataset = raw_datasets["validation"]
                    if data_args.max_eval_samples is not None:
                        eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
                    with training_args.main_process_first(desc="validation dataset map pre-processing"):
                        eval_dataset = eval_dataset.map(
                            preprocess_function,
                            batched=True,
                            num_proc=data_args.preprocessing_num_workers,
                            remove_columns=column_names,
                            load_from_cache_file=not data_args.overwrite_cache,
                            desc="Running tokenizer on validation dataset",
                        )

                if training_args.do_predict:
                    max_target_length = data_args.val_max_target_length
                    if "test" not in raw_datasets:
                        raise ValueError("--do_predict requires a test dataset")
                    predict_dataset = raw_datasets["test"]
                    if data_args.max_predict_samples is not None:
                        predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
                    with training_args.main_process_first(desc="prediction dataset map pre-processing"):
                        predict_dataset = predict_dataset.map(
                            preprocess_function,
                            batched=True,
                            num_proc=data_args.preprocessing_num_workers,
                            remove_columns=column_names,
                            load_from_cache_file=not data_args.overwrite_cache,
                            desc="Running tokenizer on prediction dataset",
                        )



                # Initialize our Trainer
                trainer = Seq2SeqTrainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset if training_args.do_train else None,
                    eval_dataset=eval_dataset if training_args.do_eval else None,
                    tokenizer=tokenizer,
                    data_collator=data_collator,
                    compute_metrics=compute_metrics if training_args.predict_with_generate else None,
                )

                # Training
                if training_args.do_train:
                    '''actual training epochs'''
                    if evolve_step == 0:
                        checkpoint = training_args.resume_from_checkpoint
                    else:
                        checkpoint = get_last_checkpoint(training_args.output_dir)

                    train_result = trainer.train(resume_from_checkpoint=checkpoint)
                    trainer.save_model()  # Saves the tokenizer too for easy upload

                    metrics = train_result.metrics
                    max_train_samples = (
                        data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
                    )
                    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

                    trainer.log_metrics("train", metrics)
                    trainer.save_metrics("train", metrics)
                    trainer.save_state()

                # Evaluation
                results = {}
                max_length = (
                    training_args.generation_max_length
                    if training_args.generation_max_length is not None
                    else data_args.val_max_target_length
                )
                num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
                if training_args.do_eval:
                    logger.info("*** Evaluate ***")
                    metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
                    max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
                    metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

                    trainer.log_metrics("eval", metrics)
                    trainer.save_metrics("eval", metrics)

                if training_args.do_predict:
                    logger.info("*** Predict ***")

                    predict_results = trainer.predict(
                        predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
                    )
                    metrics = predict_results.metrics
                    max_predict_samples = (
                        data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
                    )
                    metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

                    trainer.log_metrics("predict", metrics)
                    trainer.save_metrics("predict", metrics)

                    if trainer.is_world_process_zero():
                        if training_args.predict_with_generate:
                            predictions = tokenizer.batch_decode(
                                predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                            )
                            predictions = [pred.strip() for pred in predictions]
                            output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                            with open(output_prediction_file, "w") as writer:
                                writer.write("\n".join(predictions))

                kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "summarization"}
                if data_args.dataset_name is not None:
                    kwargs["dataset_tags"] = data_args.dataset_name
                    if data_args.dataset_config_name is not None:
                        kwargs["dataset_args"] = data_args.dataset_config_name
                        kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
                    else:
                        kwargs["dataset"] = data_args.dataset_name

                if data_args.lang is not None:
                    kwargs["language"] = data_args.lang

                if training_args.push_to_hub:
                    trainer.push_to_hub(**kwargs)
                else:
                    trainer.create_model_card(**kwargs)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()


'''
CUDA_VISIBLE_DEVICES=2 python -u baseline_BART_sequential_finetune.py --model_name_or_path facebook/bart-base --resume_from_checkpoint /home/tup51337/tmp/pretrained_BART_on_paper_tasks --do_train --train_file /home/tup51337/dataset/Natural-Instructions/all_training_tasks_in_single_csv.csv --do_predict --max_source_length 1024 --output_dir /home/tup51337/tmp/tmp --per_device_train_batch_size=2 --per_device_eval_batch_size=2 --overwrite_output_dir --predict_with_generate --num_train_epochs 3.0 --learning_rate 5e-5 --save_strategy epoch
'''
