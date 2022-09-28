import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

# import evaluate
import numpy as np
import torch
from datasets import Audio, load_from_disk, load_metric
from transformers import Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC, TrainingArguments, Trainer, \
    Wav2Vec2FeatureExtractor

from dataloader.convert_kaldi_data import get_dataset

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�]'


def remove_special_characters(batch):
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower() + " "
    return batch


def extract_all_chars(batch):
    all_text = " ".join(batch["sentence"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


def save_vocab(train_dataset, test_dataset, vocab_file):
    parent_dir = os.path.dirname(vocab_file)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    vocab_train = train_dataset.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True,
                                    remove_columns=train_dataset.column_names)
    vocab_test = test_dataset.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True,
                                  remove_columns=test_dataset.column_names)

    vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))

    vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    len(vocab_dict)

    with open(vocab_file, 'w', encoding='utf-8') as vocab_file:
        json.dump(vocab_dict, vocab_file)


def replace_hatted_characters(batch):
    batch["sentence"] = re.sub('[â]', 'a', batch["sentence"])
    batch["sentence"] = re.sub('[î]', 'i', batch["sentence"])
    batch["sentence"] = re.sub('[ô]', 'o', batch["sentence"])
    batch["sentence"] = re.sub('[û]', 'u', batch["sentence"])
    batch["sentence"] = re.sub('[é]', 'e', batch["sentence"])
    batch["sentence"] = re.sub('[é]', 'e', batch["sentence"])
    return batch


def prepare_dataset(batch):
    audio = batch["audio"]
    # batched output is "un-batched"
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids
    return batch


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', type=str, required=True, help='train csv data file')
    parser.add_argument('--test', type=str, required=True, help='test csv data file')
    parser.add_argument('--vocab', type=str, required=True, help='vocab json file')
    parser.add_argument('--num_proc', type=int, required=True, help='num process counts')
    parser.add_argument('--out_dir', type=str, required=True, help='output directory')

    args = parser.parse_args()
    train_file = args.train
    test_file = args.test
    vocab_file = args.vocab

    num_process = args.num_proc
    out_dir = args.out_dir

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print('creating test dataset')
    test_dataset = get_dataset(test_file)
    test_dataset = test_dataset.map(replace_hatted_characters)
    test_dataset = test_dataset.map(remove_special_characters)
    test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16_000))

    print('creating train dataset')
    train_dataset = get_dataset(train_file)
    train_dataset = train_dataset.map(replace_hatted_characters)
    train_dataset = train_dataset.map(remove_special_characters)
    # read audio file
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16_000))

    print('done creating datasets')
    if not os.path.exists(vocab_file):
        save_vocab(train_dataset, test_dataset, vocab_file)
        print('vocab file saved')

    tokenizer = Wav2Vec2CTCTokenizer(vocab_file, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                                 do_normalize=True, return_attention_mask=True)

    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    test_data_dir = os.path.join(out_dir, 'test-data')
    train_data_dir = os.path.join(out_dir, 'train-data')

    if not os.path.exists(test_data_dir):
        print('preparing test dataset as batches')
        test_dataset = test_dataset.map(prepare_dataset, remove_columns=test_dataset.column_names,
                                        num_proc=num_process, keep_in_memory=True)

        test_dataset.save_to_disk(test_data_dir)
    else:
        print('loading test dataset as batches')
        test_dataset = load_from_disk(test_data_dir)
    if not os.path.exists(train_data_dir):
        print('preparing train dataset as batches')
        train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names,
                                          num_proc=num_process, keep_in_memory=True)

        train_dataset.save_to_disk(train_data_dir)
    else:
        print('loading train dataset as batches')
        train_dataset = load_from_disk(train_data_dir)

    print('batch dataset completed.')

    # wer_metric = evaluate.load("wer")
    wer_metric = load_metric("wer")
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    print('initialize multi-languages model')

    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-large-xlsr-53",
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.1,
        ctc_loss_reduction="mean",
        pad_token_id=tokenizer.pad_token_id,
        vocab_size=len(tokenizer)
    )
    model.freeze_feature_extractor()

    model.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        output_dir=out_dir,
        group_by_length=False,
        auto_find_batch_size=True,
        per_device_train_batch_size=16,
        evaluation_strategy="steps",
        num_train_epochs=30,
        fp16=True,
        gradient_checkpointing=True,
        save_steps=500,
        eval_steps=500,
        logging_steps=500,
        learning_rate=1e-4,
        weight_decay=0.005,
        warmup_steps=1000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
    )
    print('training started')
    torch.cuda.empty_cache()

    if last_checkpoint is not None:
        checkpoint = last_checkpoint
    elif os.path.isdir(model_args.model_name_or_path):
        checkpoint = model_args.model_name_or_path
    else:
        checkpoint = None

    train_result = trainer.train()

    print('training finished')
    train_metrics = train_result.metrics
    train_metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", split=',', metrics=train_metrics)
    trainer.save_metrics("train", split=',', metrics=train_metrics)
    trainer.save_state()
