import argparse
import os

import torch
from datasets import Audio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor

from dataloader.convert_kaldi_data import get_dataset
from training.finetune_with_hg import replace_hatted_characters, remove_special_characters, prepare_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, required=True, help='model directory')
    parser.add_argument('--test', type=str, required=True, help='test csv file')
    parser.add_argument('--num_proc', type=int, required=True, help='num process counts')
    parser.add_argument('--vocab', type=str, required=True, help='vocab json file')
    parser.add_argument('--out_file', type=str, required=True, help='out file for wer information')

    args = parser.parse_args()
    model_dir = args.model
    vocab_file = args.vocab
    test_file = args.test
    out_file = args.out_file
    num_process = args.num_proc

    model = Wav2Vec2ForCTC.from_pretrained(model_dir)
    tokenizer = Wav2Vec2CTCTokenizer(vocab_file, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                                 do_normalize=True, return_attention_mask=True)

    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    print('creating test dataset')
    test_dataset = get_dataset(test_file)
    test_dataset = test_dataset.map(replace_hatted_characters)
    test_dataset = test_dataset.map(remove_special_characters)
    test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16_000))

    test_dataset = test_dataset.map(prepare_dataset, remove_columns=test_dataset.column_names,
                                    num_proc=num_process, keep_in_memory=True)

    parent_dir = os.path.dirname(out_file)
    if not os.path.exists(parent_dir):
        os.mkdir(parent_dir)

    f_o = open(out_file, 'w', encoding='utf-8')
    for x in range(len(test_dataset)):
        input_dict = processor(test_dataset[x]["input_values"], return_tensors="pt", padding=True)

        logits = model(input_dict.input_values).logits

        pred_ids = torch.argmax(logits, dim=-1)[0]

        transcript = processor(test_dataset[x]["labels"])

        f_o.write("\n" + "Prediction:" + processor.decode(pred_ids) + "\n" + "Reference:" +
                  transcript.lower())
