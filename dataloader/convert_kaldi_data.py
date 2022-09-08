import argparse
import csv
import json
import os

import pandas as pd
from datasets import Dataset
from unicode_tr import unicode_tr

'''
This code converts kaldi directory to wav2vec2 training csv format.
path transcript
audio.wav merhaba dünya
'''


class KaldiDataUnit:
    def __init__(self, audio_id, audio_path, text):
        self.audio_id = audio_id
        self.audio_path = audio_path
        self.text = text

    @staticmethod
    def load_kaldi_data(data_dir):
        f_w = open(os.path.join(data_dir, 'wav.scp'), 'r', encoding='utf-8')
        audio_dict = {}
        for line in f_w:
            tokens = line.split(' ', 1)
            audio_dict[tokens[0]] = tokens[1]
        kaldi_units = []
        f_t = open(os.path.join(data_dir, 'text'), 'r', encoding='utf-8')
        text_dict = {}

        for line in f_t:
            tokens = line.split(' ', 1)
            text_dict[tokens[0]] = tokens[1]
        count = 0
        for key, value in audio_dict.items():
            audio_path = value.strip()
            transcript = unicode_tr(text_dict[key].strip()).lower()
            kaldi_unit = KaldiDataUnit(key, audio_path, transcript)
            count += 1
            kaldi_units.append(kaldi_unit)

        return kaldi_units


def save_kaldi_data_as_csv(kaldi_dir, out_file):
    kaldi_units = KaldiDataUnit.load_kaldi_data(kaldi_dir)
    parent_dir = os.path.dirname(out_file)
    if not os.path.exists(parent_dir):
        os.mkdir(parent_dir)
    f_o = open(out_file, 'w', encoding='utf-8', newline='')
    writer = csv.writer(f_o)
    header = ['path', 'audio', 'sentence']
    writer.writerow(header)
    for unit in kaldi_units:
        writer.writerow([unit.audio_path, unit.audio_path, unit.text])


def get_dataset(csv_file):
    train_df = pd.read_csv(csv_file)
    return Dataset.from_pandas(train_df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--kaldi_dir', type=str, required=True, help='kaldi directory')
    parser.add_argument('--out_file', type=str, required=True, help='data file')


    args = parser.parse_args()
    kaldi_dir = args.kaldi_dir
    out_file = args.out_file



    save_kaldi_data_as_csv(kaldi_dir, out_file)
