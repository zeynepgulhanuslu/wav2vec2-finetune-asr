# Load Model From Checkpoint
import argparse

import flash
from flash.audio import SpeechRecognition, SpeechRecognitionData

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='model file')
    parser.add_argument('--test_file', type=str, required=True, help='test wav file')
    parser.add_argument('--gpus', type=int, required=True, help='gpu count')
    args = parser.parse_args()
    model_file = args.model
    test_file = args.test_file
    gpu = args.gpus
    model = SpeechRecognition.load_from_checkpoint(model_file)

    datamodule = SpeechRecognitionData.from_files(predict_files=[test_file], batch_size=1)
    predictions = flash.Trainer(gpus=gpu).predict(model, datamodule=datamodule)
    print(predictions)
