from accelerate import Accelerator, InitProcessGroupKwargs
from torch.utils.data import DataLoader
from datetime import datetime
from dataclasses import dataclass
from datasets import Dataset
from glob import glob
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
import pandas as pd
import numpy as np
import json
import os
import soundfile as sf
from datetime import timedelta
from tqdm import tqdm
from transformers import (
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
import os
import torch
from classifier import MistralForSequenceClassification

candidate_labels = ['hate',
                        'violence',
                        'self-harm',
                        'harassment',
                        'informative',
                        'lgbt',
                        'psychiatric or mental illness',
                        'racist',
                        'religion insult',
                        'sexist',
                        'porn',
                        'safe for work']

@dataclass
class ModelArguments:
    model_name_or_path: str = 'porn-mistral-mlm'
    indices_filename: 'str' = 'indices-alignment-classifier-2.json'
    folder_output: str = 'output'
    folder_output_inference: str = 'output-alignment-labelled'
    batch_size: int = 50
    dataloader_num_workers: int = 3


def main():
    global_rank = 0
    parser = HfArgumentParser(ModelArguments)
    model_args = parser.parse_args_into_dataclasses()[0]
    print(global_rank, model_args)

    os.makedirs(model_args.folder_output, exist_ok=True)
    os.makedirs(model_args.folder_output_inference, exist_ok=True)

    id2label = {0:'non porn',1:'porn'}
    label2id = {'non porn':0,'porn':1}


    torch_dtype = torch.float16
    mixed_precision = 'fp16'
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        kwargs_handlers=[kwargs],
    )
    tokenizer = AutoTokenizer.from_pretrained("malaysia-ai/sentiment-mistral-191M-MLM")

    model = MistralForSequenceClassification.from_pretrained(
        model_args.model_name_or_path, 
        torch_dtype = torch_dtype,
        device_map="cuda"
    )

    model.config.label2id = label2id
    model.config.id2label = id2label

    model.eval()
    _ = model.cuda()

    pipe = pipeline("text-classification",
                        tokenizer = tokenizer,
                        model=model)
    
    print(global_rank, model.dtype, model.device)


    start_step = 0
    steps = glob(os.path.join(model_args.folder_output, f'{global_rank}-*.json'))
    steps = [int(f.split('-')[1].replace('.json', '')) for f in steps]
    if len(steps):
        start_step = max(steps) + 1
        print(f'{global_rank}, continue from {start_step}')
    else:
        print(f'{global_rank}, failed to load last step count, continue from 0')


    data = []

    with open('nsfw-tweets-en-ms.jsonl') as f:
        for x in tqdm(f):
            data_ = json.loads(x)
            if len(data_['Content']) > 5:
                data.append(json.loads(x))

    result = {
        'Content': [str(x['Content']) for x in data],
        'label_fasttext': [str(x['label']) for x in data],  # Assuming 'left' is a string column
        'score_fasttext': [str(x['score']) for x in data],
    }


    train = Dataset.from_dict(result)


    for idx,out in tqdm(enumerate(pipe(KeyDataset(train,'Content'), batch_size=50, truncation="longest_first")),total = len(train)):
        result = train[idx]
        result['label'] = out['label']
        result['score'] = out['score']

        with open('porn-nsfw-tweet.jsonl', 'a') as fopen:
            json.dump(result, fopen)
            fopen.write('\n')
     



if __name__ == "__main__":
    main()
