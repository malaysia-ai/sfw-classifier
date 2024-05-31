# Safe for Work Classifier

This repository contains code for training a Safe for Work (SFW) classifier on Malaysian data. The classifier is designed to detect various categories including hate speech, violence, self-harm, harassment, informative content, psychiatric or mental illness-related content, racist speech, religion insults, sexist speech, pornographic content, and content that is safe for work.


Finetuned https://huggingface.co/mesolitica/malaysian-mistral-191M-MLM-512 with Malaysian NSFW data.

Refer status:
- https://github.com/orgs/malaysia-ai/projects/6


## A summary

Flowchart: https://excalidraw.com/#json=TlW-2VsgrA8YdUgUx8Sdu,iPDBbixT0_x3udcl91LZ4A

Developing a safe for work classifier involves an extensive amount of data mining. To summarize:

1. Scraped social media data
2. Use LLM to help label data to their respective label 
3. Filter out labelled data by LLM using centroid.
    - generate embedding of labelled data using baai/bge-m3 (this was the best one during this point in time!) 
    - compute centroid of the embedding (average of all the embeddings)
    - filter out based on threshold (heurestically determine the threshold for filtering based on data distribution)

"We are assuming that sentences whose embeddings are closer to the centroid are more similar in a way."

4. We still don't have enough data, or better yet we have too much of it.

5. 
   - For hate/harassment data (we have too much)
    - filter only negative sentiment data 
   - For safe for work data
    - filter only positive and neutral text 

6. For other labels, not enough labels
    - Train a single label classifier model (example: sexist classifier (sexist/non sexist)
    - Classify unlabelled texts
    - Filter out data with lower confidence
    - Heurestically evaluate the prediction
    
  Some labels are more challenging where we have to label it manually or add more dataset:
  
  - Help manually label for labels with insufficient data: https://label-studio.app.mesolitica.com/projects?page=1
  - Add external dataset on sexist/racist exists open source, label our data (translate to english if its an english model)
  - Add to your dataset
  

## LLMOps Pipeline

![Image in a markdown cell](https://github.com/mesolitica/malaysian-llmops/raw/main/e2e.png)


## Notebook & Dataset Guide

1. LLM Inference notebook `malaysian-mistral.ipynb`
2. Notebook on exploration and filtering using centroid for specific labels `filter-{labels}.`
3. Notebook for training single label data for pseudolabeling `pseudolabeling-single-label.ipynb`
4. Notebook for training sfw classifier model `train-sfw-classifier.ipynb`
5. Notebook for fasttext language detection

Important Data Files for Reference

- Dataset containing filtered and training dataset for gathering `malaysia-sfw-dataset.jsonl` 
- Dataset for labelled data from llm inference in inference-data folder `result-mixtral, result-mistral, nsfw-inference-mallam.jsonl` 
- Dataset of social media data with english and standard malay translation `noisy-translation-combined.jsonl`
- Scraped nsfw data using keyword and profiles https://huggingface.co/datasets/malaysia-ai/crawl-twitter-nsfw

## How to Use

We applied LLM2Vec paper to pretrain, pretrained malaysian mistral on masked lm task. The model outperforms other models from t5 and malaysian-mistral trained on causalm lm. We decided to use this model as the base model for this classifier.

Refer file classifier.py to understand the code of the model

```python
from classifier import MistralForSequenceClassification
model = MistralForSequenceClassification.from_pretrained('malaysia-ai/malaysian-sfw-classifier')

```

## Reference

- https://huggingface.co/collections/mesolitica/malaysian-noisy-translation-657e5f88e6759943575a91ac
- https://huggingface.co/collections/mesolitica/malaysian-llm2vec-6628500f8aa9ed3235e4d36c
- https://huggingface.co/datasets/malaysia-ai/crawl-twitter-nsfw

## Acknowledgment

Thank you to all malaysia-ai volunteers for helping to scrape and mining the data to help finish this project :)




