# https://huggingface.co/climatebert/distilroberta-base-climate-commitment
# https://huggingface.co/climatebert/distilroberta-base-climate-detector
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from transformers.pipelines.pt_utils import KeyDataset
import datasets
from tqdm.auto import tqdm
import numpy as np

import dataloader as dl
import argparse

def summation_score(pd_dataset, weight):
   classifier = 2*np.all([pd_dataset['label'] == 'yes'], axis = 0).astype(int) - 1
   classifier[classifier > 0] *= weight
   score = sum(classifier * pd_dataset['score']) / len(pd_dataset)
   return score

class ClimateBert:
   def __init__(self, max_len: int = 512):
      self.dataset_name = "climatebert/climate_commitments_actions"
      self.model_name = "climatebert/distilroberta-base-climate-commitment"

      self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
      self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, max_len=max_len)
      self.pipe = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer, device=0)

   def __call__(self, dataset):
      labels, scores = [], []
      for out in tqdm(self.pipe(KeyDataset(dataset, "paragraph"), padding=True, truncation=True)):
         labels.append(out['label'])
         scores.append(out['score'])
      return labels, scores
   
        

if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument('--sentence_chunking', nargs='?', default=None)
   parser.add_argument('--savefile', action='store_true')
   args = parser.parse_args()

   climate_speeches = dl.load_zipfile()
    
   # If you want to use your own data, simply load them as 🤗 Datasets dataset, see https://huggingface.co/docs/datasets/loading
   #dataset = datasets.load_dataset(dataset_name, split="test")
   datafr = dl.DataLoader('r221222a', sentence_chunking = int(args.sentence_chunking))
   dataset_hf = datafr.speechdata_hf
   dataset_pd = datafr.speechdata_pd

   CB_Classifier = ClimateBert()

   # See https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.pipeline
   labels, scores = CB_Classifier(dataset_hf)
   dataset_pd['label'] = labels
   dataset_pd['score'] = scores 

   score = summation_score(dataset_pd, weight = 3)
   print('final score:', score)

   dataset_pd.to_csv('test.csv')
