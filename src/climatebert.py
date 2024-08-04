# https://huggingface.co/climatebert/distilroberta-base-climate-commitment
# https://huggingface.co/climatebert/distilroberta-base-climate-detector
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from transformers.pipelines.pt_utils import KeyDataset
import datasets
from tqdm.auto import tqdm
import numpy as np

import dataloader as dl
import argparse

class ClimateBert:
   def __init__(self, max_len: int = 512):
      self.dataset_name = "climatebert/climate_detection"
      self.model_name = "climatebert/distilroberta-base-climate-detector"

      self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
      self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, max_len=max_len)

      # See https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.pipeline
      self.pipe = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer, device='cpu')

   def __call__(self, dataset):
      labels, scores = [], []
      for out in self.pipe(KeyDataset(dataset, "paragraph"), padding=True, truncation=True):
         labels.append(out['label'])
         scores.append(out['score'])
      return labels, scores
   
def summation_score(pd_dataset, weight):
   classifier = 2*np.all([pd_dataset['label'] == 'yes'], axis = 0).astype(int) - 1
   classifier *= pd_dataset['score']
   raw_scores = classifier.copy()
   classifier[classifier > 0] *= weight
   score = sum(classifier ) / len(pd_dataset)
   return score, raw_scores

def classify_speeches(ids, sentence_chunking, score_weighting):
   final_scores = []

   dat = dl.DataLoader()
   CB_Classifier = ClimateBert()
   parameters = {}

   for i, id in enumerate(tqdm(ids)):
      url, title, description, author, date, text, dataset_pd, dataset_hf = dat.fetch_text(id, sentence_chunking = int(sentence_chunking))

      if dataset_hf != '':
          labels, scores = CB_Classifier(dataset_hf)
          dataset_pd['label'] = labels
          dataset_pd['score'] = scores 

          score, raw_scores = summation_score(dataset_pd, weight = int(score_weighting))
          parameters[id] = raw_scores
          final_scores.append(score)
      else:
          final_scores.append(-999)
          parameters[id] = []

   return ids, final_scores, parameters

if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument('id', nargs='?')
   parser.add_argument('--sentence_chunking', nargs='?', default=None)
   parser.add_argument('--score-weighting', nargs='?', default='1')
   args = parser.parse_args()

   scores = classify_speeches([args.id], int(args.sentence_chunking), int(args.score_weighting))
   print('speech', args.id, 'final score:', scores[0])
   

   

   
   
