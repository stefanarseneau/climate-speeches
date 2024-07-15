# https://huggingface.co/climatebert/distilroberta-base-climate-commitment
# https://huggingface.co/climatebert/distilroberta-base-climate-detector
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from transformers.pipelines.pt_utils import KeyDataset
import datasets
from tqdm.auto import tqdm

import argparse
import cleandata

dataset_name = "climatebert/climate_commitments_actions"
model_name = "climatebert/distilroberta-base-climate-commitment"

# If you want to use your own data, simply load them as 🤗 Datasets dataset, see https://huggingface.co/docs/datasets/loading
dataset = datasets.load_dataset(dataset_name, split="test")

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, max_len=512)

pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0)

# See https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.pipeline
for out in tqdm(pipe(KeyDataset(dataset, "text"), padding=True, truncation=True)):
   print(out)
