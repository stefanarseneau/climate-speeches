# https://huggingface.co/climatebert/distilroberta-base-climate-commitment
# https://huggingface.co/climatebert/distilroberta-base-climate-detector
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from transformers.pipelines.pt_utils import KeyDataset
import datasets
from tqdm.auto import tqdm

import dataloader as dl
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', nargs='?', default=None)
    parser.add_argument('--savefile', action='store_true')
    args = parser.parse_args()

    n = int(args.n)
    savefile = args.savefile

    dataset_name = "climatebert/climate_commitments_actions"
    model_name = "climatebert/distilroberta-base-climate-commitment"

    # If you want to use your own data, simply load them as 🤗 Datasets dataset, see https://huggingface.co/docs/datasets/loading
    #dataset = datasets.load_dataset(dataset_name, split="test")
    datafr = dl.DataLoader(n, savefile)
    dataset_hf = datafr.speechdata_hf
    dataset_pd = datafr.speechdata_pd

    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, max_len=512)

    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0)

    # See https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.pipeline
    labels, scores = [], []
    for out in tqdm(pipe(KeyDataset(dataset_hf, "paragraph"), padding=True, truncation=True)):
       labels.append(out['label'])
       scores.append(out['score'])

    dataset_pd['climatebert_labels'] = labels
    dataset_pd['climatebert_scores'] = scores
    dataset_pd.to_csv('test.csv')
