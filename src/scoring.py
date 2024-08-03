import dataloader as dl
import climatebert as cb

from urllib.parse import urlparse
import pandas as pd
import argparse
import os

def fetch_dataset(dataset_name):
    if dataset_name == 'identified':
        dataset = dl.load_zipfile()
    else:
        loader = dl.DataLoader()
        dataset = loader.search_index
        dataset['id'] = [os.path.basename(urlparse(url).path).split('.')[0] for url in dataset['url']]
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', nargs='?')
    parser.add_argument('--sentence-chunking', nargs='?', default=None)
    parser.add_argument('--score-weighting', nargs='?', default='1')
    args = parser.parse_args()

    try:
        assert args.dataset in ['identified', 'all']
    except AssertionError as e:
        print("Please specify a supported dataset: 'identified' or 'all'!")
        raise

    dataset = fetch_dataset(args.dataset)
    ids = dataset['id']

    print(f'number of ids in dataset: {len(ids)}')

    ids, scores = cb.classify_speeches(ids, int(args.sentence_chunking), int(args.score_weighting))
    
    print(len(ids), len(scores))

    dataset['climatebert_scores'] = scores
    dataset.to_csv(args.dataset + '.csv')
