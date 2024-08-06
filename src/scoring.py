import dataloader as dl
import climatebert as cb

from urllib.parse import urlparse
import pickle 
import pandas as pd
import argparse
import os

def fetch_dataset(dataset_name):
    if os.path.exists(dataset_name):
        dataset = pd.read_csv(dataset_name)
    elif dataset_name == 'identified':
        dataset = dl.load_zipfile()
    else:
        loader = dl.DataLoader()
        dataset = loader.search_index
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', nargs='?')
    parser.add_argument('--raw-outfile', nargs='?', default=None)
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

    ids, scores, parameters = cb.classify_speeches(ids, int(args.sentence_chunking), int(args.score_weighting))

    if args.raw_outfile is not None:
        with open(args.raw_outfile, 'wb') as f:
            pickle.dump(parameters, f)

    dataset['climatebert_scores'] = scores
    dataset = dataset.drop('text', axis=1)
    dataset.to_csv(args.dataset + '.csv')
