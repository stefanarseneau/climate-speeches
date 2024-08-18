from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from transformers.pipelines.pt_utils import KeyDataset
import datasets
from tqdm.auto import tqdm
import numpy as np

from openai import OpenAI

import dataloader as dl
import argparse
   
def summation_score(pd_dataset, weight):
    classifier = (2*np.all([pd_dataset['label'] == 'yes'], axis = 0).astype(float) - 1)
    classifier *= pd_dataset['score'].to_numpy()
    raw_scores = classifier.copy()
    classifier[classifier > 0] *= weight
    score = sum(classifier ) / len(pd_dataset)
    return score, raw_scores

def classify_speeches(ids, sentence_chunking, score_weighting):
    final_scores = []
    parameters = {}

    dat = dl.DataLoader()
    client = OpenAI()

    for i, id in enumerate(tqdm(ids)):
        url, title, description, author, date, text, dataset_pd, dataset_hf = dat.fetch_text(id, sentence_chunking = int(sentence_chunking))

        if dataset_hf != '':
            print('making query...')
            print(dataset_pd)

            response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {
                                "role": "system",
                                "content": "You will be provided with a central bank speech which may or may not be about climate change. A central bank speech about climate change typically addresses the financial and economic implications of climate change, climate risks, and sustainability. It explores how climate change can impact economic and financial market activity, the stability of the financial system, the stability of the banking system, and it might emphasize the need for monitoring and robust financial strategies and regulations to mitigate these risks. The speech may outline the central bank's role in supervision and regulation, incorporating climate-related financial disclosures, fostering green investments, and supporting policy measures that align with climate goals. By highlighting the intersection of environmental sustainability and monetary policy, the speech aims to drive awareness and action among financial institutions, policymakers, and the public, promoting a resilient and sustainable economic framework in the face of climate challenges. Respond ``yes'' if this speech is climate related and ``no'' if it is not."
                                },
                                {
                                "role": "user",
                                "content": f"{text}"
                                }
                            ],
                            temperature=0.7,
                            top_p=1
                        )
            print(title)
            print(response)

            #score, raw_scores = summation_score(dataset_pd, weight = int(score_weighting))
            #parameters[id] = raw_scores
            #final_scores.append(score)
        else:
            final_scores.append(-999)
            parameters[id] = []

    return ids, final_scores, parameters

if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument('id', nargs='?')
   parser.add_argument('--sentence-chunking', nargs='?', default=None)
   parser.add_argument('--score-weighting', nargs='?', default='1')
   args = parser.parse_args()

   scores = classify_speeches([args.id], int(args.sentence_chunking), int(args.score_weighting))
   print('speech', args.id, 'final score:', scores[0])