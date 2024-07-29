## LLM Classification Benchmarking For Climate Change Content In Central Bank Speeches
---

This repo contains code for assessing the ability of different LLMs to identify central bank speeches from the BIS with content related to climate change. We benchmark against ClimateBERT and GPT-3.5. Benchmarking against ClimateBERT can be performed by running the command:

```
python src/climatebert.py [id] --sentence-chunking=[integer] --score-weighting=[integer]
    id                  :   the identifier of the speech in the BIS database
    sentence-chunking   :   the number of sentences to feed into ClimateBERT at a time 
    score-weighting     :   the factor by which to weight a climate-related paragraph over a non-climate-related paragraph
```

Running this for a single speech will output a unitless final score. Speeches with very little climate-related content will receive scores close to -1. Speeches that are very climate-related will recieve a more positive score, the magnitude of which is determined by the weighting coefficient specified in the script call.

The classifier works by splitting the text of each speech into chunks of several sentences (a number controlled by the `sentence-chunking` parameter). Each chunk of sentences is assigned a label (`yes` or `no`) that states whether that chunk is climate-related, as well as a score between zero and one that quanitifies the confidence of the assigned label. Once an entire speech has been processed by ClimateBERT, the results are two vectors: one containing `yes||no` labels, and one containing confidence scores. 

This vector is aggregated into a total final score as follows. For each element of the score vector, if the corresponding label is `yes`, that score is multiplied by the specified weight factor. If the corresponding label is `no`, the score is multiplied by -1. This results in a vector containing the weighted confidence values for each chunk of sentences, which is then summed to compute the final score. A good choice for weight seems to be around 3, but further calibration will be necessary.

### Next Steps

We'll need to calibrate the model by choosing the optimal weight to correctly classify each speech. I'll probably do this by splitting Mitsuhiro's dataset into a 35% test portion and an 65% validation portion. That way I can choose the best weight using a grid search on the results of the 35% test portion. Then the 65% validation portion remains an independent sample that can be tested against.

I also need to implement GPT-3.5, but that will come later since it costs money.

---

### Example

To process the speech [Challenges and Opportunities in Scaling up Green Finance](https://www.bis.org/review/r221222a.pdf), which is of course climate change related, we can run the classifier with `sentence-chunking = 5` and `score-weighting = 3`, which are both good choices:

```
arsen@arsen:/home/arsen/climate-speeches$ python src/climatebert.py r221222a --sentence_chunking=5 --score-weighting=3
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 30/30 [00:02<00:00, 12.59it/s]
final score: 0.7107816815376282
```

The final score is positive, which we can interpret as a climate-related speech.