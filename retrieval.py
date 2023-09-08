import numpy as np
from torch import nn
import torch

import yaml
import sys
import os

from MSRVTT_test_set import MSRVTT_test
from torch.utils.data import DataLoader
from model.model import PureVIT
import transformers
import random
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


device = "cuda" if torch.cuda.is_available() else "cpu"

setup_seed(0)

resume = "./vit_ours/100.pth"
model = PureVIT()
checkpoint = torch.load(resume, map_location='cpu')
model.load_state_dict(checkpoint)



def compute_similarity(image_features, text_features, bs = 2984):
    # compute similarity
    max_pairs = image_features.shape[0]
    similarity_scores = torch.zeros(max_pairs, max_pairs)
    for v in range(0, max_pairs, bs):
        for t in range(0, max_pairs, bs):
            print('Processing Visual '+str(v)+' Text '+str(t), end='\r')
            batch_visual_emb = image_features[v:v+bs]
            batch_caption_emb = text_features[t:t+bs]

            logits = batch_visual_emb @ batch_caption_emb.t()
            similarity_scores[v:v+bs,t:t+bs] = logits

    print('Done similarity')
    return similarity_scores

def compute_retrieval(a2b_sims, return_ranks=True):
    """
    Args:
        a2b_sims: Result of computing similarity between two sets of embeddings (emb1 @ emb2.T)
            with shape (num_datapoints, num_datapoints).

    Returns:
        Retrieval metrics for that similarity.
    """
    npts = a2b_sims.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    # loop source embedding indices
    for index in range(npts):
        # get order of similarities to target embeddings
        inds = np.argsort(a2b_sims[index])[::-1]
        # find where the correct embedding is ranked
        where = np.where(inds == index)
        rank = where[0][0]
        ranks[index] = rank
        # save the top1 result as well
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    report_dict = {"r1": r1, "r5": r5, "r10": r10, "r50": r50, "medr": medr, "meanr": meanr, "sum": r1 + r5 + r10}

    if return_ranks:
        return report_dict, (ranks, top1)
    else:
        return report_dict


valid_dataset = MSRVTT_test()
valid_dataloader = DataLoader(valid_dataset, batch_size=4,num_workers=16, shuffle=False,drop_last=True)

image_features = []
text_features = []

model = model.to("cuda")
tokenizer = transformers.AutoTokenizer.from_pretrained("./pretrained/distilbert-base-uncased",
                                                           TOKENIZERS_PARALLELISM=False)

from tqdm import tqdm
for batch_idx, data in enumerate(tqdm(valid_dataloader)):
    print('Evaluating batch {}/{}'.format(batch_idx, len(valid_dataloader)), end = "\r")
    #print(batch)   
    data['text'] = tokenizer(data['text'], return_tensors='pt', padding=True,
                                            truncation=True)
    data['text'] = {key: val.to("cuda") for key, val in data['text'].items()}
    data['video'] = data['video'].to("cuda")

    print(data["video"].shape)
    text_emb, image_emb = model(data)

  #  text_emb = model_out["text_features"] #embed with text encoder
   # image_emb = model_out["image_features"]  #embed with image encoder
    

    text_features.append(text_emb.detach().cpu())
    image_features.append(image_emb.detach().cpu())

    # if(batch_idx==100):
    #     break 

image_features = torch.cat(image_features, 0)
text_features = torch.cat(text_features, 0)
print('Done forward')

# normalized features
image_features = image_features / image_features.norm(dim=-1, keepdim=True)
text_features = text_features / text_features.norm(dim=-1, keepdim=True)


print(image_features.shape, text_features.shape)
similarity_scores = compute_similarity(image_features, text_features)
i2t_dict = compute_retrieval(similarity_scores.numpy())
t2i_dict = compute_retrieval(similarity_scores.t().numpy())
print('i2t', i2t_dict)
print('t2i', t2i_dict)
