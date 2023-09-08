import transformers

import torch
from trainer_MSVD import Multi_Trainer_dist_MSVD
from MSRVTT import MSRVTTDataset
from torch.utils.data import DataLoader
#from model.video_transformer_flip import SpaceTimeTransformer
from model.model import PureVIT

from model.loss import NormSoftmaxLoss

import random
import numpy as np
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def run():
    setup_seed(10)   
    # build tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained("./pretrained/distilbert-base-uncased",
                                                           TOKENIZERS_PARALLELISM=False)

    # setup data_loader instances
    data_loader = DataLoader(MSRVTTDataset(), batch_size=8, num_workers=16, pin_memory=True, shuffle=True,drop_last=True)

    model = PureVIT().cuda()

    loss = NormSoftmaxLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)


    writer = None

 
    trainer = Multi_Trainer_dist_MSVD(model, loss, optimizer,
                      data_loader=data_loader,
                      writer=writer,
                      tokenizer=tokenizer,
                      max_samples_per_epoch=500000)

    trainer.train()


if __name__ == '__main__':

    run()
