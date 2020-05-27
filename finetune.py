import torch
import random
import dataset
from trainer import train
from transformers import AutoTokenizer, AutoModelWithLMHead
torch.cuda.get_device_name(0)

DO_DATASET = True
DO_SETUP = True
DO_TRAIN = True

tokenizer = AutoTokenizer.from_pretrained("xlm-mlm-xnli15-1024")

if DO_DATASET:
    dataset.setup()
if DO_SETUP:
    model = AutoModelWithLMHead.from_pretrained("xlm-mlm-xnli15-1024")
else:
    model = AutoModelWithLMHead.from_pretrained('model')

model.cuda()

if DO_TRAIN:
    train(model, tokenizer, 1, 4)
