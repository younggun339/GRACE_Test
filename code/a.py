import argparse
from lib2to3.pytree import Base
import torch
import numpy as np
import random
import yaml
import gc
import time
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from utils import load_sc_data, load_sc_causal_data, accuracy_LP
from model import CausalGNN
import torchmetrics
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as Data

# output to a file

# set seed
def set_rng_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# input params
parser = argparse.ArgumentParser()
with open('param.yaml', encoding='utf-8') as f:
    config = yaml.load(f.read(), Loader=yaml.FullLoader)['gcn']
    for key in config.keys():
        name = '--' + key
        parser.add_argument(name, type=type(config[key]), default=config[key])
args = parser.parse_args()

# use cuda
args.cuda = args.cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

# set seed
set_rng_seed(args.seed)

# load data
data_path = "../example/mESC/ExpressionData.csv"
label_path = "../example/mESC/refNetwork.csv"

if args.flag:
    adj_train, feature, train_ids, val_ids, test_ids, train_labels, val_labels, test_labels = load_sc_causal_data(data_path, label_path)
else:
    adj_train, feature, train_ids, val_ids, test_ids, train_labels, val_labels, test_labels = load_sc_data(data_path, label_path)

adj_train = F.normalize(adj_train, p=1, dim=1)