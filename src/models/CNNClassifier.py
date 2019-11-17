

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn import metrics

from mr_loader import Corpus, read_vocab, process_text

import os
import time
from datetime import timedelta

from allennlp.modules.elmo import Elmo, batch_to_ids
import numpy as np
import pandas as pd
import csv

options_file = "/home/lily/zl379/Playing/bilm-tf/mmc_new/options.json"
weight_file = "/home/lily/zl379/Playing/bilm-tf/dump/weights.hdf5"


class CNNClassifier(nn.Module):
    """
    CNN text classification model, based on the paper.
    """

    def __init__(self, config):
        super(CNNClassifier, self).__init__()

        self.vocab_size = config.vocab_size
        self.embedding_dim = config.embedding_dim
        self.num_filters = config.num_filters
        self.kernel_sizes = config.kernel_sizes
        self.num_classes = config.num_classes
        self.dropout_prob = config.dropout_prob

        # elmo parameters
        self.elmo = True if config.elmo == 'True' else False
        self.requires_grad = True if config.elmo_train == 'True' else False

        if self.elmo:
            self.embedding_dim = 64
            # elmo
            print('requires_grad is ', self.requires_grad)
            self.elmo = Elmo(options_file, weight_file, config.elmo_level, dropout=0,
                             requires_grad=True)  # default is False
        else:
            dict = pd.read_csv(filepath_or_buffer=config.word2vec_path, header=None, sep=" ",
                               quoting=csv.QUOTE_NONE).values[:, 1:]
            dict_len, embed_size = dict.shape

            dict_len += 1
            unknown_word = np.zeros((1, embed_size))
            dict = torch.from_numpy(np.concatenate([unknown_word, dict], axis=0).astype(np.double))
            self.embedding = nn.Embedding(num_embeddings=dict_len, embedding_dim=embed_size).from_pretrained(dict,
                                                                                                             freeze=False).cuda()
            self.embedding_length = embed_size

            # self.embedding = nn.Embedding(self.vocab_size, self.embedding_length).cuda()

            # self.embedding = nn.Embedding(self.vocab_size,self.embedding_dim)

        # three different convolutional layers
        self.convs = nn.ModuleList([nn.Conv1d(self.embedding_dim, self.num_filters, k) for k in self.kernel_sizes])
        self.dropout = nn.Dropout(self.dropout_prob)  # a dropout layer

        # concatenate three layers then project into a fc layer
        self.fc1 = nn.Linear(3 * self.num_filters, self.num_classes)  # a dense layer for classification

    @staticmethod
    def conv_and_max_pool(x, conv):
        """Convolution and global max pooling layer"""
        return F.relu(conv(x).permute(0, 2, 1).max(1)[0])

    def forward(self,  inputs,batch_size=None):
        # batch_size is plugged in when different from the default value

        if self.elmo:
            elmo_embedding = self.elmo(inputs)
            # only use the top layer
            sents = elmo_embedding['elmo_representations'][-1]
            print ('CNN Sent size...',sents.size()) # [10, 125, 64]
            # Conv1d takes in (batch, channels, seq_len), but original embedded is (batch, seq_len, channels)
            embedded = sents.permute(0, 2, 1)
        else:
            embedded = self.embedding(inputs).permute(0, 2, 1)

        x = [self.conv_and_max_pool(embedded, k) for k in self.convs]  # convolution and global max pooling
        x = self.fc1(self.dropout(torch.cat(x, 1)))  # concatenation and dropout

        return x

