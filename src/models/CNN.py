'''
Wikipedia 2014 + Gigaword 5: https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76

'''

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
        self.elmo_level = config.elmo_level

        if self.elmo:
            self.embedding_dim = 64
            # elmo
            print('requires_grad is ', self.requires_grad)
            self.elmo = Elmo(options_file, weight_file, config.elmo_level, dropout=0.,
                             requires_grad=self.requires_grad)  # default is False
        else:
            self.embedding = nn.Embedding(self.vocab_size,self.embedding_dim)

        # three different convolutional layers
        self.convs = nn.ModuleList([nn.Conv1d(self.embedding_dim, self.num_filters, k) for k in self.kernel_sizes])
        self.dropout = nn.Dropout(self.dropout_prob)  # a dropout layer

        # concatenate three layers then project into a fc layer
        self.fc1 = nn.Linear(3 * self.num_filters, self.num_classes)  # a dense layer for classification

    @staticmethod
    def conv_and_max_pool(x, conv):
        """Convolution and global max pooling layer"""
        return F.relu(conv(x).permute(0, 2, 1).max(1)[0])

    def forward(self, inputs,inputs_ids,batch_size=None):
        # batch_size is plugged in when different from the default value

        if self.elmo:
            elmo_embedding = self.elmo(inputs_ids)

            # # Method 1 only use the top layer
            # sents = elmo_embedding['elmo_representations'][-1]


            # Method 2 add all
            # sents = elmo_embedding['elmo_representations'][-1]
            # for idx in range(self.elmo_level-1):
                # sents = torch.add(sents,elmo_embedding['elmo_representations'][idx])

            # Method 3, weighted sum over all layers
            # sent_list = [vect for vect in elmo_embedding['elmo_representations']]
            # var_list = [torch.Tensor(1).cuda() for _ in range(self.elmo_level)]
            # sents = torch.mul(sent_list[-1],var_list[-1])
            #
            # for idx in range(self.elmo_level-1):
            #     sents += torch.mul(sent_list[idx],var_list[idx])

            # Method 3+
            sent_list = [vect for vect in elmo_embedding['elmo_representations']]
            sents = torch.cat(sent_list,2).view(batch_size,-1,self.embedding_dim, self.elmo_level)
            vars = torch.Tensor(self.elmo_level,1).cuda()
            sents = torch.matmul(sents,vars).view(batch_size,-1,self.embedding_dim)


            # end of all methods
            # print ('CNN Sent size...',sents.size()) # [10, 125, 64]
            # Conv1d takes in (batch, channels, seq_len), but original embedded is (batch, seq_len, channels)
            embedded = sents.permute(0, 2, 1)
        else:
            embedded = self.embedding(inputs).permute(0, 2, 1)

        x = [self.conv_and_max_pool(embedded, k) for k in self.convs]  # convolution and global max pooling
        x = self.fc1(self.dropout(torch.cat(x, 1)))  # concatenation and dropout

        return x

