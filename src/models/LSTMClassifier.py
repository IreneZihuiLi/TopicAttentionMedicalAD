
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
import pandas as pd
import csv

from allennlp.modules.elmo import Elmo, batch_to_ids


options_file = "/home/lily/zl379/Playing/bilm-tf/mmc_new/options.json"
weight_file = "/home/lily/zl379/Playing/bilm-tf/dump/weights.hdf5"



class LSTMClassifier(nn.Module):


    def __init__(self, config):

        super(LSTMClassifier, self).__init__()

        """
        Arguments
        ---------
        batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        output_size : 2 = (pos, neg)
        hidden_sie : Size of the hidden_state of the LSTM
        vocab_size : Size of the vocabulary containing unique words
        embedding_length : Embeddding dimension of GloVe word embeddings
        weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 

        """

        self.batch_size = config.batch_size
        self.output_size = config.num_classes
        self.hidden_size = config.embedding_dim
        self.vocab_size = config.vocab_size
        self.embedding_length = config.embedding_dim

        # elmo parameters
        self.elmo = True if config.elmo == 'True' else False
        self.requires_grad = True if config.elmo_train == 'True' else False

        print('*' * 30, self.elmo, self.requires_grad)
        if self.elmo:
            # elmo
            self.embedding_length = 64
            print('requires_grad is ', self.requires_grad)
            self.elmo = Elmo(options_file, weight_file, config.elmo_level, dropout=0,
                             requires_grad=False)  # default is False
        else:

            dict = pd.read_csv(filepath_or_buffer=config.word2vec_path, header=None, sep=" ",
                               quoting=csv.QUOTE_NONE).values[:,1:]
            dict_len, embed_size = dict.shape

            dict_len += 1
            unknown_word = np.zeros((1, embed_size))
            dict = torch.from_numpy(np.concatenate([unknown_word, dict], axis=0).astype(np.float))
            self.embedding = nn.Embedding(num_embeddings=dict_len, embedding_dim=embed_size).from_pretrained(dict,freeze=False).cuda()
            self.embedding_length = embed_size
            # self.embedding = nn.Embedding(self.vocab_size, self.embedding_length)

        # self.word_embeddings.weight = nn.Parameter(weights,
        #                                            requires_grad=True)  # Assigning the look-up table to the pre-trained GloVe word embedding.

        self.lstm = nn.LSTM(self.embedding_length, self.hidden_size)
        self.label = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_sentences, inputs_elmo, batch_size=None):

        """
        Parameters
        ----------
        input_sentence: input_sentence of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)

        Returns
        -------
        Output of the linear layer containing logits for positive & negative class which receives its input as the final_hidden_state of the LSTM
        final_output.shape = (batch_size, output_size)

        """

        ''' Here we will map all the indexes present in the input sequence to the corresponding word vector using our pre-trained word_embedddins.'''

        if self.elmo:
            elmo_embedding = self.elmo(inputs_elmo)
            sents = elmo_embedding['elmo_representations'][-1]
            input = sents.permute(1, 0, 2)
        else:
            input = self.embedding(input_sentences).permute(1, 0, 2).float()

        # import pdb;pdb.set_trace()

        # print('INPUT shape...', input_sentence.size())
        # input = self.word_embeddings(input_sentence)  # embedded input of shape = (batch_size, num_sequences,  embedding_length)
        # print ('EMBD shape...',input.size())
        # input = input.permute(1, 0, 2)  # input.size() = (num_sequences, batch_size, embedding_length)
        if batch_size is None:


            h_0 = Variable(torch.ones(1, self.batch_size, self.hidden_size).cuda())  # Initial hidden state of the LSTM
            c_0 = Variable(torch.ones(1, self.batch_size, self.hidden_size).cuda())  # Initial cell state of the LSTM
        else:

            h_0 = Variable(torch.ones(1, batch_size, self.hidden_size).cuda())
            c_0 = Variable(torch.ones(1, batch_size, self.hidden_size).cuda())


        output, (final_hidden_state, final_cell_state) = self.lstm(input,(h_0,c_0))
        final_output = self.label(final_hidden_state[-1])  # final_hidden_state.size() = (1, batch_size, hidden_size) & final_output.size() = (batch_size, output_size)



        return final_output