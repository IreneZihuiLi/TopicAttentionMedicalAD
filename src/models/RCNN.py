# _*_ coding: utf-8 _*_
'''
This is the one using ELMO!
'''

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from allennlp.modules.elmo import Elmo, batch_to_ids
import numpy as np
import pandas as pd
import csv

options_file = "/home/lily/zl379/Playing/bilm-tf/mmc_new/options.json"
weight_file = "/home/lily/zl379/Playing/bilm-tf/dump/weights.hdf5"


class RCNN(nn.Module):
    def __init__(self, config):
        super(RCNN, self).__init__()

        """
        Arguments
        ---------
        batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        output_size : 2 = (pos, neg)
        hidden_sie : Size of the hidden_state of the LSTM
        vocab_size : Size of the vocabulary containing unique words
        embedding_length : Embedding dimension of GloVe word embeddings
        weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 
        
        """

        self.batch_size = config.batch_size
        self.output_size = config.num_classes
        self.hidden_size = config.embedding_dim
        self.vocab_size = config.vocab_size
        self.embedding_dim = config.embedding_dim

        # elmo parameters
        self.elmo = True if config.elmo == 'True' else False
        self.requires_grad = True if config.elmo_train == 'True' else False

        print ('*'*30,self.elmo,self.requires_grad)
        if self.elmo:
            self.embedding_dim = 64

            # elmo
            print ('requires_grad is ',self.requires_grad)
            self.elmo = Elmo(options_file, weight_file, config.elmo_level, dropout=0, requires_grad=True) # default is False
        else:
            dict = pd.read_csv(filepath_or_buffer=config.word2vec_path, header=None, sep=" ",
                               quoting=csv.QUOTE_NONE).values[:, 1:]
            dict_len, embed_size = dict.shape

            dict_len += 1
            unknown_word = np.zeros((1, embed_size))
            dict = torch.from_numpy(np.concatenate([unknown_word, dict], axis=0).astype(np.double))
            self.embedding = nn.Embedding(num_embeddings=dict_len, embedding_dim=embed_size).from_pretrained(dict,
                                                                                                             freeze=False).cuda()
            self.embedding_dim = embed_size

            # self.embedding = nn.Embedding(self.vocab_size, self.embedding_length).cuda()

        # self.word_embeddings.weight = nn.Parameter(self.weights, requires_grad=True)  # Assigning the look-up table to the pre-trained GloVe word embedding.

        self.dropout = 0.8
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_size, dropout=self.dropout, bidirectional=True)
        self.W2 = nn.Linear(2 * self.hidden_size + self.embedding_dim, self.hidden_size)
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

        """
        
        The idea of the paper "Recurrent Convolutional Neural Networks for Text Classification" is that we pass the embedding vector
        of the text sequences through a bidirectional LSTM and then for each sequence, our final embedding vector is the concatenation of 
        its own GloVe embedding and the left and right contextual embedding which in bidirectional LSTM is same as the corresponding hidden
        state. This final embedding is passed through a linear layer which maps this long concatenated encoding vector back to the hidden_size
        vector. After this step, we use a max pooling layer across all sequences of texts. This converts any varying length text into a fixed
        dimension tensor of size (batch_size, hidden_size) and finally we map this to the output layer.

        """

        if self.elmo:
            elmo_embedding = self.elmo(inputs_elmo)
            sents = elmo_embedding['elmo_representations'][-1]

            input = sents.permute(1, 0, 2)
        else:
            input = self.embedding(input_sentences).permute(1, 0, 2)


        # input = self.word_embeddings(
        #     input_sentence)  # embedded input of shape = (batch_size, num_sequences, embedding_length)
        # input = input.permute(1, 0, 2)  # input.size() = (num_sequences, batch_size, embedding_length)
        if batch_size is None:
            h_0 = Variable(torch.zeros(2, self.batch_size, self.hidden_size).cuda())  # Initial hidden state of the LSTM
            c_0 = Variable(torch.zeros(2, self.batch_size, self.hidden_size).cuda())  # Initial cell state of the LSTM
        else:
            h_0 = Variable(torch.zeros(2, batch_size, self.hidden_size).cuda())
            c_0 = Variable(torch.zeros(2, batch_size, self.hidden_size).cuda())

        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0,c_0))

        final_encoding = torch.cat((output, input), 2).permute(1, 0, 2)
        y = self.W2(final_encoding)  # y.size() = (batch_size, num_sequences, hidden_size)
        y = y.permute(0, 2, 1)  # y.size() = (batch_size, hidden_size, num_sequences)
        y = F.max_pool1d(y, y.size()[2])  # y.size() = (batch_size, hidden_size, 1)
        y = y.squeeze(2)
        logits = self.label(y)

        return logits
