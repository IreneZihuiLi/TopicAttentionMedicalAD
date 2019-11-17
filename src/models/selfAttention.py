# _*_ coding: utf-8 _*_

import csv
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from allennlp.modules.elmo import Elmo, batch_to_ids
import pandas as pd
import numpy as np

options_file = "/home/lily/zl379/Playing/bilm-tf/mmc_new/options.json"
weight_file = "/home/lily/zl379/Playing/bilm-tf/dump/weights.hdf5"


class SelfAttention(nn.Module):
    def __init__(self, config):
        super(SelfAttention, self).__init__()

        """
        Arguments
        ---------
        batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        output_size : 2 = (pos, neg)
        hidden_sie : Size of the hidden_state of the LSTM
        vocab_size : Size of the vocabulary containing unique words
        embedding_length : Embeddding dimension of GloVe word embeddings
        weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 
        
        --------
        
        """
        self.vocab_size = config.vocab_size
        self.embedding_dim = config.embedding_dim
        self.num_classes = config.num_classes
        self.dropout_prob = config.dropout_prob
        self.hidden_size = config.embedding_dim

        # elmo parameters
        self.elmo = True if config.elmo == 'True' else False
        self.requires_grad = True if config.elmo_train == 'True' else False
        print('*' * 30, self.elmo, self.requires_grad)


        if self.elmo:
            # elmo
            print('requires_grad is ', self.requires_grad)
            self.elmo = Elmo(options_file, weight_file, config.elmo_level, dropout=0,
                             requires_grad=self.requires_grad)  # default is False
        else:
            # Method1 : self embedding
            # self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)

            print('Now loading GloVe')
            # Method 2: apply glove
            dict = pd.read_csv(filepath_or_buffer=config.word2vec_path, header=None, sep=" ", quoting=csv.QUOTE_NONE).values[:,
                   1:]
            dict_len, embed_size = dict.shape
            dict_len += 1
            unknown_word = np.zeros((1, embed_size))
            dict = torch.from_numpy(np.concatenate([unknown_word, dict], axis=0).astype(np.float))
            self.embedding = nn.Embedding(num_embeddings=dict_len, embedding_dim=embed_size).from_pretrained(dict,freeze=False)

        # self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)
        # self.word_embeddings.weights = nn.Parameter(weights, requires_grad=True)
        self.dropout = 0.8
        self.bilstm = nn.LSTM(self.embedding_dim, self.hidden_size, dropout=self.dropout, bidirectional=True)

        # We will use da = 350, r = 30 & penalization_coeff = 1 as per given in the self-attention original ICLR paper

        # for attention
        self.W_s1 = nn.Linear(2 * self.hidden_size, 128)
        self.W_s2 = nn.Linear(128, 32)


        self.fc_layer = nn.Linear(32 * 2 * self.hidden_size, 128)
        self.label = nn.Linear(128, self.num_classes)

    def attention_net(self, lstm_output):

        """
        Now we will use self attention mechanism to produce a matrix embedding of the input sentence in which every row represents an
        encoding of the inout sentence but giving an attention to a specific part of the sentence. We will use 30 such embedding of
        the input sentence and then finally we will concatenate all the 30 sentence embedding vectors and connect it to a fully
        connected layer of size 2000 which will be connected to the output layer of size 2 returning logits for our two classes i.e.,
        pos & neg.

        Arguments
        ---------

        lstm_output = A tensor containing hidden states corresponding to each time step of the LSTM network.
        ---------

        Returns : Final Attention weight matrix for all the 30 different sentence embedding in which each of 30 embeddings give
                  attention to different parts of the input sentence.

        Tensor size : lstm_output.size() = (batch_size, num_seq, 2*hidden_size)
                      attn_weight_matrix.size() = (batch_size, 30, num_seq)

        """
        # it seems to be a two-layer NN here
        attn_weight_matrix = self.W_s2(F.tanh(self.W_s1(lstm_output)))
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)

        return attn_weight_matrix

    def forward(self, inputs, input_sentences, batch_size=None):

        """
        Parameters
        ----------
        input_sentence: input_sentence of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)

        Returns
        -------
        Output of the linear layer containing logits for pos & neg class.

        """
        if self.elmo:
            elmo_embedding = self.elmo(input_sentences)
            sents = elmo_embedding['elmo_representations'][-1]

            input = sents.permute(1, 0, 2)
        else:
            input = self.embedding(inputs)
            # print ('The input size...',input.size()) #[10, 200, 64]
            input = input.permute(1, 0, 2).float()

        # input = self.word_embeddings(input_sentences)
        # input = input.permute(1, 0, 2)
        if batch_size is None:
            h_0 = Variable(torch.zeros(2, self.batch_size, self.hidden_size).cuda())
            c_0 = Variable(torch.zeros(2, self.batch_size, self.hidden_size).cuda())
        else:
            h_0 = Variable(torch.zeros(2, batch_size, self.hidden_size).cuda())
            c_0 = Variable(torch.zeros(2, batch_size, self.hidden_size).cuda())

        output, (h_n, c_n) = self.bilstm(input, (h_0, c_0))
        output = output.permute(1, 0, 2)
        # output.size() = (batch_size, num_seq, 2*hidden_size)
        # h_n.size() = (1, batch_size, hidden_size)
        # c_n.size() = (1, batch_size, hidden_size)
        attn_weight_matrix = self.attention_net(output)
        # attn_weight_matrix.size() = (batch_size, r, num_seq)
        # output.size() = (batch_size, num_seq, 2*hidden_size)
        hidden_matrix = torch.bmm(attn_weight_matrix, output)
        # hidden_matrix.size() = (batch_size, r, 2*hidden_size)
        # Let's now concatenate the hidden_matrix and connect it to the fully connected layer.
        fc_out = self.fc_layer(hidden_matrix.view(-1, hidden_matrix.size()[1] * hidden_matrix.size()[2]))
        logits = self.label(fc_out)
        # logits.size() = (batch_size, output_size)

        return logits
