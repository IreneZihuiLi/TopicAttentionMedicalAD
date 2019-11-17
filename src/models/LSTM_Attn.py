# _*_ coding: utf-8 _*_

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from allennlp.modules.elmo import Elmo, batch_to_ids
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import csv
import pandas as pd

options_file = "/home/lily/zl379/Playing/bilm-tf/mmc_new/options.json"
weight_file = "/home/lily/zl379/Playing/bilm-tf/dump/weights.hdf5"

class AttentionModel(torch.nn.Module):
    def __init__(self, config):
        super(AttentionModel, self).__init__()

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
            self.embedding_length = 64
            # elmo
            print('requires_grad is ', self.requires_grad)
            self.elmo = Elmo(options_file, weight_file, config.elmo_level, dropout=0,
                             requires_grad=self.requires_grad)  # default is False
        else:
            dict = pd.read_csv(filepath_or_buffer=config.word2vec_path, header=None, sep=" ",
                               quoting=csv.QUOTE_NONE).values[:, 1:]
            dict_len, embed_size = dict.shape

            dict_len += 1
            unknown_word = np.zeros((1, embed_size))
            dict = torch.from_numpy(np.concatenate([unknown_word, dict], axis=0).astype(np.double))
            self.embedding = nn.Embedding(num_embeddings=dict_len, embedding_dim=embed_size).from_pretrained(dict,
                                                                                                             freeze=False)
            self.embedding_length = embed_size

            self.embedding = nn.Embedding(self.vocab_size, self.embedding_length).cuda()

            # self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_length)
        # self.word_embeddings.weights = nn.Parameter(weights, requires_grad=True)
        self.lstm = nn.LSTM(self.embedding_length, self.hidden_size)
        self.label = nn.Linear(self.hidden_size, self.output_size)

    # self.attn_fc_layer = nn.Linear()

    def attention_net(self, lstm_output, final_state):

        """
        Now we will incorporate Attention mechanism in our LSTM model. In this new model, we will use attention to compute soft alignment score corresponding
        between each of the hidden_state and the last hidden_state of the LSTM. We will be using torch.bmm for the batch matrix multiplication.

        Arguments
        ---------

        lstm_output : Final output of the LSTM which contains hidden layer outputs for each sequence.
        final_state : Final time-step hidden state (h_n) of the LSTM

        ---------

        Returns : It performs attention mechanism by first computing weights for each of the sequence present in lstm_output and and then finally computing the
                  new hidden state.

        Tensor Size :
                    hidden.size() = (batch_size, hidden_size)
                    attn_weights.size() = (batch_size, num_seq)
                    soft_attn_weights.size() = (batch_size, num_seq)
                    new_hidden_state.size() = (batch_size, hidden_size)

        """

        hidden = final_state.squeeze(0)

        lstm_output = torch.Tensor(lstm_output.float().cpu()).cuda()

        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state

    def forward(self, inputs, input_sentences, batch_size=None):

        """
        Parameters
        ----------
        input_sentence: input_sentence of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)

        Returns
        -------
        Output of the linear layer containing logits for pos & neg class which receives its input as the new_hidden_state which is basically the output of the Attention network.
        final_output.shape = (batch_size, output_size)

        """

        if self.elmo:
            elmo_embedding = self.elmo(input_sentences)
            sents = elmo_embedding['elmo_representations'][-1]

            input = sents.permute(1, 0, 2)
        else:
            input = self.embedding(inputs).permute(1, 0, 2)


        # input = self.word_embeddings(input_sentences)
        # input = input.permute(1, 0, 2)
        # if batch_size is None:
        #     h_0 = Variable(torch.ones(1, self.batch_size, self.hidden_size).cuda())
        #     c_0 = Variable(torch.ones(1, self.batch_size, self.hidden_size).cuda())
        # else:
        #     h_0 = Variable(torch.ones(1, batch_size, self.hidden_size).cuda())
        #     c_0 = Variable(torch.ones(1, batch_size, self.hidden_size).cuda())


        output, (final_hidden_state, final_cell_state) = self.lstm(input, None)  # final_hidden_state.size() = (1, batch_size, hidden_size)
        output = output.permute(1, 0, 2)  # output.size() = (batch_size, num_seq, hidden_size)

        attn_output = self.attention_net(output, final_hidden_state)
        logits = self.label(attn_output)


        return logits
