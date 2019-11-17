# _*_ coding: utf-8 _*_

'''
To debug:
>> time python main_pytorch.py -m self -elmo False

SelfAttention(
  (embedding): Embedding(400002, 100)
  (bilstm): LSTM(100, 100, dropout=0.8, bidirectional=True)
  (W_s1): Linear(in_features=200, out_features=128, bias=True)
  (W_s2): Linear(in_features=128, out_features=32, bias=True)
  (W_topic): Linear(in_features=32, out_features=32, bias=True)
  (fc_layer): Linear(in_features=6400, out_features=128, bias=True)
  (label): Linear(in_features=128, out_features=4, bias=True)
  (glove_lda): Embedding(400002, 100)
)


Note...if using elmo, change the dimension to be 64; if not, use 100
>> python main_pytorch.py -m self -elmo True -e 5 -topic True


'''


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

def load_lda(data_path,top_k):
    '''
    Loaded by other classes.
    :param data_path: LDA.txt
    :param top_k: number of top words (top k of that topic
    :return: a list of list, the inner list is the list of top words in that topic.
    '''
    topic_words = []
    with open (data_path,'r') as f:
        for line in f.readlines():
            content = line.strip().split(' ')
            if len(content) == top_k:
                topic_words.append(content)
    # print('Topic 1:',topic_words[0])
    # return is a list of list, the inner list is the list of top words in that topic
    return topic_words


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
        self.use_elmo = config.elmo
        self.requires_grad = True if config.elmo_train == 'True' else False

        # lda parameters
        self.lda_path = config.lda_path
        self.use_topic = True if config.use_topic == 'True' else False

        self.lda_embed_size = config.word2vec_dim


        if self.use_elmo == 'True':
            # elmo
            print ('Now in to elmo..',self.use_elmo)
            self.embedding_dim = 64
            self.elmo = Elmo(options_file, weight_file, config.elmo_level, dropout=0,
                             requires_grad=self.requires_grad)  # default is False

            # Following part is loading word2id dict for topic words
            dict_glove = pd.read_csv(filepath_or_buffer=config.word2vec_path, header=None, sep=" ",
                                     quoting=csv.QUOTE_NONE).values[:, 1:]

            dict_len, embed_size = dict_glove.shape
            self.lda_embed_size = embed_size
            dict_len += 1
            unknown_word = np.zeros((1, embed_size))
            self.dict_glove = torch.from_numpy(np.concatenate([unknown_word, dict_glove], axis=0).astype(np.float))
            self.embedding = nn.Embedding(num_embeddings=dict_len, embedding_dim=embed_size).from_pretrained(
                self.dict_glove, freeze=False)


            content = pd.read_csv(filepath_or_buffer=config.word2vec_path, header=None, sep=" ", quoting=csv.QUOTE_NONE,
                                  usecols=[0]).values
            words = [word[0] for word in content]
            print('Glove Loaded...', len(words))
            self.word_to_id = dict(zip(words, range(len(words))))
            self.load_topic(self.dict_glove, 400000, self.hidden_size)

        else:
            # Method1 : self embedding
            # self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)


            # Method 2: apply glove
            # print('Now loading GloVe')

            dict_glove = pd.read_csv(filepath_or_buffer=config.word2vec_path, header=None, sep=" ", quoting=csv.QUOTE_NONE).values[:,1:]


            dict_len, embed_size = dict_glove.shape
            self.lda_embed_size = embed_size
            dict_len += 1
            unknown_word = np.zeros((1, embed_size))
            self.dict_glove = torch.from_numpy(np.concatenate([unknown_word, dict_glove], axis=0).astype(np.float))
            self.embedding = nn.Embedding(num_embeddings=dict_len, embedding_dim=embed_size).from_pretrained(self.dict_glove,freeze=False)

            # Following part is loading word2id dict for topic words
            content = pd.read_csv(filepath_or_buffer=config.word2vec_path, header=None, sep=" ", quoting=csv.QUOTE_NONE,
                                  usecols=[0]).values
            words = [word[0] for word in content]
            print('Glove Loaded...', len(words))
            self.word_to_id = dict(zip(words, range(len(words))))
            self.load_topic(self.dict_glove, dict_len, embed_size)


        # self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)
        # self.word_embeddings.weights = nn.Parameter(weights, requires_grad=True)
        self.dropout = 0.8
        self.bilstm = nn.LSTM(self.embedding_dim, self.hidden_size, dropout=self.dropout, bidirectional=True)

        # We will use da = 350, r = 30 & penalization_coeff = 1 as per given in the self-attention original ICLR paper

        self.attention_dim = 8
        # for attention
        self.W_s1 = nn.Linear(2 * self.hidden_size, 128)
        self.W_s2 = nn.Linear(128, self.attention_dim) # changed 32 to 4

        # for topic attention
        self.W_topic = nn.Linear(self.attention_dim * 2 * self.hidden_size, self.lda_embed_size) # 6400, 100

        # for fc (final layer)
        self.fc_layer = nn.Linear(self.attention_dim * 2 * self.hidden_size, 128)

        self.fc_layer_topic = nn.Linear(self.attention_dim * 2 * self.hidden_size + self.hidden_size,128)

        self.label = nn.Linear(128, self.num_classes)



    def load_topic(self,dict_glove,dict_len, embed_size):
        topic_words = load_lda(self.lda_path,100)

        topic_words_id = []
        for topic in topic_words:
            word_id = []
            for x in topic:
                if x in self.word_to_id:
                    word_id.append(self.word_to_id[x])
                else:
                    word_id.append(0)

            topic_words_id.append(word_id)# not considering OOV

        topic_words_id_2d = np.asarray(topic_words_id)
        print('Shape of topics 1..',np.shape(topic_words_id_2d))
        topic_words_id = torch.tensor(topic_words_id_2d)
        # topic_words_id = torch.tensor(topic_words_id_2d)

        # change to vars
        topic_words_id_var = Variable(topic_words_id).cuda()
        print ('Shape of topics..',topic_words_id_var.size()) #  torch.Size([50, 20])

        # change to embedding
        self.glove_lda = nn.Embedding(num_embeddings=dict_len, embedding_dim=embed_size).from_pretrained(dict_glove, freeze=False).cuda()
        # self.glove_lda = nn.Embedding(num_embeddings=dict_len, embedding_dim=embed_size).from_pretrained(dict_glove,freeze=True)
        self.topic_words_embed = self.glove_lda(topic_words_id_var)

        print('Loaded topics...', self.topic_words_embed.size()) # torch.Size([50, 20, 100]) (n_topics, n_top_words, dim)

        #sum and average as each topic
        self.topic_matrix = torch.div(torch.sum(self.topic_words_embed,1),embed_size).type(torch.FloatTensor).cuda()
        print('Sum up topics...', self.topic_matrix.size()) # Sum up topics... torch.Size([50, 100])


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

        Returns : Final Attention weight matrix for all the 32 different sentence embedding in which each of 30 embeddings give
                  attention to different parts of the input sentence.

        Tensor size : lstm_output.size() = (batch_size, num_seq, 2*hidden_size)
                      attn_weight_matrix.size() = (batch_size, 32, num_seq)

        """
        # print ('Attention...,', lstm_output.size()) # [10, 179, 200]
        # it seems to be a two-layer NN here
        attn_weight_matrix = self.W_s2(F.tanh(self.W_s1(lstm_output)))
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)

        # print ('Atten_weight_matrix...',attn_weight_matrix.size())
        #Atten_weight_matrix... torch.Size([10, 32, 179])

        # TODO: shape [10, 32, 179]  [batch_size, hidden, sequence_len]
        return attn_weight_matrix


    def topic_attention(self,content):
        '''
        This is the topic attention model, it takes context vector, and topic matrix as inputs
        and computes a weighted sum of the topics.
        :return:  a vector
        '''
        # print ('content Matrix...',content.size())
        # torch.Size([10, 6400])

        content = F.tanh(self.W_topic(content))
        content_atten = torch.mm(content,self.topic_matrix.t()) #matrix multiplication with out batch
        content_atten = F.softmax(content_atten,dim=1)
        # print('content_atten ...', content_atten.size()) # (10,50)


        return content_atten


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

        if self.use_elmo  == 'True':
            elmo_embedding = self.elmo(input_sentences)
            sents = elmo_embedding['elmo_representations'][-1]
            input = sents.permute(1, 0, 2)
        else:
            # print('The input size...', inputs[0])  # [10, 200]
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

        attn_weight_matrix = self.attention_net(output) # TODO: this is not the context vector!!!! this is the weights after the softmax..shape: [10, 32, 200]


        hidden_matrix = torch.bmm(attn_weight_matrix, output) # TODO: this is the context vector!!

        # print('attn_weight_matrix...', attn_weight_matrix.size()) # (batch, atten_dim, step)
        # print ('output...',output.size()) # (batch, step, dim)
        # print ('hidden...',hidden_matrix.size()) # (batch, step, dim)


        # original codes stop here
        if self.use_topic:
            # print('Use topic?', self.use_topic)

            # applying topics...
            content = hidden_matrix.view(-1, hidden_matrix.size()[1] * hidden_matrix.size()[2])
            topic_weight_matrix = self.topic_attention(content)  # (10,50)  (batch_size, n_topics)
            topic_content = torch.mm(topic_weight_matrix, self.topic_matrix)  # 10*100

            topic_content_concat = torch.cat([content,topic_content],1)

            fc_out = self.fc_layer_topic(topic_content_concat)

            logits = self.label(fc_out)
        else:
            # Let's now concatenate the hidden_matrix and connect it to the fully connected layer.
            fc_out = self.fc_layer(hidden_matrix.view(-1, hidden_matrix.size()[1] * hidden_matrix.size()[2]))
            logits = self.label(fc_out)

        return logits
