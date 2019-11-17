#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import Counter
import numpy as np
import re,csv
from allennlp.modules.elmo import Elmo, batch_to_ids
import os
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from allennlp.modules.elmo import Elmo, batch_to_ids
import pandas as pd

token = '<HERE>'

# glove_file = '/home/lily/zl379/data/GloVe/glove.6B.100d.txt'
glove_file = '300d_w2v.txt'

lda_dim = 300

def open_file(filename, mode='r'):
    """
    Commonly used file reader and writer, change this to switch between python2 and python3.
    :param filename: filename
    :param mode: 'r' and 'w' for read and write respectively
    """
    return open(filename, mode)

def clean_str(s):
    # pass in the whole sentence

    # filter out \n, <p> tags

    s = s.lower()
    # remove filters
    s = s.replace('*', '')
    s = s.replace('\n','')
    s = s.replace('\r', '')
    s = s.replace('\t', '')
    s = s.replace('`', '')
    s = s.replace(',', '')
    s = s.replace(';', '')
    s = s.replace('.', '')
    s = s.replace('-', '')
    s = s.replace('\'', '')
    s = s.replace('\"', '')
    s = s.replace('(', '')
    s = s.replace(')', '')
    s = s.replace('-', ' ')
    s = s.replace('[!@#$]', '')


    return s
# def clean_str(string):
#     """
#     Tokenization/string cleaning for all datasets except for SST.
#     Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
#     """
#     string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
#     string = re.sub(r"\'s", " \'s", string)
#     string = re.sub(r"\'ve", " \'ve", string)
#     string = re.sub(r"n\'t", " n\'t", string)
#     string = re.sub(r"\'re", " \'re", string)
#     string = re.sub(r"\'d", " \'d", string)
#     string = re.sub(r"\'ll", " \'ll", string)
#     string = re.sub(r",", " , ", string)
#     string = re.sub(r"!", " ! ", string)
#     string = re.sub(r"\(", " \( ", string)
#     string = re.sub(r"\)", " \) ", string)
#     string = re.sub(r"\?", " \? ", string)
#     string = re.sub(r"\s{2,}", " ", string)
#     return string.strip().lower()





def build_vocab(data, vocab_dir, vocab_size=400001):
    """
    Build vocabulary file from training data.
    """
    print('Building vocabulary...')

    all_data = []  # group all data
    for content in data:
        all_data.extend(content.split())

    counter = Counter(all_data)  # count and get the most common words
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))

    words = ['<PAD>'] + list(words)  # add a padding with id 0 to pad the sentence to same length
    open_file(vocab_dir, 'w').write('\n'.join(words) + '\n')


def read_vocab(vocab_file):
    """
    Read vocabulary from file.
    """
    words = open_file(vocab_file).read().strip().split('\n')
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def read_vocab_glove(dict_path):
    """
    Read vocabulary from glove, get dictionary
    """

    content = pd.read_csv(filepath_or_buffer=dict_path, header=None, sep=" ", quoting=csv.QUOTE_NONE,
                            usecols=[0]).values

    print (content)
    words = [word[0] for word in content]

    print ('Glove Loaded...',len(words),dict_path)
    print (words[:100])

    word_to_id = dict(zip(words, range(len(words))))

    return words, word_to_id

def process_text(text, word_to_id, max_length, clean=True):
    """tokenizing and padding"""
    if clean:  # if the data needs to be cleaned
        text = clean_str(text)
    # text = text.split()

    text = [word_to_id[x] for x in text if x in word_to_id] # not considering OOV
    if len(text) < max_length:
        text = [0] * (max_length - len(text)) + text
    return text[:max_length]



def process_text_tokenize(text, clean=True):
    """tokenizing and padding"""
    if clean:  # if the data needs to be cleaned
        text = clean_str(text)
    text = text.split()

    # return text[:max_length]
    return text



def read_test(test_path,abbre):
    # abbre = test_path.split('/')[-1].split('_')[1]

    X=[]
    y=[]

    xl = pd.read_excel(open(test_path, 'rb'), header=None) #xl is a DataFrame

    print ('*'*100)

    for i in range(len(xl)):
        # if xl[0][i] == 'Y' or (pd.isnull(xl[0][i]) and xl[1][i].startswith('http')):
        if str(xl[0][i]).lower().startswith('y') or (pd.isnull(xl[0][i]) and str(xl[1][i]).startswith('http')):
            content = xl[6][i].strip().replace('\n', '').replace('\t', '')
            sentence = content.replace('***** ' + abbre + ' *****', token)
            X.append(sentence)
            y.append(xl[2][i])

    # for line in spamreader:
    #     print (line)
    #     if line[0] == 'Y':
    #     if len(line) == 7 and line[0] == 'Y':
    #         content = line[-1].strip()
    #         sentence = content.replace('***** ' + abbre + ' *****', token)
    #         X.append(sentence)
    #         y.append(line[2])

    print ('Loaded...',len(X))

    return X,y



class Corpus(object):
    """
    Preprocessing training data.
    """

    def __init__(self, file_path, test_path, abbre, max_length=50, vocab_size=8000,over_sample=False):

        x_data = []
        labels = []
        # load train
        with open(file_path, 'r', encoding="ISO-8859-1") as f:
            for line in f.readlines():
                if line.startswith('\"'):
                    line = line[1:-2]

                meaning = line.split('|')[0]

                content = line.split('|')[1]

                labels.append(meaning)
                x_data.append(content)

        n_train = len(x_data)

        print ('Train Loaded..',len(x_data))
        X_test, y_test = read_test(test_path,abbre)
        labels += y_test
        x_data += X_test

        print ('Everything is loaded...',len(x_data))

        # convert labels

        le = preprocessing.LabelEncoder()
        y_data = le.fit_transform(labels)



        vocab_file = 'tmp_vocab'
        if not os.path.exists(vocab_file):
            build_vocab(x_data, vocab_file, vocab_size)

        # this works good
        # self.words, self.word_to_id = read_vocab(vocab_file)

        # this is used when using glove
        self.words, self.word_to_id = read_vocab_glove(glove_file)

        # os.remove(vocab_file)



        for i in range(len(x_data)):
            # tokenize not padding
            x_data[i] = process_text_tokenize(x_data[i], clean=True)

        # splitting
        x_data_text = x_data.copy()
        for i in range(len(x_data)):  # tokenizing and padding
            x_data_text[i] = process_text(x_data[i], self.word_to_id, max_length, clean=False)
        x_data_text = np.array(x_data_text)
        # print ('x_data_text shapeeee',np.shape(x_data_text)) # (530, 200)


        self.num_classes = len(set(y_data))

        # x_data = np.array(x_data)
        y_data = np.array(y_data)


        # a list of things

        self.x_train = x_data[:n_train]
        self.y_train = y_data[:n_train]
        self.x_test = x_data[n_train:]
        self.y_test = y_data[n_train:]

        self.x_train_text = x_data_text[:n_train]
        self.x_test_text = x_data_text[n_train:]

        # shuffle training set

        self.x_train, self.y_train, self.x_train_text = shuffle(self.x_train, self.y_train,self.x_train_text)

        print('Before...', np.shape(self.x_train_text), np.shape(self.y_train))

        if over_sample:
            # TODO: note that this is only used when not using ELMo
            from imblearn.over_sampling import RandomOverSampler
            ros = RandomOverSampler(random_state=0,return_indices=True) #return indices
            self.x_train_text, self.y_train, resampled_id = ros.fit_resample(self.x_train_text,self.y_train)
            # make up self.x_train
            new_x_train=[]
            for id in resampled_id:
                new_x_train.append(self.x_train[0])
            self.x_train = new_x_train

            print('After...', np.shape(self.x_train_text), np.shape(self.y_train))

        # print(x_data[:3])
        self.x_train_ids = batch_to_ids(self.x_train)
        self.x_test_ids = batch_to_ids(self.x_test)

        # print ('ID shape for CNN...',np.shape(self.x_train_ids[:3])) # this is id (3,200)
        # print ('Text shape for LSTM...',np.shape(self.x_train_text[:3]))



    def __str__(self):
        return 'Training: {}, Testing: {}, Vocabulary: {}'.format(len(self.x_train), len(self.x_test), len(self.words))



# testing code
# file_path='/home/lily/zl379/BioNLP/Disambiguation/50_normal/AB.txt'
# test_path='/home/lily/zl379/BioNLP/test_data3/OP_fill_done.xlsx'
# #
# Corpus(file_path, test_path, 'OP', over_sample=True)
