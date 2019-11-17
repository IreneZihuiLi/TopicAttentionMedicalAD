#!/usr/bin/python
# -*- coding: utf-8 -*-

"""

# below is my readme. #####
dir: (my home dir )~/BioNLP/med_cnn_mmc

https://github.com/gaussic/text-classification
This scripts is to train on 500 samples, and test on the annotated samples.
Run  with python3 (****py36**** on tangra)
>> python main_pytorch.py -m [cnn,lstm,att]

Run with parameters
>>time python main_pytorch.py -m rcnn -elmo False -elmo_level 2 -elmo_train False -e 8

>>time python main_pytorch.py -m cnn -elmo True -elmo_level 2 -elmo_train False -e 4



# below is the original readme. #####
This example demonstrates the use of Conv1D for CNN text classification.
Original paper could be found at: https://arxiv.org/abs/1408.5882

This is the baseline model: CNN-rand.

The implementation is based on PyTorch.

We didn't implement cross validation,
but simply run `python cnn_mxnet.py` for multiple times,
the average accuracy is close to 78%.

It takes about 2 minutes for training 20 epochs on a GTX 970 GPU.



"""


import torch
import argparse
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

from configuration import Config
from models.LSTMClassifier import LSTMClassifier
# from models.CNNClassifier import CNNClassifier
# from models.CNN_with_glove import CNNClassifier
from models.CNN import CNNClassifier
# from models.selfAttention_lda import SelfAttention
from models.selfAttention_lda_elmo import SelfAttention
from models.selfAttention_lda_elmo_cnn import SelfAttention

from models.LSTM_Attn import AttentionModel
from models.RCNN import RCNN

options_file = "/home/lily/zl379/Playing/bilm-tf/mmc_new/options.json"
weight_file = "/home/lily/zl379/Playing/bilm-tf/dump/weights.hdf5"

train_path = '/home/lily/zl379/BioNLP/Disambiguation/50_normal_clean/'


# the old
# test_path = '/home/lily/zl379/BioNLP/test_data/'
# testing_abbre = ['PDA','SBP']

# test_path = '/home/lily/zl379/BioNLP/test_data2/'
test_path = '/home/lily/zl379/BioNLP/test_data4/'

# testing_abbre = ['AV','PR','RT','SA','VGB']
# testing_abbre = ['OP','IT']

# final results.. keep results
result_path = 'final_after'
if not os.path.exists(result_path):
    os.mkdir(result_path)
else:
    import shutil
    shutil.rmtree(result_path)
    os.mkdir(result_path)



save_path = 'checkpoints_elmo_topic_cnn'  # model save path
if not os.path.exists(save_path):
    os.mkdir(save_path)
else:
    import shutil
    shutil.rmtree(save_path)
    os.mkdir(save_path)


use_cuda = torch.cuda.is_available()



def get_time_dif(start_time):
    """
    Return the time used since start_time.
    """
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def evaluate(data, model, loss):
    """
    Evaluation, return accuracy and loss
    """
    model.eval()  # set mode to evaluation to disable dropout
    data_loader = DataLoader(data, batch_size=config.batch_size)

    data_len = len(data)
    total_loss = 0.0
    y_true, y_pred = [], []

    for data, data_ids, label in data_loader:
        data, data_ids, label = Variable(data, volatile=True),Variable(data_ids, volatile=True), Variable(label, volatile=True)
        if use_cuda:
            data, data_ids, label = data.cuda(), data_ids.cuda(), label.cuda()
        # if (data.size()[0] is not config.batch_size):  # One of the batch returned by BucketIterator has length different than 32.
        #         #     continue

        output = model(data,data_ids, batch_size=data.size()[0])
        losses = loss(output, label)

        total_loss += losses.data[0]
        pred = torch.max(output.data, dim=1)[1].cpu().numpy().tolist()
        y_pred.extend(pred)
        y_true.extend(label.data)

    acc = (np.array(y_true) == np.array(y_pred)).sum()
    return float(acc) / float(data_len), total_loss / data_len


def train(config):
    """
    Train and evaluate the model with training and validation data.
    """
    print('Loading data...')
    start_time = time.time()

    corpus = Corpus(config.file_path, config.test_path, config.abbre, config.seq_length, config.vocab_size,over_sample=config.over)
    config.num_classes = corpus.num_classes
    print(corpus)
    config.vocab_size = len(corpus.words) #useless now


    config.model_file = config.model_file + '.pk'


    train_data = TensorDataset(torch.LongTensor(corpus.x_train_text), torch.LongTensor(corpus.x_train_ids),torch.LongTensor(corpus.y_train))
    test_data = TensorDataset(torch.LongTensor(corpus.x_test_text), torch.LongTensor(corpus.x_test_ids),torch.LongTensor(corpus.y_test))



    print('Configuring model...', config.elmo)

    if config.model_name =='cnn':

        model = CNNClassifier(config)
        print('You choose to use CNN')
    elif config.model_name=='lstm':
        model = LSTMClassifier(config)
        print('You choose to use LSTM')
    elif config.model_name=='rcnn':
        model = RCNN(config)
        print('You choose to use RCNN')
    elif config.model_name=='self':
        model = SelfAttention(config)
        print('You choose to use Self-Attention')
    else:
        model = AttentionModel(config)
        print('You choose to use LSTM-attention')



    print(model)

    if use_cuda:
        model.cuda()

    # optimizer and loss function
    criterion = nn.CrossEntropyLoss(size_average=False)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # set the mode to train
    print("Training and evaluating...")

    best_acc = 0.0
    for epoch in range(config.num_epochs):
        # load the training data in batch
        model.train()
        train_loader = DataLoader(train_data, batch_size=config.batch_size)
        for x_batch, x_batch_elmo, y_batch in train_loader:

            inputs, inputs_elmo, targets = Variable(x_batch),  Variable(x_batch_elmo),  Variable(y_batch)
            if use_cuda:
                inputs, inputs_elmo, targets = inputs.cuda(), inputs_elmo.cuda(), targets.cuda()
            # if (inputs.size()[0] is not config.batch_size):  # One of the batch returned by BucketIterator has length different than 32.
            #     print ('Size wrong')
            #     continue
            optimizer.zero_grad()
            outputs = model(inputs, inputs_elmo, batch_size=inputs.size()[0])  # forward computation: provide batch_size
            loss = criterion(outputs, targets)

            # backward propagation and update parameters
            loss.backward(retain_graph=True)
            optimizer.step()

        # evaluate on both training and test dataset
        train_acc, train_loss = evaluate(train_data, model, criterion)
        test_acc, test_loss = evaluate(test_data, model, criterion)

        if test_acc > best_acc:
            # store the best result
            best_acc = test_acc
            improved_str = '*'
            torch.save(model.state_dict(), config.model_file)
        else:
            improved_str = ''

        time_dif = get_time_dif(start_time)
        msg = "Epoch {0:3}, Train_loss: {1:>7.2}, Train_acc {2:>6.2%}, " \
              + "Test_loss: {3:>6.2}, Test_acc {4:>6.2%}, Time: {5} {6}"
        print(msg.format(epoch + 1, train_loss, train_acc, test_loss, test_acc, time_dif, improved_str))

    test_acc, test_f1, y_true,y_pred = test(model, test_data, config.model_file)


    return test_acc, test_f1, y_true,y_pred


def test(model, test_data, model_file):
    """
    Test the model on test dataset.
    """
    print("Testing...")
    start_time = time.time()
    test_loader = DataLoader(test_data, batch_size=config.batch_size)

    # restore the best parameters
    model.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))

    y_true, y_pred = [], []
    for data, data_ids, label in test_loader:
        data, data_ids,label = Variable(data, volatile=True), Variable(data_ids, volatile=True), Variable(label, volatile=True)
        if use_cuda:
            data, data_ids, label = data.cuda(), data_ids.cuda(), label.cuda()
        # if (data.size()[0] is not config.batch_size):  # One of the batch returned by BucketIterator has length different than 32.
        #     continue
        output = model(data, data_ids,batch_size=data.size()[0])
        pred = torch.max(output.data, dim=1)[1].cpu().numpy().tolist()
        y_pred.extend(pred)
        y_true.extend(label.data)

    test_acc = metrics.accuracy_score(y_true, y_pred)
    test_f1 = metrics.f1_score(y_true, y_pred, average='macro')
    # print("Test accuracy: {0:>7.2%}, F1-Score: {1:>7.2%}".format(test_acc, test_f1))

    # print("Precision, Recall and F1-Score...")
    # print(metrics.classification_report(y_true, y_pred, target_names=['POS', 'NEG']))

    # print('Confusion Matrix...')
    cm = metrics.confusion_matrix(y_true, y_pred)
    # print(cm)
    print('---' * 100)
    y_true = [m.detach().tolist() for m in y_true]
    print('Y_true\n', y_true)
    print('Y_pred\n', y_pred)

    # print("Time usage:", get_time_dif(start_time))

    return test_acc,test_f1,y_true,y_pred


# def predict(text):
#     # load config and vocabulary
#     config = TCNNConfig()
#     _, word_to_id = read_vocab(vocab_file)
#     labels = ['POS', 'NEG']
#
#     # load model
#     model = TextCNN(config)
#     model.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))
#
#     # process text
#     text = process_text(text, word_to_id, config.seq_length)
#     text = Variable(torch.LongTensor([text]), volatile=True)
#
#     if use_cuda:
#         model.cuda()
#         text = text.cuda()
#
#     # predict
#     model.eval()  # very important
#     output = model(text)
#     pred = torch.max(output, dim=1)[1]
#
#     return labels[pred.data[0]]


if __name__ == '__main__':

    # apply model
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--method', type=str, help='model: cnn,lstm')
    parser.add_argument('-e', '--epoch', type=int, default=5, help='number of epochs')
    parser.add_argument('-elmo', '--elmo', type=str, default='False', help='apply elmo or not')
    parser.add_argument('-elmo_level', '--elmo_level', type=int, default=3, help='elmo layer')
    parser.add_argument('-elmo_train', '--elmo_train', type=str, default='True', help='elmo layer')
    parser.add_argument('-e_dim', '--ebd_dim', type=int, default=100, help='embedding dimension')
    parser.add_argument('-topic', '--topic', type=str, default='False', help='embedding dimension')
    parser.add_argument('-over', '--over', type=str, default='False', help='embedding dimension')

    args = parser.parse_args()
    model_name = args.method or 'cnn'

    print(parser.parse_args())

    # iterate each small dataset
    file_list = [f for f in os.listdir(train_path)]
    test_file_list = [f for f in os.listdir(test_path) if f.endswith('.xlsx')]
    accuracy = []
    f1 = []
    print ('There are...',test_file_list)
    for i,test_file in enumerate(test_file_list):
        config = Config(train_path,test_path)

        config.abbre = test_file.split('_')[0]

        # set model
        config.model_name = model_name
        config.elmo = args.elmo
        config.elmo_level=args.elmo_level
        config.elmo_train = args.elmo_train
        config.num_epochs=args.epoch
        config.embedding_dim=args.ebd_dim
        config.use_topic=args.topic

        config.file_path = train_path + config.abbre + '.txt'
        config.test_path = test_path + test_file
        config.model_file = os.path.join(save_path, config.abbre)

        # over sampling
        config.over = False if args.over == 'False' else 'True'


        # show
        config.show_params()

        score, f1_score ,y_true,y_pred= train(config)

        # write out results.. into the file
        with open(os.path.join(result_path, config.abbre + '.txt'), 'w') as f:
            f.write('\t'.join([str(s) for s in y_true]))
            f.write('\n')
            f.write('\t'.join([str(s) for s in y_pred]))
        print('Writing file finished...', os.path.join(result_path, config.abbre + '.txt'))


        accuracy.append(score)
        f1.append(f1_score)
        print(config.abbre, score, f1_score)
        print('Finished on...', (i + 1))

    print('Tested on...', len(test_file_list))
    for id, file_name in enumerate(test_file_list):
        print (file_name,accuracy[id],f1[id])
    print ('On average...,',sum(accuracy) / len(accuracy),sum(f1) / len(f1))


    # file_list = [f for f in os.listdir(test_path)]
    # accuracy = []
    # f1 = []
    # for f in file_list:
    #     config = TCNNConfig()
    #     # update train and test path
    #     config.file_path += f
    #     config.test_path += f
    #     config.model_file = os.path.join(save_path, f)
    #
    #     score, f1_score = train(config)
    #
    #     accuracy.append(score)
    #     f1.append(f1_score)
    #     print(f, score, f1_score)
    #
    # print('Average acc %0.4f' % (sum(accuracy) / len(accuracy)))
    # print('Average f1 %0.4f' % (sum(f1) / len(f1)))

