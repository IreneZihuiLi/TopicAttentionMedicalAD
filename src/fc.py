'''

CNN model for text classification
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from load import preprocess
import torch.utils.data as Data

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

TEST_SIZE=1000

class Net(torch.nn.Module):
    def __init__(self, n_feature, max_len):
        super(Net, self).__init__()
        self.n_feature = n_feature
        self.max_len = max_len
        self.embedding = nn.Embedding(self.n_feature, 50)

        self.hidden = torch.nn.Linear(self.max_len * 50, 100)   # hidden layer
        self.hidden2 = torch.nn.Linear(100, 10)
        self.out = torch.nn.Linear(10, 2)   # output layer

    def forward(self, input):
        x = self.embedding(input)
        x = x.view(-1,self.max_len * 50)
        x = self.hidden(x)
        x = F.relu(self.hidden2(x))      # activation function for hidden layer
        x = self.out(x)

        return x
#
# cnn = CnnTextClassifier(18587)
# print(cnn)  # net architecture




# data loading
X,y,vcab_size = preprocess()
X = torch.tensor(X, dtype=torch.long)
y = torch.tensor(y, dtype=torch.long)
# use data batch loader
torch_dataset = Data.TensorDataset(X[:-TEST_SIZE], y[:-TEST_SIZE])

loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=500,      # mini batch size
    shuffle=True,               # random shuffle for training
    num_workers=2,              # subprocesses for loading data
)

net = Net(vcab_size,53)
print (net)



optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted



for epoch in range(50):   # train entire dataset 3 times
    for (batch_x, batch_y) in loader:

        out = net(batch_x)
        loss = loss_func(out, batch_y)

        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients


    # eval
    # print 10 predictions from test data
    test_output= net(X[-TEST_SIZE:])
    correct = 0
    pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    correct += (pred_y == y[-TEST_SIZE:]).sum().item()
    print(loss,float(correct)/TEST_SIZE)

