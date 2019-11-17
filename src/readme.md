## Models
In `models`, you will find a list of implemented models including CNN, LSTM, LSTM with attiontion, and our topic-attention model with LSTM.

## Data
Once should get access to MIMIC-III can have a full set of our datasets for training and testing. However, we provide some samples in `Sample_Data`.

## Code
Implemented in PyTorch, tested on Python 3.

Run with `main_pytorch.py` script. It provides a list of models (`-m`: cnn, lstm, atten, etc). 
 
Run with self-attention model without pre-trianed ELMO, with out topic-attention: `python main_pytorch.py -m self -elmo True -e 10 -topic False`.

Run our proposed model: `python main_pytorch.py -m self -elmo True -e 10 -topic True`. 

## Pre-trained embeddings.
ELMO model, you need `options.json` and `weights.hdf5`. We also provide pretrained word embeddings `300d_w2v.txt`. Please check this google drive [link](https://drive.google.com/open?id=1RHEMl2Y0AQsKzWdpYjGRttETIqAgZoij) to get them.