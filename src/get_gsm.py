import gensim

from gensim.models.doc2vec import Doc2Vec

model_path='/data/corpora/mimic/experiments/models/dbow_dim_300_negative_5_window_15.model'

model= Doc2Vec.load(model_path)

word_list = model.wv.vocab.keys()

writing = []
for word in word_list:
    embd = model.wv[word]
    embd_str = ' '.join([str(e) for e in embd])
    writing.append(word+' '+embd_str)

with open('300d_w2v.txt','w') as f:
    f.write('\n'.join(writing))
print ('Finished..')
print ('length',len(writing))
