


class Config(object):
    """
    CNN Parameters
    """
    def __init__(self, train_path,test_path):

        self.file_path = train_path
        self.test_path = test_path
        # set abbreviation
        self.abbre = 'AB' # as default

        # set model to be used
        self.model_name = 'cnn'
        self.elmo = 'False'
        self.elmo_level = 3
        self.elmo_train='True'

        self.embedding_dim = 100  # embedding vector size
        self.seq_length = 179  # maximum length of sequence
        self.vocab_size = 8000  # most common words

        self.num_filters = 100  # number of the convolution filters (feature maps)
        self.kernel_sizes = [3, 4, 5]   # three kind of kernels (windows)
        self.hidden_dim = 100  # hidden size of fully connected layer

        self.dropout_prob = 0.5  # how much probability to be dropped
        self.learning_rate = 1e-3  # learning rate
        self.batch_size = 4  # batch size for training
        self.num_epochs = 5  # total number of epochs

        self.num_classes = 3  # number of classes

        self.dev_split = 0.2  # percentage of dev data

        self.model_file = '' # save path
        self.k_fold = 0

        self.word2vec_dim = 100
        # self.word2vec_path = '/home/lily/zl379/data/GloVe/glove.6B.100d.txt'
        self.word2vec_path='300d_w2v.txt'

        # self.lda_path = '/home/lily/zl379/BioNLP/LDA/topics_full_50.txt'
        self.lda_path = '/home/lily/zl379/BioNLP/LDA/topics_clean.txt'
        self.use_topic = 'True'

        # doc2vec word embedding
        self.dwv_path = '/data/corpora/mimic/experiments/medical_term/mmc_300d.txt'
        self.doc2vec_dim = 300

        # oversampling
        self.over = False


    def show_params(self):
        print ('--'*50)
        print ('Configurations:')

        print ('Model ',self.model_name)
        print('If Elmo ', self.elmo)
        print ('Elmo level ',self.elmo_level)
        print ('If fine-tune ', self.elmo_train)
        print ('File_path ',self.file_path)
        print('Testing_path ',self.test_path)
        print ('Model saving path ',self.model_file)
        print ('Oversampling? ',self.over)
        print('--' * 50)

