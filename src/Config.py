
train_path = '/home/lily/zl379/BioNLP/Disambiguation/50_normal/'


test_path = '/home/lily/zl379/BioNLP/test_data/'




class Config(object,train_path,test_path):
    """
    CNN Parameters
    """

    file_path = train_path
    test_path = test_path
    # set abbreviation
    abbre = 'AB' # as default

    # set model to be used
    model_name = 'cnn'
    elmo = 'True'
    elmo_level = 3
    elmo_train='True'

    embedding_dim = 64  # embedding vector size
    seq_length = 200  # maximum length of sequence
    vocab_size = 8000  # most common words

    num_filters = 100  # number of the convolution filters (feature maps)
    kernel_sizes = [3, 4, 5]   # three kind of kernels (windows)
    hidden_dim = 64  # hidden size of fully connected layer

    dropout_prob = 0.5  # how much probability to be dropped
    learning_rate = 1e-3  # learning rate
    batch_size = 10  # batch size for training
    num_epochs = 5  # total number of epochs

    num_classes = 3  # number of classes

    dev_split = 0.2  # percentage of dev data

    model_file = '' # save path
    k_fold = 0


    def show_params(self):
        print ('Configurations:')

        print ('Model ',self.model_name)
        print('If Elmo ', self.elmo)
        print ('Elmo level ',self.elmo_level)
        print ('If fine-tune ', self.elmo_train)
        print ('File_path ',self.file_path)
        print('Testing_path ',self.test_path)
        print ('Model saving path ',self.model_file)


