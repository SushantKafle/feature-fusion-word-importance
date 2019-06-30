import os

class Config():
    def __init__(self):
        self.setup()

    def setup(self):
        # directory for training outputs
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)


    def re_init(self, output_path):
        self.output_path = output_path
        self.model_output = self.output_path + "model.weights/"
        self.log_path = self.output_path + "log.txt"
        self.predictions_path = self.output_path + "prediction_da.csv"
        self.setup()

    speech_keys_path = "data/speech_keys"
    text_words_path = "data/words"
    text_chars_path = "data/chars"

    word_embeddings_trimmed_path = "data/embeddings.npz"
    speech_features_path = "data/feats.npz"

    text_word_vocab_size = 2000

    word_features_dim = 300
    speech_features_dim = 30
    speech_lexical_features_dim = 6

    train_setup = "default"

    input_type = "attn-fuse"
    speech_features = "all-raw"

    random_seed = 10
    loss_factor = 0.8

    # general config
    output_path = "saved_models/attn-fuse_tanh_lf_0.800000_3/"
    model_output = output_path + "model.weights/"
    log_path = output_path + "log.txt"
    predictions_path = output_path + "prediction_da.csv"

    # other hyperparameters
    word_rnn_size = 128
    speech_rnn_size = 64

    # training
    train_embeddings = False #True
    nepochs = 100
    dropout = 0.5
    batch_size = 20 #120
    lr_method = "adam"
    lr = 0.001
    lr_decay = 1
    clip = -1 # if negative, no clipping
    nepoch_no_imprv = 7
    reuse_model = False

