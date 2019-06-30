import numpy as np
import os, csv, math
import tensorflow as tf
from utils.data_utils import minibatches, pad_sequences, pad_ordinally
from utils.model_utils import create_feedforward, get_rnn_cell
from utils.lookup import get_features


class WImpModel(object):

    def __init__(self, config, embeddings, speech_features, unk_id, inverse_vocab, inverse_speech_vocab):
            self.config     = config
            self.word_lookup_embeddings = embeddings
            self.unk_id = unk_id
            self.speech_lookup_embeddings = speech_features
            self.inverse_vocab = inverse_vocab
            self.inverse_speech_vocab = inverse_speech_vocab
            self.vocabulary_size = len(inverse_vocab)
            self.rng = np.random.RandomState(self.config.random_seed)
            self.eps = 1e-16


    def create_initializer(self, size):
        """ Netural net layer initializer. 
        """
        return tf.constant(np.asarray(self.rng.normal(loc = 0.0, scale = 0.1,
            size = size), dtype = np.float32))


    def init_graph(self):
        """Define placeholders needed for the graph.
        """

        #shape: batchsize, max sentence length, max word length, max intervals
        self.speech_features = tf.placeholder(tf.float32, shape=[None, None, None, None])

        #shape: batchsize, max sentence length, max word length
        self.speech_lexical_features = tf.placeholder(tf.float32, shape=[None, None, None])

        #shape: batchsize, max sentence length
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None],
            name="word_ids")

        #shape: batchsize, max sentence length, max word
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None],
            name="char_ids")

        #shape: batchsize
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],
                        name="sequence_lengths")

        #shape: batch_size, max sentence length
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None],
                        name="word_lengths")

        #shape: batch_size, max sentence length
        self.speech_lengths = tf.placeholder(tf.int32, shape=[None, None],
                        name="speech_lengths")

        self.labels = tf.placeholder(tf.float32, shape=[None, None],
                                name="labels")

        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                        name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], 
                        name="lr")


    def _define_input_text_(self):
        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(self.word_lookup_embeddings, name = "_word_embeddings", 
                dtype = tf.float32, trainable = self.config.train_embeddings)
            word_embeddings = tf.nn.embedding_lookup(_word_embeddings, self.word_ids, 
                name = "word_embeddings")
        self.word_embeddings =  word_embeddings


    def _define_input_speech_(self):
        with tf.variable_scope("speech_intervals"):
            s = tf.shape(self.speech_features)
            feats_dim = self.config.speech_features_dim - self.config.speech_lexical_features_dim

            interval_features = tf.reshape(self.speech_features, shape=[-1, s[-2], feats_dim])
            speech_lengths = tf.reshape(self.speech_lengths, shape=[-1])

            cell_fw = get_rnn_cell(self.config.speech_rnn_size, "GRU")
            cell_bw = get_rnn_cell(self.config.speech_rnn_size, "GRU")
            _, (output_fw,  output_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, 
                interval_features, sequence_length=speech_lengths, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            feats_dim = 2 * self.config.speech_rnn_size

            ## combine the lexical features with the learned features
            speech_lexical_feats = tf.reshape(self.speech_lexical_features, 
                shape=[-1, self.config.speech_lexical_features_dim])
            output_ = tf.concat([output, speech_lexical_feats], axis=-1)
            feats_dim = feats_dim + self.config.speech_lexical_features_dim
            features = tf.reshape(output_, shape=[-1, s[1], feats_dim])

        self.speech_embeddings =  features
        self.speech_embeddings_dim = feats_dim


    def define_input(self):
        """ Defines the input for the model (input can be speech-based, text-based or a 
            combination of both.) 

            Initializes input and input_size variable. 
        """
        with tf.variable_scope(self.config.input_type):
            if self.config.input_type == "speech":
                self._define_input_speech_()
                self.input = self.speech_embeddings
                self.input_size = self.speech_embeddings_dim
            elif self.config.input_type == "text":
                self._define_input_text_()
                self.input = self.word_embeddings
                self.input_size = self.config.word_features_dim
            else:
                self._define_input_text_()
                self._define_input_speech_()
                ntime_steps = tf.shape(self.word_embeddings)[1]

                if self.config.input_type == "combine":
                    self.input = tf.concat([self.word_embeddings, self.speech_embeddings], 
                        axis=-1)
                    self.input_size = self.config.word_features_dim + self.speech_embeddings_dim
                    self.input = tf.reshape(self.input, [-1, self.input_size])

                elif self.config.input_type == "attn-fuse":
                    h_w = tf.reshape(self.word_embeddings, 
                        [-1, self.config.word_features_dim])

                    h_s = tf.reshape(self.speech_embeddings, 
                        [-1, self.speech_embeddings_dim])
                    h_s = create_feedforward(h_s, 
                        self.speech_embeddings_dim, self.config.word_features_dim, 
                        self.create_initializer, "tanh", "speech_to_word_size")

                    h_ws = tf.concat([h_w, h_s], axis=-1)
                    self.alpha = create_feedforward(h_ws, 2 * self.config.word_features_dim,
                        self.config.word_features_dim, self.create_initializer, "tanh", "h_ws_tanh")

                    h_s_delta = tf.multiply(h_s, self.alpha)

                    self.input = h_w + h_s_delta
                    self.input_size = self.config.word_features_dim

                self.input = tf.reshape(self.input, [-1, ntime_steps, self.input_size])
            self.input =  tf.nn.dropout(self.input, self.dropout)


    def define_logits(self):
        """ Defines how the logits are computed.
        """
        with tf.variable_scope("bi-lstm"):
            cell_fw = get_rnn_cell(self.config.word_rnn_size, "LSTM")
            cell_bw = get_rnn_cell(self.config.word_rnn_size, "LSTM")
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, 
                cell_bw, self.input, sequence_length=self.sequence_lengths, 
                dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)

        ntime_steps = tf.shape(output)[1]
        output = tf.reshape(output, [-1, 2 * self.config.word_rnn_size])

        self.predictions = create_feedforward(output, 2 * self.config.word_rnn_size, 1, 
            self.create_initializer, "sigmoid", "final_projection")
        self.predictions = tf.reshape(self.predictions, [-1, ntime_steps])



    def define_loss(self):
        """ Declare the loss to minimize.
        """
        Y = tf.reshape(self.labels, [-1, 1])
        pred_Y = tf.reshape(self.predictions, [-1, 1])
        self.loss = tf.sqrt(tf.reduce_mean(tf.pow(pred_Y - Y, 2)))

        misc_loss = 0

        if self.config.input_type == "attn-fuse":
            attention_weights = self.alpha + self.eps
            misc_total_loss = -tf.log(tf.math.abs(attention_weights))
            mask = tf.reshape(tf.equal(self.word_ids, self.unk_id), [-1])
            misc_loss = tf.reduce_mean(tf.boolean_mask(misc_total_loss, mask))
        
        self.loss += (self.config.loss_factor * misc_loss)


    def setup_optimizer(self):
        with tf.variable_scope("optimizer_setup"):
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.optimize_ = optimizer.minimize(self.loss)


    def build_graph(self):
        self.init_graph()
        self.define_input()
        self.define_logits()
        self.define_loss()
        self.setup_optimizer()
        self.init = tf.global_variables_initializer()


    def get_feed_dict(self, words, speech = None, labels = None, lr = None, dropout = None):
        feed = {}
        if speech is not None:
            speech_interval_feats = get_features(speech, self.config.speech_features)
            speech_interval_feats_pad_, speech_lengths = pad_sequences(speech_interval_feats, 
                pad_tok = [0] * self.config.speech_features_dim, nlevels=2)
            speech_feats = speech_interval_feats_pad_[:, :, :, self.config.speech_lexical_features_dim:]
            speech_lexical_feats = speech_interval_feats_pad_[:, :, 0, :self.config.speech_lexical_features_dim]

            feed[self.speech_features] = speech_feats
            feed[self.speech_lexical_features] = speech_lexical_feats
            feed[self.speech_lengths] = speech_lengths

        char_ids, word_ids = list(zip(*words))
        word_ids, sequence_lengths = pad_sequences(word_ids, 0)
        char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0, nlevels=2)

        feed[self.word_ids] = word_ids
        feed[self.char_ids] = char_ids
        feed[self.sequence_lengths] = sequence_lengths
        feed[self.word_lengths] = word_lengths

        if labels is not None:
            labels, _ = pad_sequences(labels, 0)
            feed[self.labels] = labels

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed, sequence_lengths


    def predict_batch(self, sess, words, speech=None):
        fd, sequence_lengths = self.get_feed_dict(words = words, speech = speech, dropout=1.0)
        predictions, sequence_lengths = sess.run(self.predictions, feed_dict=fd)
        return predictions, sequence_lengths


    def reset(self):
        tf.reset_default_graph()


    def test(self, feed):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.config.model_output)
            predictions = sess.run(self.predictions, feed_dict=feed)
            return predictions
            
