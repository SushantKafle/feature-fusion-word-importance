import numpy as np
import os
from collections import defaultdict
import operator


# shared global variables to be imported from model also
UNK = "$UNK$"
NUM = "$NUM$"
NONE = "O"

# special error message
class MyIOError(Exception):
    def __init__(self, filename):
        # custom error message
        message = """
ERROR: Unable to locate file {}.

FIX: Have you tried running python build_data.py first?
This will build vocab file from your train, test and dev sets and
trimm your word vectors.
""".format(filename)
        super(MyIOError, self).__init__(message)



class AnnotationDataset(object):

    def __init__(self, filename, word_processor = None, speech_processor = None):
        self.filename = filename
        self.processing_word = word_processor
        self.processing_speech = speech_processor
        self.length = None

    def __iter__(self):
        with open(self.filename) as f:
            words, speech_keys, tags = [], [], []
            for line in f:
                line = line.strip()
                if (len(line) == 0):
                    if len(words) != 0:
                        yield words, speech_keys, tags
                        words, speech_keys, tags = [], [], []
                else:
                    ls = line.split(' ')
                    word, speech_key, tag = ls[0], ls[1], ls[-1]
                    if self.processing_word is not None:
                        word = self.processing_word(word)
                    if self.processing_speech is not None:
                        speech_key = self.processing_speech(speech_key)
                    words += [word]
                    speech_keys += [speech_key]
                    tags += [tag]


    def __len__(self):
        """
        Iterates once over the corpus to set and store length
        """
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length


def get_vocab(datasets, text=True, limit=-1):
    print("Building vocab...")
    vocab = defaultdict(int)

    data_id = 0 if text else 1
    for dataset in datasets:
        for sent in dataset:
            data, tags = sent[data_id], sent[-1]
            if type(data[0]) in [list, tuple]:
                for val in data:
                    for x in val:
                        vocab[x] += 1
                    #vocab.update(x)
            else:
                #vocab.update(data)
                for val in data:
                    vocab[val] += 1

    if limit != -1:
        vocab = sorted(vocab.items(), key=operator.itemgetter(1), reverse=True)
        vocab = set(list(zip(*vocab[:limit]))[0])
    else:
        vocab = set(vocab.keys())


    print("- done. {} tokens".format(len(vocab)))
    return vocab


def get_char_vocab(dataset):
    vocab_char = set()
    for words, _, _ in dataset:
        for word in words:
            vocab_char.update(word)

    return vocab_char


def write_vocab(vocab, filename):
    """
    Writes a vocab to a file

    Args:
        vocab: iterable that yields word
        filename: path to vocab file
    Returns:
        write a word per line
    """
    print("Writing vocab...")
    with open(filename, "w") as f:
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write("{}\n".format(word))
            else:
                f.write(word)
    print("- done. {} tokens".format(len(vocab)))


def load_vocab(filename):
    """
    Args:
        filename: file with a word per line
    Returns:
        d: dict[word] = index
    """
    try:
        d = dict()
        with open(filename) as f:
            for idx, word in enumerate(f):
                word = word.strip()
                d[word] = idx

    except IOError:
        raise MyIOError(filename)
    return d


def get_glove_vocab(filename):
    print("Building vocab...")
    vocab = set()
    with open(filename) as f:
        for line in f:
            word = line.strip().split(' ')[0]
            vocab.add(word)
    print("- done. {} tokens".format(len(vocab)))
    return vocab


def export_trimmed_glove_vectors(vocab, glove_filename, trimmed_filename, dim):
    embeddings = np.zeros([len(vocab), dim])
    with open(glove_filename) as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)

    np.savez_compressed(trimmed_filename, features=embeddings)


def export_speech_features(vocab, trimmed_filename, dim):
    from lookup_feats import get_interval_feature
    features = np.zeros([len(vocab), dim])

    for key in vocab.keys():
        key_idx = vocab[key]
        features[key_idx] = combine_features(get_interval_feature(key, "real"), get_interval_feature(key))
    np.savez_compressed(trimmed_filename, features=features)

'''
Assumes features are of same size.
'''
def combine_features(feat_1, feat_2):
    feat = []
    for i, x in enumerate(feat_1):
        feat.append(x)
        feat.append(feat_2[i])
    return feat


def get_trimmed_features(filename):
    """
    Args:
        filename: path to the npz file
    Returns:
        matrix of embeddings (np array)
    """
    try:
        with np.load(filename) as data:
            return data["features"]

    except IOError:
        raise MyIOError(filename)


def get_processing_word(vocab_words=None, vocab_chars=None,
                    lowercase=False, chars=False):
    def f(word):
        # 0. get chars of words
        if vocab_chars is not None and chars == True:
            char_ids = []
            for char in word:
                # ignore chars out of vocabulary
                if char in vocab_chars:
                    char_ids += [vocab_chars[char]]

        # 1. preprocess word
        if lowercase:
            word = word.lower()
        if word.isdigit():
            word = NUM

        # 2. get id of word
        if vocab_words is not None:
            if word in vocab_words:
                word = vocab_words[word]
            else:
                word = vocab_words[UNK]

        # 3. return tuple char ids, word id
        if vocab_chars is not None and chars == True:
            return char_ids, word
        else:
            return word

    return f


def get_processing_speech(vocab_speech = None):
    """
    Function: 
        Return function that maps the speech key to a set of interval and word ids
    Args:
        vocab: dict[key] = idx
    Returns:
        f("2005-A-10-1-5") = ([12, 4, 32], 12345)
                 = (list of interval ids, word id)
    """
    def f(key):
        num_interval = int(key.split("-")[-1])
        word_key = "-".join(key.split("-")[:-1])

        # 0. get interval of words
        intervals = []
        for interval in range(num_interval):
            intervals.append(word_key + "-" + str(interval))

        # 1. get interval_ids (if possible)
        if vocab_speech is not None:
            interval_ids = []
            for interval in intervals:
                interval_ids.append(vocab_speech[interval])
            return interval_ids
        else:
            return intervals

    return f


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
        sequence_padded +=  [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    if nlevels == 1:
        max_length = max(map(lambda x : len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences,
                                            pad_tok, max_length)

    elif nlevels == 2:
        max_length_word = max([max(map(lambda x: len(x), seq)) for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x : len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded, [pad_tok]*max_length_word,
                                            max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0, max_length_sentence)


    return np.asarray(sequence_padded), sequence_length


def pad_ordinally(sequences, pad_tok, ntags):
    max_length = max(map(lambda x: len(x), sequences))
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        seq = list(seq)
        ordinal_seq = []
        for label in seq:
            encoded_label = [1] * (label + 1) + [0] * (ntags - label - 1)
            ordinal_seq.append(encoded_label)

        seq_ = ordinal_seq[:max_length] + [[pad_tok] * ntags] * max(max_length - len(seq), 0)
        sequence_padded +=  [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def minibatches(data, minibatch_size):
    """
    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size: (int)
    Returns:
        list of tuples
    """
    x_batch, y_batch, z_batch = [], [], []
    for (x, y, z) in data:
        if len(x_batch) == minibatch_size:
            yield x_batch, y_batch, z_batch
            x_batch, y_batch, z_batch = [], [], []

        if type(x[0]) == tuple:
            x = list(zip(*x))
        if type(y[0]) == tuple:
            y = list(zip(*y))
        x_batch += [x]
        y_batch += [y]
        z_batch += [z]

    if len(x_batch) != 0:
        yield x_batch, y_batch, z_batch
