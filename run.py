import os, csv
import numpy as np
from utils.data_utils import get_trimmed_features, load_vocab, \
    get_processing_word, get_processing_speech, AnnotationDataset, \
    pad_sequences
from config import Config
import tensorflow as tf
from SpeechTextModel import WImpModel


EVAL_FOLDER = "examples"

def read_csv(path, map_func):
    data = []
    with open(path) as f:
        csv_reader = csv.reader(f)
        next(csv_reader, None)
        for row in csv_reader:
            row = map_func(row)
            data.append(row)
    return data


def read_examples(path, f):
    examples = {}
    for split_folder in os.listdir(path):
        split_folder_ = os.path.join(path, split_folder)
        if os.path.isfile(split_folder_):
            continue

        for sent_folder in os.listdir(split_folder_):
            sent_folder_ = os.path.join(split_folder_, sent_folder)
            if os.path.isfile(sent_folder_):
                continue

            wrd_key = "%s-%s" % (split_folder, sent_folder)
            int_feats = os.listdir(os.path.join(sent_folder_, 'words'))
            int_feats = [f for f in int_feats if f.endswith('.norm')]

            examples[sent_folder] = []
            for word_id in range(len(int_feats)):
                interval_key = "%s-%d" % (wrd_key, word_id)
                word_file = os.path.join(os.path.join(sent_folder_, 'words', str(word_id) + ".norm"))
                word_ = read_csv(word_file, lambda x: str(x[0]))

                
                map_func = lambda x: list(map(float, x[1:]))
                norm_feats = np.asarray(read_csv(word_file, map_func))
                real_feats = np.asarray(read_csv(word_file[:-5] + "_.csv", map_func))

                '''all_feats = np.zeros((len(norm_feats), 60))
                all_feats[:, 0::2] = real_feats
                all_feats[:, 1::2] = norm_feats'''

                examples[sent_folder].append((word_[0], f(word_[0]), norm_feats))
                
    return examples


def main(config, eval_folder):

    # local the vocab file

    text_words_vocab = load_vocab(config.text_words_path)
    text_chars_vocab = load_vocab(config.text_chars_path)
    inv_text_vocab = {v: k for k, v in text_words_vocab.items()}

    # get the processing function
    processing_word = get_processing_word(text_words_vocab, text_chars_vocab,
        lowercase=True, chars=True)

    #load features:
    word_features = get_trimmed_features(config.word_embeddings_trimmed_path)


    examples = read_examples(eval_folder, processing_word)

    # build WImpModel
    
    model = WImpModel(config, word_features, None, text_words_vocab["$UNK$"],
        inv_text_vocab, None)
    model.build_graph()


    words, word_feats, speech_interval_feats = [], [], []
    for sent_key in examples.keys():
        words_, word_feats_, speech_feats_ = zip(*examples[sent_key])
        word_feats_ = list(zip(*word_feats_))

        word_feats.append(word_feats_)
        speech_interval_feats.append(speech_feats_)
        words.append(words_)



    speech_interval_feats_pad_, speech_lengths = pad_sequences(speech_interval_feats, 
        pad_tok = [0] * config.speech_features_dim, nlevels=2)
    speech_feats = speech_interval_feats_pad_[:, :, :, config.speech_lexical_features_dim:]
    speech_lexical_feats = speech_interval_feats_pad_[:, :, 0, :config.speech_lexical_features_dim]

    feed, sequence_lengths = model.get_feed_dict(words=word_feats, dropout=1.0)
    feed[model.speech_features] = speech_feats
    feed[model.speech_lexical_features] = speech_lexical_feats
    feed[model.speech_lengths] = speech_lengths

    predictions = model.test(feed)
    

    print ("\n")
    print ("WORD IMPORTANCE PREDICTION OUTPUT")
    print ("=================================")
    for sent_id in range(len(words)):
        scores = predictions[0][:sequence_lengths[sent_id]]
        tokens = words[sent_id]
        result = ["%s (%f)" % (w, s) for w, s in zip(tokens, scores)]
        print ("--> " + " ".join(result) + "\n")


if __name__ == "__main__":
    config = Config()
    main(config, EVAL_FOLDER)
