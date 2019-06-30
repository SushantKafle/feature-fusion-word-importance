from utils.data_utils import get_trimmed_features
from config import Config
import numpy as np

# get pre-trained embeddings
interval_features = get_trimmed_features(Config.speech_features_path)

feats_vocab = ['word_duration', 'word_duration_z', 'word_pos', 'word_pos_z', 'silence_duration', 'silence_duration_z', 
'syllable_count', 'syllable_count_z', 'avg_syllable_duration', 'avg_syllable_duration_z', 'articulation_rate', 
'articulation_rate_z', 'spectral_tilt', 'spectral_tilt_z', 'hnr', 'hnr_z', 'voiced_unvoiced_ratio', 'voiced_unvoiced_ratio_z', 
'f0_min', 'f0_min_z', 'f0_tmin', 'f0_tmin_z', 'f0_max', 'f0_max_z', 'f0_tmax', 'f0_tmax_z', 'f0_mean', 'f0_mean_z', 'f0_median', 
'f0_median_z', 'f0_range', 'f0_range_z', 'f0_slope', 'f0_slope_z', 'f0_skewness', 'f0_skewness_z', 'f0_std', 'f0_std_z', 
'int_min', 'int_min_z', 'int_tmin', 'int_tmin_z', 'int_max', 'int_max_z', 'int_tmax', 'int_tmax_z', 'int_mean', 'int_mean_z', 
'int_median', 'int_median_z', 'int_range', 'int_range_z', 'int_slope', 'int_slope_z', 'int_skewness', 'int_skewness_z', 'int_std', 
'int_std_z', 'spectral_emphasis', 'spectral_emphasis_z']

feature_categories = {
    "all": feats_vocab,

    "z_norm": [f for i, f in enumerate(feats_vocab) if i % 2 != 0],

    "raw": [f for i, f in enumerate(feats_vocab) if i % 2 == 0],

    "no-lexical": feats_vocab[12:],

    "lexical": feats_vocab[:12],

    "frequency": ['f0_min', 'f0_min_z', 'f0_tmin', 'f0_tmin_z', 'f0_max', 'f0_max_z', 'f0_tmax', 'f0_tmax_z', 'f0_mean', 
    'f0_mean_z', 'f0_median', 'f0_median_z', 'f0_range', 'f0_range_z', 'f0_slope', 'f0_slope_z', 'f0_skewness', 'f0_skewness_z', 'f0_std', 'f0_std_z'],

    "frequency_clean": ['f0_min', 'f0_min_z', 'f0_max', 'f0_max_z', 'f0_mean', 'f0_slope', 'f0_slope_z'],

    "energy": ['int_min', 'int_min_z', 'int_tmin', 'int_tmin_z', 'int_max', 'int_max_z', 'int_tmax', 'int_tmax_z', 
    'int_mean', 'int_mean_z', 'int_median', 'int_median_z', 'int_range', 'int_range_z', 'int_slope', 'int_slope_z', 
    'int_skewness', 'int_skewness_z', 'int_std', 'int_std_z', 'spectral_emphasis', 'spectral_emphasis_z'],

    "energy_clean": ['int_min', 'int_min_z', 'int_max', 'int_max_z', 'int_mean', 'int_mean_z', 'int_slope', 'int_slope_z', 'spectral_emphasis', 'spectral_emphasis_z'],

    "voicing": ['spectral_tilt', 'spectral_tilt_z', 'hnr', 'hnr_z', 'voiced_unvoiced_ratio', 'voiced_unvoiced_ratio_z']
}

def get_subset(feature, ids_):
    return list(feature[i] for i in ids_)

def feats_to_id(feats_):
    ids_ = []
    
    if "-" in feats_:
        features = feats_.split("-")
        feats = set(feature_categories[features[0]])
        ablated_feats = set(feature_categories[features[1]])
        feats = list(feats - ablated_feats)
    else:
        feats = feature_categories[feats_]

    for feature in feats:
        ids_.append(feats_vocab.index(feature))
    return ids_

def get_features(ids_, feats):
    features = []
    feat_ids = feats_to_id(feats)
    for seq in ids_:
        seq_ = []
        for word_seq in seq:
            word_seq_ = []
            for i in word_seq:
                x = get_subset(interval_features[i], feat_ids)
                word_seq_.append(x)
            seq_.append(word_seq_)
        features.append(seq_)
    return np.asarray(features)