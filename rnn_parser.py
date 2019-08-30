import os
import sys
import numpy as np
import librosa
from fnmatch import fnmatch
from scipy.io import wavfile
from statsmodels.tools import categorical

nb_classes = 30
n_mfcc=26

dataset_path = '/calc/SHARED/speech_commands_v0.02'
validation_list = [line.rstrip('\n') for line in open(os.path.join(dataset_path, 'validation_list.txt'))]
testing_list = [line.rstrip('\n') for line in open(os.path.join(dataset_path, 'testing_list.txt'))]


def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size / 2)

def in_group(path, group='train'):
    if group is 'test':
        return path in testing_list
    elif group is 'validation':
        return path in validation_list
    else:
        return path not in testing_list and path not in validation_list

def extract_features(root, group):
    assert group in ['train', 'validation', 'test']
    files = []
    labels = []
    features = []
    i=0
    maxs = []
    max_len = 0
    for path, subdirs, files in os.walk(root):
        for file_name in files:
            full_path = os.path.join(path, file_name)
            if fnmatch(file_name, '*.wav') and full_path not in files:
                name = full_path[len(root)+1:full_path.rfind('/')]
                if not in_group(os.path.join(name, file_name), group):
                    continue
                # sample_rate, sample = wavfile.read(full_path)
                sample, sample_rate = librosa.load(full_path)

                mfccs = librosa.feature.mfcc(y=sample, sr=sample_rate, n_mfcc=n_mfcc)
                # print(mfccs.shape)  # (n_mfcc, len)
                length = mfccs.shape[1]
                if length > max_len:
                    maxs.append(length)
                if length > 44:
                    print("Length longer then 44 !!!! SKIPPING")
                    print(full_path, name, mfccs.shape)
                    continue
                    
                files.append(full_path)
                labels.append(name)

                max_len = length if length > max_len else max_len
                mfccs = np.transpose(mfccs)
                features.append(mfccs)
                i=i+1
                # print(full_path, "label", name, str(i)+" files done.")

    for idx, mfcc in enumerate(features):
        mfccs = np.lib.pad(mfccs,((0,max_len-mfccs.shape[0]),(0,0)), mode = 'constant', constant_values=0) # ( max_len, n_mfccs )
        features[idx] = mfccs
    print(group, "MAXS =", maxs)
    return np.array(features), np.asarray(labels)

def one_hot_encode(labels):
    labels = categorical(labels, drop=True)
    return labels


val_features, val_labels = extract_features(root=dataset_path, group="validation")
val_labels = one_hot_encode(val_labels)
filelabel = 'features_rnn2'
np.save('valid_'+filelabel, val_features)
print('VALID rnn features saved: ',val_features.shape)
np.save('valid_labels_rnn', val_labels)
print('VALID labels saved: ', val_labels.shape)

ts_features, ts_labels = extract_features(root=dataset_path, group="test")
ts_labels = one_hot_encode(ts_labels)

np.save('test_'+filelabel, ts_features)
print('TEST rnn features saved: ', ts_features.shape)
np.save('test_labels_rnn', ts_labels)
print('TEST rnn labels saved: ', ts_labels.shape)

tr_features, tr_labels = extract_features(root=dataset_path, group="train")
tr_labels = one_hot_encode(tr_labels)

np.save('train_'+filelabel, tr_features)
print('TRAIN rnn features saved: ',tr_features.shape)
np.save('train_labels_rnn', tr_labels)
print('TRAIN labels saved: ', tr_labels.shape)

