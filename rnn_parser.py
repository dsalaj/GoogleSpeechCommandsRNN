import os
import sys
import numpy as np
import librosa
from scipy.io import wavfile
from statsmodels.tools import categorical

nb_classes = 30

def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size / 2)

def extract_features(path):

    features = np.array([], dtype='float32')

    files = [file for file in os.listdir(path)]
    label = ["" for x in range(len(files))]
    np.random.shuffle(files)

    i=0
    count=0
    print(os.path.basename(path))
    for file_name in files:

        unit = os.path.join(path,(str)(file_name))
        sample_rate, sample = wavfile.read(unit)
        #print("Loaded: ",unit)

        name = file_name[:file_name.index('_')]
        label[i] = name
        i=i+1
        print((str)(i)+" files done.")

        mfccs = librosa.feature.mfcc(y=sample, sr=sample_rate, n_mfcc=26)
        mfccs = np.transpose(mfccs) # ( frames, 26 )
        mfccs = np.lib.pad(mfccs,((0,32-mfccs.shape[0]),(0,0)), mode = 'constant', constant_values=0) # ( 32, 26 )
        mfccs = mfccs[np.newaxis, :, :]

        if count==0:
            features = mfccs
            count = 1
        else:
            features = np.vstack((features, mfccs)) # ( number of files, frames=32, 26 )

    return np.array(features), np.asarray(label)

def one_hot_encode(labels):
    labels = categorical(labels, drop=True)
    return labels


tr_features,tr_labels = extract_features(path="C:\\Users\\Shivank\\Desktop\\RnnSpeech\\train")
tr_labels = one_hot_encode(tr_labels)

np.save('train_features_rnn', tr_features)
print('rnn features saved: ',tr_features.shape)
np.save('train_labels_rnn', tr_labels)
print('labels saved: ',tr_labels.shape)

ts_features,ts_labels = extract_features(path="C:\\Users\\Shivank\\Desktop\\RnnSpeech\\test")
ts_labels = one_hot_encode(ts_labels)

np.save('test_features_rnn', ts_features)
print('rnn features saved: ', ts_features.shape)
np.save('test_labels_rnn', ts_labels)
print('rnn labels saved: ', ts_labels.shape)
