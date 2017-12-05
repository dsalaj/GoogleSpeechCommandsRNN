# RecurrentNN_SpeechRecognition
A model based in Tensorflow to recognize words from the 30 word Speech Commands Dataset from Google using 
LSTM based Recurrent Neural Network.

    rnn_parser.py = { simply walks through all the individual .wav audio files in the training and 
    test datasets and calculates MFCC values for each file and stores them in a 32x26 matrix. 
    32 is the number of frames in the longest file and the files which do not yield 32 frames 
    are padded with zeroes in the last few rows. The matrices are stacked on top of each other 
    two yield one huge 3D array which hold all data files' features. It also extracts labels. 
    In the end, the arrays are saved as numpy arrays for faster extraction duing training. }
  
    model_rnn.py = { the model contains one initial layer with ReLU activation followed by 
    two LSTM cells with 100 hidden units each, combined as a static RNN model. }

After 35,000 epochs,

TRAINING ACCURACY : 90 %
TEST ACCURACY : 86 %
