# Speech Commands Example

This is a basic speech recognition example. For more information, see the
tutorial at https://www.tensorflow.org/tutorials/sequences/audio_recognition.

# Added LSTM model

run with:

    python3 train.py --model_architecture=lstm --n_hidden=512 --n_layer=1 --dropout_prob=0.4 --optimizer=adam

Resulting accuracy:

| Iteration     | Validation    | Test  |
| ------------- |:-------------:| -----:|
| 400           | 81.4%         |       |
| 1200          | 90.9%         |       |
| 18000         | 94.6%         | 94.4% |

# Default CNN model

run with:

    python3 train.py --model_architecture=conv

Resulting accuracy:

| Iteration     | Validation    | Test  |
| ------------- |:-------------:| -----:|
| 18000         | 88.4%         | 87.6% |

# Environment

tested on `tensorflow-gpu==2.0.0`