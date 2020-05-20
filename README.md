# Speech Commands Dataset

This is a basic speech recognition example using recurrent neural networks. For more information, see the
tutorial at https://www.tensorflow.org/tutorials/sequences/audio_recognition.

# Spiking model LSNN

This model implements the recurrent Long short-term Spiking Neural Network (LSNN) and reproduces the Google Speech Commands results from the paper:

> Salaj, D., Subramoney, A., Kraisnikovic, C., Bellec, G., Legenstein, R. and Maass, W., 2020.  
> [*Spike-frequency adaptation provides a long short-term memory to networks of spiking neurons*. bioRxiv](https://www.biorxiv.org/content/10.1101/2020.05.11.081513v1.abstract).

To reproduce result from the paper (91.2% test accuracy) run the following commands:

    python3 train.py --model_architecture=lsnn --window_stride_ms=1

The details that allow spiking network to achieve the high accuracy are:

- Spiking network is able to exploit the higher temporal resolution of the input so we use `--window_stride_ms=1`
- For classification we consider the output of spiking network throught the sequence `--avg_spikes=True`
- We use larger number of neurons `--n_hidden=2048`

Resulting accuracy:

| Iteration     | Validation    | Test  |
| ------------- |:-------------:| -----:|
| 400           | 68.6%         |       |
| 1200          | 79.4%         |       |
| 2400          | 85.3%         |       |
| 4800          | 87.5%         |       |
| 18000         | 91.5%         | 91.2% |

# LSTM model

run with:

    python3 train.py --model_architecture=lstm --n_hidden=512 --n_layer=1 --dropout_prob=0.4 --optimizer=adam

Resulting accuracy:

| Iteration     | Validation    | Test  |
| ------------- |:-------------:| -----:|
| 400           | 81.4%         |       |
| 1200          | 90.9%         |       |
| 18000         | 94.6%         | 94.4% |

# Default CNN model

CNN model is the default one used in TensorFlow GSC example, which is based on
`cnn-trad-fpool3` in the ['Convolutional Neural Networks for Small-footprint Keyword Spotting'](http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf) paper.

run with:

    python3 train.py --model_architecture=conv

Resulting accuracy:

| Iteration     | Validation    | Test  |
| ------------- |:-------------:| -----:|
| 18000         | 88.4%         | 87.6% |

# Environment

Tested with TensorFlow `2.0` and `2.1`.

To get started create the conda environment from file and activate:

    conda env create -f environment.yml
    conda activate venv2.1
    python3 train.py --model_architecture=lsnn --window_stride_ms=1
