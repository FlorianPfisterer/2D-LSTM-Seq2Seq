# 2D-LSTM Seq2Seq Model
This repository contains a PyTorch implementation of a 2D-LSTM model for sequence-to-sequence learning.

In addition, it contains code to apply the 2D-LSTM to neural machine translation (NMT) based on the paper
["Towards two-dimensional sequence to sequence model in neural machine translation"](https://arxiv.org/abs/1810.03975)
by Parnia Bahar, Christopher Brix and Hermann Ney.

## Getting Started
### Prerequisites
Clone the project and make sure to install the dependencies listed in [`requirements.txt`](./requirements.txt).

If you use the included dataset helper functions for the small IWSLT14 deu-eng NMT dataset (taken from
[harvardnlp/var-attn/data](https://github.com/harvardnlp/var-attn/tree/master/data)), it will automatically 
preprocess the data into `.csv` files before the first run.

I've successfully run all tests using:
* [PyTorch](http://pytorch.org) `1.0.1.post2`
* [torchtext](https://github.com/pytorch/text) `0.3.1` 
* [NumPy](http://www.numpy.org) `1.16.2`
* [pandas](https://pandas.pydata.org) `0.24.1`
* [tensorboardX](https://github.com/lanpa/tensorboardX) `1.6`

### Running Scripts
With the dependencies installed, you can run the scripts in the [`main/`](./main) folder. 
To run the IWSLT14 training script for example, just run
```
python -m main.train_iwslt14_small
```
The available command line arguments for this script can be found below.

## The 2D-LSTM Seq2Seq Model
2D recurrent neural networks are widely used in many applications manipulating 2D objects such as like images.
For instance, 2D-LSTMs have become the state-of-the-art in Handwritten Text Recognition
[[1]](https://www.vision.rwth-aachen.de/media/papers/MDLSTM_final.pdf).

The method described in [[2]](https://arxiv.org/abs/1810.03975) is an approach to apply such 2D-LSTMs to 
sequence-to-sequence tasks such as neural machine translation.

### General Architecture
A source sentence is read by a standard (i.e. 1D) bidirectional LSTM encoder using end-to-end trained embedding vectors.
Its hidden states ![](https://timodenk.com/api/tex2img/h_0%2C%20%5Cdots%2C%20h_n?format=svg)
(concatenating both directions) are then used as the inputs in the horizontal dimension of the 2D-LSTM.

Vertically, the generated (embedded) tokens 
![](https://timodenk.com/api/tex2img/y_0%2C%20%5Cdots%2C%20y_m?format=svg) of the respective previous row 
are given to the 2D cell. In training mode, teacher forcing is used (i.e. the correct tokens are used).

The hidden state of the cell in the last column is then fed into a fully-connected softmax layer which forms
the prediction for the next output token.

The basic idea is that the 2D-LSTM re-reads the input sentence for each new output token, conditioned on the 
previously generated token.  

### The 2D-LSTM Cell
The 2D-LSTM cell at horizonal step `i` and vertical step `j` consumes encoder hidden state 
![](https://timodenk.com/api/tex2img/h_i?format=svg) concatenated to the (embedded) token
![](https://timodenk.com/api/tex2img/y_j?format=svg) as well as the hidden and cell states from the 
previous vertical and the previous horizontal step.

See [`lstm2d_cell.py`](./model/lstm2d_cell.py) or the [paper](https://arxiv.org/abs/1810.03975) for details.

### Training vs. Inference 
In inference mode, the target tokens ![](https://timodenk.com/api/tex2img/y_0%2C%20%5Cdots%2C%20y_m?format=svg)
are not known in advance. Thus, only the naive ![](https://timodenk.com/api/tex2img/%5Cmathcal%7BO%7D(mn)?format=svg)
implementation of going through each row after the other is feasible.

In training mode however, the target tokens _are_ known in advance (and we use teacher forcing).
Thus, we can traverse the 2D-grid in an 
efficient ![](https://timodenk.com/api/tex2img/%5Cmathcal%7BO%7D(m%2Bn)?format=svg) diagonal-wise fashion 
([[1]](https://www.vision.rwth-aachen.de/media/papers/MDLSTM_final.pdf), [[2]](https://arxiv.org/abs/1810.03975)).

To enable both training and inference yet make use of the possible parallelization in training mode,
the [2D-LSTM code](./model/lstm2d.py) contains two different implementations of the forward propagation:
one for training (`forward(x, x_lengths, y)`) and another one for inference (`predict(x, x_lengths)`).

## Running Training
The [train_iwslt14_small.py](./main/train_iwslt14_small.py) script contains code to train a 2D-LSTM model on 
the small IWSLT14 deu-eng NMT dataset
(taken from [harvardnlp/var-attn/data](https://github.com/harvardnlp/var-attn/tree/master/data)).

The following command line arguments are supported, with the given default values:
* `--batch_size=32`: The batch size to use for training and inference.
* `--epochs=20`: The number of epochs to train.
* `--shuffle=True`: Whether or not to shuffle the training examples.
* `--lr=0.0005`: The learning rate to use.
* `--embed_dim=128`: The dimension of the embedding vectors for both the source and target language.
* `--encoder_state_dim=64`: The dimension of the bidirectional encoder LSTM states.
* `--state_2d_dim=128`: The dimension of the 2D-LSTM hidden & cell states.
* `--disable_cuda=False`: Disable CUDA (i.e. use the CPU for all computations).
* `--dropout_p=0.2`: The dropout probability, used after the embeddings and before the final softmax layer.

## Tests
This repository contains test cases in the [`test/`](./test) folder that make sure the 2D-LSTM model 
does what it should do. After installing the additional dependencies listed in
[`test-requirements.txt`](./test-requirements.txt), you can run all of them using 
```bash
python -m unittest 
```

### 2D-LSTM Cell Tests
These tests make sure the input and output dimensions of a single 2D-LSTM cell are as expected and 
the outputs are the same for each example in the batch, if the same is true for the inputs.
They can be found in [`test_lstm2d_cell.py`](test/test_lstm2d_cell.py). 

### 2D-LSTM Model Tests
These tests have varied purposes:
* The tests in [`test_lstm2d_training.py`](test/test_lstm2d_training.py)
and [`test_lstm2d_inference.py`](test/test_lstm2d_inference.py) make sure the input and output dimensions are as 
expected in training and inference mode, respectively.
* The tests in [`test_lstm2d_train_vs_inference.py`](test/test_lstm2d_train_vs_inference.py) validate the training
and inference forward propagation code by comparing the predictions in both modes to each other when the same 
target tokens are used. This includes the handling of padding for batches that contain sequences of different lengths.
* The tests in [`test_lstm2d_fit.py`](test/test_lstm2d_fit.py) make sure the 2D-LSTM can fit a small synthetic 
dataset in a few iterations (to sanity-check it).

## Future Work
This model currently does not use any attention mechanism. Future research might try removing the 2D-recurrence
in favor of a Transformer-like self-attention mechanism [[3]](https://arxiv.org/abs/1706.03762). 

### Contributing
If you have ideas on how to improve or extend this code or you have spotted a problem, feel free to open a PR
or contact me (see below).

## Author
I'm Florian Pfisterer. [Email me](mailto:florian.pfisterer1@gmail.com) or reach out on
Twitter [@FlorianPfi](https://twitter.com/@FlorianPfi).

## License
This project is licensed under the MIT License - see [LICENSE.md](./LICENSE.md) for details.

## Acknowledgments
I would like to thank:
* [Ngoc Quan Pham](https://scholar.google.com/citations?hl=en&user=AzzJssIAAAAJ)
for his advice and support throughout this project. 
* [Parnia Bahar](https://scholar.google.com/citations?user=eyc24McAAAAJ&hl=en)
for her thorough response to my email questions about the details of [her paper](https://arxiv.org/abs/1810.03975).
* [Timo Denk](https://timodenk.com) for our inspiring paper discussions around the topic,
his ideas and last but not least his awesome
[TeX2Img API](https://tools.timodenk.com/tex-math-to-image-conversion) used in this README! 

## References
[1] Voigtlander et al., 2016, "Handwriting Recognition with Large Multidimensional Long Short-Term Memory
Recurrent Neural Networks", https://www.vision.rwth-aachen.de/media/papers/MDLSTM_final.pdf

[2] Bahar et al., 2018, "Towards Two-Dimensional Sequence to Sequence Model in Neural Machine Translation", 
https://arxiv.org/abs/1810.03975

[3] Vaswani et al., 2017, "Attention Is All You Need", https://arxiv.org/abs/1706.03762
