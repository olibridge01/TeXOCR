[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C.svg?logo=pytorch&logoColor=white)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-009485.svg?logo=fastapi&logoColor=white)](#)
[![Docker](https://img.shields.io/badge/Docker-2496ED.svg?logo=docker&logoColor=white)](#)


<p align="center">
  <img src="https://github.com/user-attachments/assets/e4435ca7-6d5f-4c3c-8a6d-0bef4adfb4f7" width="65%"/>
</p>

## Image to $\LaTeX$: Optical Character Recognition with PyTorch


This repository contains an OCR (Optical Character Recognition) model for recognizing LaTeX code from images. This repository allows for custom dataset generation, training, and evaluation of the model. The implementation is written with PyTorch.

I have also written a web application to get fast predictions of LaTeX code from images. The app is built with a FastAPI backend to serve the model.

<p align="center">
  <img src="https://github.com/user-attachments/assets/61670de1-ecca-4bd2-9092-b0d40eca4ccf" width="65%"/>
  <img src="https://github.com/user-attachments/assets/3d2b1703-d343-4fb6-ad5f-e495fd17dd43" width="65%"/>
</p>

### Model Overview
The model consists of an encoder-decoder architecture that is common for many current OCR systems. *TeXOCR* is based on the TrOCR model [[1]](#ref1) which utilises a Vision Transformer (ViT) [[2]](#ref2) encoder and a Transformer [[3]](#ref3) decoder. The model architecture is depicted in the figure below:

![TeXOCR_model](https://github.com/user-attachments/assets/de4a23d6-bed2-453f-9743-1b2b647ecbfd)


The vision encoder receives images of LaTeX equations and processes them into a series of embeddings $\mathbf{z}^{(i)} \in \mathbb{R}^{d}$ for each of the $N$ patches. The embeddings are passed into a Transformer decoder along with sequences of tokenized LaTeX code. The decoder generates a probability distribution over the vocabulary of LaTeX tokens to sample the next token in the sequence. The solution is then generated in an autoregressive manner to yield an overall prediction.

<!-- ### Running the Code
- Clone the repository
- Install the required packages by running `pip install -r requirements.txt`.
- Dataset preparation
- Training the model
- Testing the model
- Docker container? -->

### Installation
To clone the repository, run the following:
  ```bash
  git clone https://github.com/olibridge01/TeXOCR.git
  cd TeXOCR
  ```
For package management, set up a conda environment and install the required packages as follows:
  ```bash
  conda create -n texocr python=3.11 anaconda
  conda activate texocr
  pip install -r requirements.txt
  ```

For dataset rendering, `latex`, `dvipng`, and `imagemagick` are required. To install these dependencies, follow the instructions in the [`data_wrangling/`](data_wrangling/) directory.


### Data

The data used in this project is taken from the [Im2LaTeX-230k](https://www.kaggle.com/datasets/gregoryeritsyan/im2latex-230k) dataset (equations only). For use with a model consisting of a ViT encoder, I created custom scripts to generate the full dataset of image-label pairs, where each image has its dimensions altered to the nearest multiple of the patch size. To generate the dataset, simply execute:
  
  ```bash
  ./generate_dataset.sh
  ```
This takes the original equation data `data/master_labels.txt`, creates the data splits with `split_data.py` and renders the images with `render_images.py` (located in the `data_wrangling` directory). The rendered images are stored in `data/train`, `data/val`, and `data/test` directories. To create the dataset pickle files used in the training/testing scripts, run:
  
  ```bash
  ./generate_pickles.sh
  ```

### Tokenizer
This repository contains an implementation of the Byte Pair Encoding (BPE) [4] algorithm for tokenizing LaTeX code. To train the tokenizer on the Im2LaTeX-230k equation data, run:

  ```bash
  ./train_tokenizer.sh
  ```

To train the tokenizer on any text data, you can play around with the `tokenizer/tokenizer.py` script:

```bash
python tokenizer/tokenizer.py -v [vocab_size] -t -d [data_path] -s [save_path] --special [special_tokens] --verbose
```
where `vocab_size` is the desired vocabulary size, `data_path` is the path to the training data, `save_path` is the path to save the tokenizer (.txt file), and `special_tokens` is the path to a .txt file containing special tokens (e.g. [BOS], [PAD], etc.). Additionally, one can tinker with the `RegExTokenizer` class in Python as follows:

```python
from TeXOCR.tokenizer import RegExTokenizer

tokenizer = RegExTokenizer()
text = open('path/to/train.txt').read()
tokenizer.train(text)
tokenizer.save('path/to/tokenizer.txt')

# Tokenize a LaTeX string
tokens = tokenizer.encode('\int _ { 0 } ^ { 1 } x ^ 2 d x')
print(tokens)
```
where `train.txt` is some file containing tokenization training data. The tokenizer can be saved and loaded using the `save()` and `load()` methods.

### References 
[1]<a id="ref1"></a> [Li *et al*. - TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models (2021)](https://arxiv.org/abs/2109.10282)

[2] <a id="ref2"></a> [Dosovitskiy *et al*. - An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (2020)](https://arxiv.org/abs/2010.11929) 

[3] <a id="ref3"></a> [Vaswani *et al*. - Attention is All You Need (2017)](https://arxiv.org/abs/1706.03762)

