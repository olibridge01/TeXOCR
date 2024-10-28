
<p align="center">
  <img src="https://github.com/user-attachments/assets/e4435ca7-6d5f-4c3c-8a6d-0bef4adfb4f7" width="65%"/>
</p>

## Image to $\LaTeX$: Optical Character Recognition with PyTorch

This repository contains an OCR (Optical Character Recognition) system for recognizing LaTeX code from images. The system is implemented in PyTorch.

### Model Overview
The model consists of an encoder-decoder architecture that is common for many current OCR systems. *TeXOCR* is based on the TrOCR model [1] which utilises a Vision Transformer (ViT) [2] encoder and a Transformer [3] decoder.

$$\text{Model diagram}$$

### Running the Code
- Clone the repository
- Install the required packages by running `pip install -r requirements.txt`.
- Dataset preparation
- Training the model
- Testing the model
- Docker container?

### Data

Mention of the dataset used for training and testing the model. If I get round to it, mention how I generate my own dataset.

### Tokenizer
This repository contains an implementation of the Byte Pair Encoding (BPE) [4] algorithm for tokenizing LaTeX code. To train the tokenizer on some text data, run the following command:

```python
from tokenizer import RegExTokenizer

tokenizer = RegExTokenizer()
text = open('path/to/train.txt').read()
tokenizer.train(text)

# Tokenize a LaTeX string
tokens = tokenizer.encode('\int _ { 0 } ^ { 1 } x ^ 2 d x')
```
where `train.txt` is some file containing tokenization training data. The tokenizer can be saved and loaded using the `save()` and `load()` methods.

### References 
[1] [Li *et al*. - TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models (2021)](https://arxiv.org/abs/2109.10282)  
[2] [Dosovitskiy *et al*. - An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (2020)](https://arxiv.org/abs/2010.11929)  
[3] [Vaswani *et al*. - Attention is All You Need (2017)](https://arxiv.org/abs/1706.03762)

