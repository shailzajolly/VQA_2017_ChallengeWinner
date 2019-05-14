# VQA_2017_ChallengeWinner
Pytorch implementation for 2017 VQA Challenge Winner [Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering]. 

## Prerequisites
- python 3.6+
- numpy
- [pytorch](http://pytorch.org/) 0.4
- [tqdm](https://pypi.python.org/pypi/tqdm)
- [nltk](http://www.nltk.org/install.html)
- [pandas](https://pandas.pydata.org/)

## Data
- [VQA 2.0](http://visualqa.org/download.html)
- [COCO 36 features pretrained resnet model](https://github.com/peteanderson80/bottom-up-attention#pretrained-features)
- [GloVe pretrained Wikipedia+Gigaword word embedding](https://nlp.stanford.edu/projects/glove/)

## Steps to use this repository
- First, download the data using:
```bash
  bash scripts/download_extract.sh
  ```
This will download and extract annotation files of VQA V2, glove embeddings and pretrained image features.  

- Secondly, the following command is used to prepare data for training. 
python prepro.py

