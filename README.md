# Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering
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
  bash download_extract.sh
  ```
This will download and extract annotation files of VQA V2, glove embeddings and pretrained image features.  

- Secondly, the following command is used to prepare data for training. 
```bash
python prepro.py
  ```
-Once the data is ready for training, the training can be started using this command. 
```bash
bash train.sh
  ```
  
## Notes
- Training for 31 epochs reach around 62.4% validation accuracy.
- Some of `preproc.py` and `utils.py` are based on [this repo](https://github.com/markdtw/vqa-winner-cvprw-2017) 

## Resources
- [The paper](https://arxiv.org/pdf/1708.02711.pdf).
- [Their CVPR Workshop slides](http://cs.adelaide.edu.au/~Damien/Research/VQA-Challenge-Slides-TeneyAnderson.pdf).
