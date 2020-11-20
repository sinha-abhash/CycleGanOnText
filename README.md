# CycleGANOnText in Tensorflow 

[Paper](https://arxiv.org/abs/1703.10593) | [PyTorch Code](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) | [Tensorflow Code](https://www.tensorflow.org/tutorials/generative/cyclegan)

This project is an attempt to try CycleGAN on text. The input of the model is a question with lots of irrelevant information around it and the output of the model is the question without any irrelevant information.

## Dataset
The dataset used in the project are:
- [News Dataset](https://www.kaggle.com/rmisra/news-category-dataset)
- [SQUAD](https://rajpurkar.github.io/SQuAD-explorer/)

The irrelevant dataset or domain A is created by concatenating the headline of the news, SQUAD question and description of the news.
```
<headline of news> + ' ' + <squad question> + ' ' + <description of news>
```

## Preprocessing
 - Each sentence in the question (domain A/B) are tokenized (nltk sentence tokenizer).
 - Universal Sentence Encoder is used to extract feature vectors of each sentence in the question.
 - For LSTM input, each sentence in the question is a timestamp (second dimension and the feature vector is the feature (third dimension).
 
 ## Architecture
 As per CycleGAN, there are two generators (G and F) and two discriminators (D_X and D_Y).
 - Generator is made of few layers of encoder/decoder style LSTM layers.
 - Discriminator uses Conv1D layers along with Dense to distinguish between fake and real.
 
 ## Loss
 Below losses are explained in the [Tensorflow](https://www.tensorflow.org/tutorials/generative/cyclegan) implementation
 - Adversial Loss
 - Cycle Loss
 - Identity Loss
