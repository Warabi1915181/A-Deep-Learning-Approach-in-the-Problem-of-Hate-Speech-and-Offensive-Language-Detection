# A Deep Learning Approach in the Problem of Hate Speech and Offensive Language Detection 
This is my penultimate undergraduate individual project as a demonstration of natural language processing (NLP) and deep learning (DL).

# NOTE: The codes will work fine, but they are a bit clunky at the moment (especially the syntax and poor object programming). I will update it when I have time.

## About the project
Offensive words may not be always deemed offensive. A casual conversation between closed friends may involve offensive words but they are deemed as inoffensive based on the context. Some rap lyrics may also contain offensive or even sexist words but they are generally not considered as hate speech. In some other occasions, some offensive words are indeed hate speech. An automated process of detecting hate speech requires the machine to consider the context of words. A robust algorithm should not flag a mere message with bad words immediately without considering the context behind the words. This problem was addressed and analyzed by other researchers using non-deep-learning approaches. Their paper could be found [here](https://arxiv.org/abs/1703.04009). Instead of the methodology adopted by the aforementioned researchers, I attempt to apply deep learning methods in this problem.

## Models
Three types of models can be applied:
1. Recurrent model (RNN) with one-direction LSTM (long short-term memory) module
2. RNN with bidirectional LSTM module
3. Pretrained distilled BERT (bidirectional encoder representations from transformers) model

## Files
#### split_data.py
  - randomly split a dataset into training, validation and testing datasets.
#### preprocess.py
  - if datasets are not preprocessed, preprocess the Tweets in the datasets such as removing url or mention in each Tweet, then store preprocessed data as a .pkl file in ./cache
  - if .pkl files exist, load the .pkl files directly to skip preprocessing
#### feature_extractor.py
  - a simple feature extractor that index each word in each Tweet for LSTM and bidirectional LSTM RNNs
  - distilled BERT model does not require this.
#### utils.py
  - core functions for defining the models and their behaviour
#### main.py
  - executable main file
  
## To Use
You can download the files for your own use. For the dataset I used, you can download the *labeled_data.csv* [here](https://github.com/t-davidson/hate-speech-and-offensive-language/tree/master/data).
