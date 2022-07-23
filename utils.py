from typing import List, Tuple
from argparse import ArgumentParser
from abc import abstractmethod
import os
import pickle
import numpy as np
import pandas as pd
from torch import Tensor, batch_norm
import torch
import torch.nn as nn
from math import ceil
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
from seaborn import heatmap

import gensim.downloader

from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from transformers import get_linear_schedule_with_warmup

from preprocess import TweetExample
from feature_extractor import Vocab

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)


class LSTMModel(nn.Module):
    '''A LSTM model that outputs the normalized probability distribution'''
    def __init__(self, input_dim: int, embedding_model: Tensor, hidden_dim: int = 64, num_class: int = 3) -> None:
        super().__init__()

        self.embedding = nn.Embedding.from_pretrained(embedding_model, freeze=False)
        embedding_dim = embedding_model.size(1)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Linear(hidden_dim, num_class)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        
        embeds = self.embedding(x)

        lstm_out, (h_n, c_n) = self.lstm(embeds)
        out = self.fc1(h_n.view(x.shape[0], -1))
        #out = self.dropout(out)
        out = self.fc2(out)

        return self.logsoftmax(out)


class BiLSTMModel(nn.Module):
    '''A bidirectional LSTM model that outputs the normalized probability distribution'''
    def __init__(self, input_dim: int, embedding_model: Tensor, hidden_dim: int = 64, num_class: int = 3) -> None:
        super().__init__()

        self.embedding = nn.Embedding.from_pretrained(embedding_model, freeze=False)
        embedding_dim = embedding_model.size(1)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(4*hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_class)
        )
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        
        embeds = self.embedding(x)

        lstm_out, (h_n, c_n) = self.lstm(embeds)
        avg_pool = torch.mean(lstm_out, 1)
        max_pool, _ = torch.max(lstm_out, 1)
        out = torch.cat((avg_pool, max_pool), 1)
        out = self.fc(out)

        return self.logsoftmax(out)

class HateSpeechClassifier(object):
    @abstractmethod
    def fit(self, train_X: np.array, train_y: np.array, val_X: np.array, val_y: np.array) -> None:
        raise NotImplementedError("Calling abstract method!")

    @abstractmethod
    def predict(self, feats: np.array) -> np.array:
        raise NotImplementedError("Calling abstract method!")


class LSTMClassifier(HateSpeechClassifier):
    """
    A classifier using LSTM model

    model: 'lstm' or 'bilstm'
    sampling: 'uniform', 'oversampling' or 'focal_loss'
    """

    def __init__(
                self,
                vocab: Vocab,
                num_class: int = 3,
                hidden_dim: int = 64,
                batch_size: int = 64,
                learning_rate: float = 1e-3,
                epochs: int = 20,
                optimizer: str = 'adam',
                scheduler: List[int] = [5,10,15],
                scheduler_gamma: int = 0.1,
                model: str ='lstm',
                sampling: str = 'uniform',
                gpu: str = 'false') -> None:

        print("Initializing classifier...")

        self.vocab = vocab
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.sampling = sampling
        self.device = torch.device('cuda' if (gpu.lower()=='true') else 'cpu')
        

        # prepare the pretrained GloVe Model
        embedding_matrix = None
        vocab_size = self.vocab.get_vocab_size()
        if os.path.exists("cache/glove-twitter-200.pkl"):
            with open("cache/glove-twitter-200.pkl", "rb") as f:
                embedding_matrix = pickle.load(f)
        else:
            glove_vectors = gensim.downloader.load('glove-twitter-200')
            embedding_matrix = np.zeros((vocab_size, 200))  # 200 is the embedding dimension
            for i, word in enumerate(self.vocab.get_dict()):
                if glove_vectors.has_index_for(word):
                    embedding_matrix[i] = glove_vectors[word]
            with open("cache/glove-twitter-200.pkl", "wb") as f:
                pickle.dump(embedding_matrix, f)

        embedding_matrix = torch.from_numpy(embedding_matrix).to(self.device)

        print("Done!")
        print("Initializing model")
        # initialize a model
        if model == 'lstm':
            self.model = LSTMModel(vocab_size, embedding_matrix, self.hidden_dim, self.num_class).float()
            self.model.to(self.device)
        elif model == 'bilstm':
            self.model = BiLSTMModel(vocab_size, embedding_matrix, self.hidden_dim, self.num_class).float()
            self.model.to(self.device)

        # initialize an optiimizer and a scheduler
        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=scheduler, gamma=scheduler_gamma)

        # define the loss function
        if self.sampling == 'focal_loss':
            self.criterion = None
        elif self.sampling == 'oversampling' or self.sampling == 'uniform':
            self.criterion = nn.NLLLoss()
        else:
            raise NotImplementedError("Required sampling method not supported.")
        
        print("Done!")

    # fit the model
    def fit(self, train_X: np.array, train_y: np.array, val_X: np.array, val_y: np.array) -> None:
        
        device = self.device

        torch.autograd.set_detect_anomaly(True)

        if self.sampling == 'focal_loss':   # find and assign class weight
            classes = np.unique(train_y).astype(int)
            class_weight = np.zeros(len(classes))
            for _class in classes:
                class_weight[_class] = np.sum(train_y == _class)
            class_weight = np.sum(class_weight) / class_weight  # inverse weight
            class_weight = class_weight / np.sum(class_weight)  # normalize
            class_weight = torch.from_numpy(class_weight).float().to(device)

            self.criterion = nn.NLLLoss(class_weight)

        elif self.sampling == 'oversampling': # we have prior knowledge that class 0 is the most scarce label.
            class_0_id = (train_y == 0)
            class_2_id = (train_y == 2)
            iterations = ceil(np.sum(class_2_id) / np.sum(class_0_id))
            train_X_class_0 = train_X[class_0_id]
            train_y_class_0 = train_y[class_0_id]
            for i in range(iterations):
                train_X = np.concatenate((train_X, train_X_class_0), axis=0)
                train_y = np.concatenate((train_y, train_y_class_0), axis=0)
            
        num_samples = len(train_X)

        train_X = torch.from_numpy(train_X).long().to(device)
        train_y = torch.from_numpy(train_y).long().to(device)
        val_X = torch.from_numpy(val_X).long().to(device)
        val_y = torch.from_numpy(val_y).long()

        loss_values = []

        for epoch in range(self.epochs):
            running_loss = 0
            running_nsample = 0

            cur_id = torch.randperm(num_samples).to(device)
            cur_train_X = train_X[cur_id]
            cur_train_y = train_y[cur_id]

            # training
            for step in tqdm(range(num_samples // self.batch_size + 1), desc=f"Epoch {epoch}", leave=False):
                left = step * self.batch_size
                right = min((step+1)*self.batch_size, num_samples)
                batch_X = cur_train_X[left:right]
                batch_y = cur_train_y[left:right]

                self.model.train()

                pred = self.model(batch_X)
                loss = self.criterion(pred, batch_y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * (right-left)
                running_nsample += right - left
            
            epoch_loss = running_loss / running_nsample
            loss_values.append(epoch_loss)

            # validation 
            self.model.eval()
            preds = self.model(val_X).detach()
            # calculate validation loss
            loss = self.criterion(preds, val_y.to(device))
            # calculate validation accuracy
            if(preds.is_cuda):
                preds = preds.to('cpu')
            preds = preds.data.numpy()
            preds = np.argmax(preds, axis=1)

            accuracy = accuracy_score(val_y.data.numpy(), preds)
            fscore = f1_score(val_y.data.numpy(), preds, average='macro')
            print(f"Epoch: {epoch:02d}\tTrain Loss: {epoch_loss:.4f}\tValid Loss: {loss.item():.4f}\tValid Accuracy: {accuracy:.4f}\tValid F Score: {fscore: .4f}")
            self.scheduler.step()
            print("GPU allocated:", torch.cuda.memory_allocated('cuda'))
            print("GPU reserved:", torch.cuda.memory_reserved('cuda'))
            torch.save(self.model.state_dict(),f'model_log/{epoch}.pth')

    def predict(self, feats: np.array) -> np.array:

        feats = torch.from_numpy(feats).long().to(self.device)

        self.model.eval()
        preds = self.model(feats).detach()
        if preds.is_cuda:
            preds = preds.to('cpu')
        preds = preds.data.numpy()
        preds = np.argmax(preds, axis=1)

        return preds


class TransformerClassifier(HateSpeechClassifier):
    """
    A classifier using pretrained transformer model
    """

    def __init__(
                self,
                num_class: int = 3,
                batch_size: int = 64,
                learning_rate: float = 1e-3,
                epochs: int = 10,
                optimizer: str = 'adam',
                scheduler: List[int] = [5,10,15],
                scheduler_gamma: int = 0.1,
                model: str = 'bert',
                sampling: str = 'uniform',
                gpu: str = 'false',
                load_epoch: int = None) -> None:
        print("Initializing classifier...")

        self.num_class = num_class
        self.batch_size = batch_size
        self.epochs = epochs
        self.sampling = sampling
        self.device = torch.device('cuda' if (gpu.lower()=='true') else 'cpu')

        if model == 'bert':
            if load_epoch is not None:
                self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-cased")
                self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-cased', num_labels=num_class)
                if num_class == 2:
                    self.model.load_state_dict(torch.load(f"model_log/transformers/binary/{load_epoch}.pth"))
                else:
                    self.model.load_state_dict(torch.load(f"model_log/transformers/{load_epoch}.pth"))
            else:
                self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-cased")
                self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-cased', num_labels=num_class)
                self.model.to(self.device)
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=scheduler, gamma=scheduler_gamma)
                #num_training_steps = 1408 * self.epochs
                #self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)


        # define the loss function
        if self.sampling == 'focal_loss':
            self.criterion = None
        elif self.sampling == 'oversampling' or self.sampling == 'uniform':
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Required sampling method not supported.")        
        
        #count_parameters(self.model)
        
        print("Done!")

    def train(self, train_X: dict, train_y: np.array, val_X: dict, val_y: np.array) -> None:
        '''Only takes the first 128 samples to train the downstream task'''
        device = self.device
        self.model.to(device)
        num_samples = len(train_y)
        val_num_samples = 128

        if self.sampling == 'focal_loss':   # find and assign class weight
            classes = np.unique(train_y).astype(int)
            class_weight = np.zeros(len(classes))
            for _class in classes:
                class_weight[_class] = np.sum(train_y == _class)
            class_weight = np.sum(class_weight) / class_weight  # inverse weight
            class_weight = class_weight / np.sum(class_weight)  # normalize
            class_weight = torch.from_numpy(class_weight).float().to(device)

            self.criterion = nn.CrossEntropyLoss(weight=class_weight)

        elif self.sampling == 'oversampling': # we have prior knowledge that class 0 is the most scarce label.
            class_0_id = (train_y == 0)
            class_2_id = (train_y == 2)
            iterations = ceil(np.sum(class_2_id) / np.sum(class_0_id))
            train_X_class_0 = train_X[class_0_id]
            train_y_class_0 = train_y[class_0_id]
            for i in range(iterations):
                train_X = np.concatenate((train_X, train_X_class_0), axis=0)
                train_y = np.concatenate((train_y, train_y_class_0), axis=0)

        train_X.input_ids = train_X.input_ids[0:num_samples]
        train_X.attention_mask = train_X.attention_mask[0:num_samples]
        train_y = torch.from_numpy(train_y[0:num_samples]).long()

        val_X.input_ids = val_X.input_ids[0:val_num_samples]
        val_X.attention_mask = val_X.attention_mask[0:val_num_samples]
        val_y = torch.from_numpy(val_y[0:val_num_samples]).long()

        loss_values = []

        for epoch in range(self.epochs):
            running_loss = 0
            running_nsample = 64

            cur_id = torch.randperm(num_samples)
            cur_train_X_input_ids = train_X.input_ids[cur_id]
            cur_train_X_attention_mask = train_X.attention_mask[cur_id]
            cur_train_y = train_y[cur_id]

            # training
            for step in tqdm(range(num_samples // self.batch_size + 1), desc=f"Epoch {epoch}", leave=False):
                left = step * self.batch_size
                if left >= num_samples:
                    break
                right = min((step+1)*self.batch_size, num_samples)
                batch_X_input_ids = cur_train_X_input_ids[left:right]
                batch_X_attention_mask = cur_train_X_attention_mask[left:right]
                batch_y = cur_train_y[left:right]

                self.model.train()

                preds = self.model(batch_X_input_ids.to(device), batch_X_attention_mask.to(device))
                preds = preds.logits
                loss = self.criterion(preds, batch_y.to(device))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * (right - left)
                running_nsample += right - left
            
            epoch_loss = running_loss / running_nsample
            loss_values.append(epoch_loss)

            # validation
            self.model.eval()
            preds = self.model(val_X.input_ids.to(device), val_X.attention_mask.to(device)).logits.detach()
            loss = self.criterion(preds, val_y.to(device))
            preds = torch.softmax(preds, dim=-1)
            if(preds.is_cuda):
                preds = preds.to('cpu')
            preds = np.argmax(preds, axis=-1)

            accuracy = accuracy_score(val_y.data.numpy(), preds)
            fscore = f1_score(val_y.data.numpy(), preds, average='macro')
            print(f"Epoch: {epoch:02d}\tTrain Loss: {epoch_loss:.4f}\tValid Loss: {loss.item():.4f}\tValid Accuracy: {accuracy:.4f}\tValid F Score: {fscore: .4f}")
            self.scheduler.step()

            if self.num_class == 3:
                torch.save(self.model.state_dict(), f"model_log/transformers/{epoch}.pth")
            else:
                torch.save(self.model.state_dict(), f"model_log/transformers/binary/{epoch}.pth")

    
    def predict(self, tweets: dict):
        out = []
        device = self.device
        num_samples = len(tweets['input_ids'])

        tweets['input_ids'] = tweets['input_ids'].to(device)
        tweets['attention_mask'] = tweets['attention_mask'].to(device)


        self.model.eval()
        self.model.to(device)

        for step in range(ceil(num_samples / 64)):
            left = step * 64
            right = min((step+1)*64, num_samples)
            batch_X_input_ids = tweets['input_ids'][left:right]
            batch_X_attention_mask = tweets['attention_mask'][left:right]
            preds = self.model(batch_X_input_ids, batch_X_attention_mask).logits.detach()
            preds = torch.softmax(preds, dim=-1)
            if preds.is_cuda:
                preds = preds.to('cpu')
            preds = preds.data.numpy()
            preds = np.argmax(preds, axis=-1)
            out.extend(preds)

        out = np.array(out)
        return out


def draw_confusion_matrix(test_y: np.array, pred_y: np.array, num_class: int = 3, accuracy: bool = True) -> None:
    k = num_class
    confusionMatrix = confusion_matrix(test_y, pred_y)
    matrix_proportion = np.zeros((k,k))
    if accuracy:
        for i in range(0,k):
            matrix_proportion[i,:] = confusionMatrix[i,:] / float(confusionMatrix[i,:].sum())
    else:
        matrix_proportion = confusionMatrix
    if k == 3:
        names = ['Hate', 'Offensive', 'Neither']
    else:
        names = ['Hate', 'Offensive/Neither']
    confusion_df = pd.DataFrame(matrix_proportion, index=names, columns=names)
    plt.figure(figsize=(6,6))
    heatmap(confusion_df, annot=True, annot_kws={"size": 12}, cmap='gist_gray_r', cbar=False, square=True, fmt='.2f')
    plt.ylabel(r'True categories', fontsize=14)
    plt.xlabel(r'Predicted categories', fontsize=14)
    plt.tick_params(labelsize=12)
    plt.savefig('output/confusion.png')
    plt.show()
    
    
def predictFromModel(model: nn.Module, feats: np.array, gpu: str = 'false') -> np.array:
    device = torch.device('cuda' if (gpu.lower()=='true') else 'cpu')

    feats = torch.from_numpy(feats).long().to(device)

    model.eval()
    preds = model(feats).detach()
    if preds.is_cuda:
        preds = preds.to('cpu')
    preds = preds.data.numpy()
    preds = np.argmax(preds, axis=1)

    return preds


from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
    






