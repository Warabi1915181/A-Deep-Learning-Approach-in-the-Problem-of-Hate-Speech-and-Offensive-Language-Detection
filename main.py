from typing import List, Tuple
import argparse
from argparse import ArgumentParser
import os, pickle
from preprocess import TweetExample, load_examples, load_examples_transformer
from feature_extractor import Vocab, RawTextFeatureExtractor
from utils import LSTMModel, BiLSTMModel, LSTMClassifier, draw_confusion_matrix, predictFromModel, TransformerClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import pandas as pd
import torch

import timeit


NUM_EPOCHS = 10
OPTIMIZER = 'adam'
SCHEDULER = [5]
SCHEDULER_GAMMA = 0.1

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--train_file", type=str,
                        default="data/train_data.csv", help="path to training file")
    parser.add_argument("--val_file", type=str,
                        default="data/val_data.csv", help="path to validation file")
    parser.add_argument("--test_file", type=str,
                        default="data/test_data.csv", help="path to test file")
    parser.add_argument("--num_class", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model", type=str, default="lstm")
    parser.add_argument("--sampling", type=str, default="uniform")
    parser.add_argument("--gpu", type=str, default="false")
    parser.add_argument("--test", type=int, default=None)
    args = parser.parse_args()
    return args

def train(args: ArgumentParser, train_exs: List[TweetExample], val_exs: List[TweetExample], feat_extractor: RawTextFeatureExtractor) -> LSTMClassifier:

    # clear cuda cache
    if(args.gpu.lower() == 'true'):
        torch.cuda.empty_cache()

    print("Extracting features...")

    if os.path.exists("cache/train_X.pkl") and os.path.exists("cache/train_y.pkl"):
        with open("cache/train_X.pkl", "rb") as f:
            train_X = pickle.load(f)
        with open("cache/train_y.pkl", "rb") as f:
            train_y = pickle.load(f)
    else:
        train_X = feat_extractor.extract_features(train_exs)
        train_y = np.zeros(len(train_X))
        for i, ex in enumerate(train_exs):
            train_y[i] = ex.label
        with open("cache/train_X.pkl", "wb") as f:
            pickle.dump(train_X, f)
        with open("cache/train_y.pkl", "wb") as f:
            pickle.dump(train_y, f)

    if os.path.exists("cache/val_X.pkl") and os.path.exists("cache/val_y.pkl"):
        with open("cache/val_X.pkl", "rb") as f:
            val_X = pickle.load(f)
        with open("cache/val_y.pkl", "rb") as f:
            val_y = pickle.load(f)
    else:
        val_X = feat_extractor.extract_features(val_exs)
        val_y = np.zeros(len(val_X))
        for i, ex in enumerate(val_exs):
            val_y[i] = ex.label
        with open("cache/val_X.pkl", "wb") as f:
            pickle.dump(val_X, f)
        with open("cache/val_y.pkl", "wb") as f:
            pickle.dump(val_y, f)

    print("Done!")
    
    model = LSTMClassifier(
        feat_extractor.get_vocab(),
        args.num_class,
        args.hidden_dim,
        args.batch_size,
        args.lr,
        epochs=NUM_EPOCHS,
        optimizer=OPTIMIZER,
        scheduler=SCHEDULER,
        scheduler_gamma=SCHEDULER_GAMMA,
        model=args.model,
        sampling=args.sampling,
        gpu=args.gpu
    )

    model.fit(train_X, train_y, val_X, val_y)

    return model

def test(model, test_exs, feat_extractor, epoch: int = None) -> Tuple[np.array, np.array]:
    test_X = feat_extractor.extract_features(test_exs)
    test_y = np.zeros(len(test_X))
    for i, ex in enumerate(test_exs):
        test_y[i] = ex.label

    preds = model.predict(test_X)
    accuracy = accuracy_score(test_y, preds)
    precision, recall, fscore, _ = precision_recall_fscore_support(test_y, preds, average='macro')
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F-score: {fscore:.4f}")

    return test_y, preds
    


def main():
    args = parse_args()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    start = timeit.default_timer()
    if args.model.lower() == 'lstm' or args.model.lower() == 'bilstm':
        if args.test is not None:
            train_exs = load_examples(args.train_file, "train")
            #test_exs = load_examples(args.test_file, "test")
            test_exs = load_examples("data/labeled_data.csv", "all")
            featExtractor = RawTextFeatureExtractor(train_exs)


            if os.path.exists("cache/glove-twitter-200.pkl"):
                with open("cache/glove-twitter-200.pkl", "rb") as f:
                    embedding_matrix = pickle.load(f)
                    embedding_matrix = torch.from_numpy(embedding_matrix)
            else:
                raise OSError('embedding model not found')
            if args.model.lower() == 'lstm':
                model = LSTMModel(featExtractor.get_vocab().get_vocab_size(), embedding_matrix, args.hidden_dim, args.num_class).float()
            else:
                model = BiLSTMModel(featExtractor.get_vocab().get_vocab_size(), embedding_matrix, args.hidden_dim, args.num_class).float()

            path_name = f"model_log/{args.test}.pth"
            if os.path.exists(path_name):
                model.load_state_dict(torch.load(path_name))
            else:
                raise OSError('Save state not found')
            

            
            preds = predictFromModel(model, featExtractor.extract_features(test_exs))
            test_y = np.zeros(len(preds))
            for i, ex in enumerate(test_exs):
                test_y[i] = ex.label
            
            misclassified = []
            for i in range(len(preds)):
                if int(preds[i]) != int(test_y[i]):
                    misclassified.append(i)
            MAX_ITERATION = 10
            np.random.seed(42)
            print(test_exs[0:5])

            print("-------------------------")
            for i in range(MAX_ITERATION):
                j = np.random.randint(0, len(misclassified))
                print(misclassified[j])
                print("Ground Truth: ", test_y[misclassified[j]])
                print("predicted: ", preds[misclassified[j]])
                print(test_exs[misclassified[j]])
                

            accuracy = accuracy_score(test_y, preds)
            precision, recall, fscore, _ = precision_recall_fscore_support(test_y, preds, average='macro')
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F-score: {fscore:.4f}")
            draw_confusion_matrix(test_y, preds)

            return

        # load train data
        train_exs = load_examples(args.train_file, "train")

        # load validation data
        val_exs = load_examples(args.val_file, "valid")

        # load test data
        test_exs = load_examples(args.test_file, "test")

        # feature extraction
        featExtractor = RawTextFeatureExtractor(train_exs)




        model = train(args, train_exs, val_exs, featExtractor)

        print("======== Test accuracy =========")
        test_y, preds = test(model, test_exs, featExtractor)
        stop = timeit.default_timer()
        draw_confusion_matrix(test_y, preds)
        
    elif args.model.lower() == 'bert':

        classifier = TransformerClassifier(
            num_class=args.num_class,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            epochs=NUM_EPOCHS,
            optimizer=args.optimizer,
            scheduler=SCHEDULER,
            scheduler_gamma = SCHEDULER_GAMMA,
            model=args.model,
            sampling=args.sampling,
            gpu=args.gpu,
            load_epoch=args.test)

        if args.test is not None:
            #test_X, test_y = load_examples_transformer(args.test_file, "test", classifier.tokenizer)
            test_X, test_y = load_examples_transformer("data/labeled_data.csv", "all", classifier.tokenizer)
            if args.num_class == 2:
                temp = np.copy(test_y)
                test_y[test_y == 2] = 1
            preds = classifier.predict(test_X)
            accuracy = accuracy_score(test_y, preds)
            precision, recall, fscore, _ = precision_recall_fscore_support(test_y, preds, average='macro')
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F-score: {fscore:.4f}")

            if args.num_class == 2:
                ids = (preds == 0)
                false_positives = np.sum(temp[ids] == 1)
                print(f"Number of offensive language tweets but classified as hate speech: {false_positives}")
                

            stop = timeit.default_timer()
            draw_confusion_matrix(test_y, preds, args.num_class, accuracy=True)
            print("--------------------------------")
            print(f'Total run time: {stop - start} seconds')

            return



        train_X, train_y = load_examples_transformer(args.train_file, "train", classifier.tokenizer)
        val_X, val_y = load_examples_transformer(args.val_file, "valid", classifier.tokenizer)
        test_X, test_y = load_examples_transformer(args.test_file, "test", classifier.tokenizer)

        if args.num_class == 2:
            train_y[train_y == 2] = 1
            val_y[val_y == 2] = 1
            test_y[test_y == 2] = 1

        classifier.train(train_X, train_y, val_X, val_y)

        preds = classifier.predict(test_X)
        accuracy = accuracy_score(test_y, preds)
        precision, recall, fscore, _ = precision_recall_fscore_support(test_y, preds, average='macro')
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F-score: {fscore:.4f}")
        stop = timeit.default_timer()
        draw_confusion_matrix(test_y, preds, args.num_class)

    print("--------------------------------")
    print(f'Total run time: {stop - start} seconds')

if __name__ == "__main__":
    main()