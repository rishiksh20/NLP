# For Problem 5,
#
# We will implement a naive Bayesian sentiment classifier which learns to classify movie reviews as positive or negative.

import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import os
import numpy as np
from nltk.corpus import stopwords
stop = stopwords.words()
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
import re

class Classifier(object):

    def __init__(self):
        self.vectorizer = CountVectorizer(tokenizer=word_tokenize, ngram_range=(2,2), analyzer='word', lowercase=True)
        self.model = MultinomialNB()

    def clean(self, data):
        wordnet_lem = WordNetLemmatizer()
        for _ in range(len(data)):
            # data[_] = ' '.join([w for w in data[_].split() if w not in (stop)]) # stop words ## DOUBT : Removing Stop Words is decreasing the accuracy.
            data[_] = re.sub("http : //www . [\w.-]+ . com\/", '', data[_]) # removing URL links
            data[_] = re.sub("[\W_]+", ' ', data[_]) # remove special symbols
            data[_] = re.sub('\n', '', data[_]) # new line
            l = []
            for w in data[_].split():
                l.append(wordnet_lem.lemmatize(w)) # Lemmatization
            data[_] = ' '.join([w for w in l])
        
        return data

    def preprocess(self, data):
        data = self.clean(data)
        return self.vectorizer.fit_transform(data)

    def predict(self, data):
        data = self.clean(data)
        data = self.vectorizer.transform(data)
        return self.model.predict(data)

def load_data(path):
    corpus = []
    labels = []
    for label in os.listdir(path):
        if label in ['pos', 'neg']:
            for file in os.listdir(os.path.join(path, label)):
                with open(os.path.join(path, label, file), encoding='latin-1') as f:
                    corpus.append(f.read())
                    labels.append(1 if label == 'pos' else 0)

    return corpus, labels

def train(training_path):
    trained_model = Classifier()
    data, labels = load_data(training_path)
    data = trained_model.preprocess(data)
    trained_model.model.fit(data, labels)
    return trained_model


def predict(trained_model, testing_path):
    data, ground_truth = load_data(testing_path)
    model_predictions = trained_model.predict(data)
    return model_predictions, np.array(ground_truth)


def evaluate(model_predictions, ground_truth):
    accuracy = metrics.accuracy_score(ground_truth, model_predictions)
    tn, fp, fn, tp = metrics.confusion_matrix(ground_truth, model_predictions).ravel()
    print('_____________________________________________________________')
    print('|                 | Predicted Negative | Predicted Positive |')
    print('| Actual Negative |      {:03d}           |       {:03d}          |'.format(tn, fp))
    print('| Actual Positive |      {:03d}           |       {:03d}          |'.format(fn, tp))
    print('‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\n')
    print('Precision = ', round(tp / (tp + fp),4))
    print('Recall = ', round(tp / (tp + fn), 4))
    print('F-1 Score = ', round(metrics.f1_score(ground_truth, model_predictions), 4), '\n')

    return round(accuracy*100,4)