import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import pickle


def load_data(file_name):
    assert os.path.isfile(str(file_name)), str(file_name)+" file does not exist"
    file_path = str(file_name)
    text = []
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            text.append(line.split("\n/',")) 
    text_split = []
    for l in text:
        text_split.append(l[0].replace('\n','').split(' '))
    text_clean = []
    for _ in text_split:
        text_clean.append([x for x in _ if x!=''])
    return text_clean

def basic_cleaning_separating(data):
    TEXT = []
    TAGS = []
    for line in data:
        TEXT.append([line[x].lower() for x in range(len(line)) if x%2==0])
        TAGS.append([line[x] for x in range(len(line)) if x%2!=0])
    return TEXT, TAGS

def get_vocabulary(corpus):
    vocabulary = []
    v_size = -1
    for d in corpus:
        for w in d:
            if w not in vocabulary:
                vocabulary.append(w)
    v_size = len(vocabulary)
    return sorted(vocabulary), v_size

def treat_unka_train(TEXT_VOCAB, TEXT):
    TEXT_VOCAB_OCC = dict.fromkeys(TEXT_VOCAB, 0)
    for w in [word for sent in TEXT for word in sent]:
        TEXT_VOCAB_OCC[w] += 1
    TEXT_VOCAB_UNKA = list(dict((k, v) for k, v in TEXT_VOCAB_OCC.items() if v < 3).keys())
    TEXT_W_UNKA = []
    for sent in TEXT:
        sent_w_unka = []
        for word in sent:
            if word in TEXT_VOCAB_UNKA:
                sent_w_unka.append('UNKA')
            else:
                sent_w_unka.append(word)
        TEXT_W_UNKA.append(sent_w_unka)
    return TEXT_W_UNKA, TEXT_VOCAB_UNKA

def word_to_ind(vocabulary):
    word2ind = {}
    for _ in vocabulary:
        word2ind[_] = vocabulary.index(_)
    return word2ind

def tokenization(text, text_w2i):
    TEXT_EMB = []
    for sent in text:
        word_ind_emb = []
        for word in sent:
            word_ind_emb.append(text_w2i[word])
        TEXT_EMB.append(word_ind_emb)
    return TEXT_EMB

def padding(text, max_len = 250):
    lengths = []
    for sent in text:
        lengths.append(len(sent))
        l = [0]*(max_len - len(sent))
        sent.extend(l)
    return lengths

import gensim.downloader as api
def load_embedding_model(model):
    wv_from_bin = api.load(model)
    return wv_from_bin

def get_glove_emb(wv_from_bin, vocab, w2i):
    EMBEDDING_SIZE = 200
    VOCABULARY_SIZE = len(vocab)
    embedding_weights = np.zeros((VOCABULARY_SIZE, EMBEDDING_SIZE))
    word2id = w2i
    for word, index in word2id.items():
        try:
            embedding_weights[index, :] = wv_from_bin.get_vector(word)
        except KeyError:
            pass
    return embedding_weights



class RNNTagger(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNTagger, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        tag_space = self.fc(lstm_out)
        return tag_space


torch.manual_seed(595)
np.random.seed(595)

batch_size = 64
learning_rate = 0.001
num_epochs = 5
embedding_dim = 200
hidden_dim = 128
max_len = 250 # padding

def train(training_file):

    assert os.path.isfile(training_file), "Training file does not exist"

    print('Loading GloVe embeddings.')
    glove = "glove-wiki-gigaword-200"
    wv_from_bin = load_embedding_model(glove)
    print('Done.')

    print('Loading and Preprocessing Data. (Estimated time - around 2 mins.)')

    train = load_data(training_file)
    train_text_initial, train_tags = basic_cleaning_separating(train)
    train_text_initial_vocab, train_text_initial_vocab_size = get_vocabulary(train_text_initial)
    train_text, train_text_unka_vocab = treat_unka_train(train_text_initial_vocab, train_text_initial)
    train_text_vocab, train_text_vocab_size = get_vocabulary(train_text)
    train_tags_vocab, train_tags_vocab_size = get_vocabulary(train_tags)
    train_text_w2i = word_to_ind(train_text_vocab)
    train_tags_w2i = word_to_ind(train_tags_vocab)
    pickle.dump(train_text_w2i, open('train_text_w2i.p', 'wb'))
    pickle.dump(train_tags_w2i, open('train_tags_w2i.p', 'wb'))
    train_text_tokenized = tokenization(train_text, train_text_w2i)
    train_tags_tokenized = tokenization(train_tags, train_tags_w2i)
    train_text_lengths = padding(train_text_tokenized)
    train_tags_lengths = padding(train_tags_tokenized)

    train_text_emb = get_glove_emb(wv_from_bin, train_text_vocab, train_text_w2i)

    print('Done.\nBeginning the Training.')

    X = train_text_tokenized
    y = train_tags_tokenized

    vocab_size = train_text_vocab_size
    output_dim = train_tags_vocab_size
    
    train_data = TensorDataset(torch.LongTensor(X), torch.LongTensor(y))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    model = RNNTagger(vocab_size, embedding_dim, hidden_dim, output_dim)
    model.embedding.weight.data.copy_(torch.from_numpy(train_text_emb))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    from tqdm import tqdm
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        for inputs, labels in tqdm(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.view(-1, output_dim)
            labels = labels.view(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            for _ in range(max_len):
                if labels[_]!=0:
                    total_samples += 1
                    if predicted[_] == labels[_]:
                        total_correct += 1 
        epoch_loss = total_loss / len(train_loader)
        epoch_accuracy = total_correct / total_samples * 100

        print(f'Epoch [{epoch+1}/{num_epochs}] Train Accuracy: {epoch_accuracy:.2f}%')

    return model



def test(model_file, data_file, label_file):
    assert os.path.isfile(model_file), "Model file does not exist"
    assert os.path.isfile(data_file), "Data file does not exist"
    assert os.path.isfile(label_file), "Label file does not exist"

    train_text_w2i = pickle.load(open("train_text_w2i.p", "rb"))
    train_tags_w2i = pickle.load(open("train_tags_w2i.p", "rb"))
    vocab_size = len(train_text_w2i)
    output_dim = len(train_tags_w2i)

    model = RNNTagger(vocab_size, embedding_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(model_file))

    test_text_initial = load_data(data_file)
    test_text=[]
    for s in test_text_initial:
        test_text.append([x.lower() if x!='UNKA' else x for x in s])
    test_text_tokenized = tokenization(test_text, train_text_w2i)

    truth = load_data(label_file)
    _, truth_tags = basic_cleaning_separating(truth)
    truth_tags_tokenized = tokenization(truth_tags, train_tags_w2i)

    test_text_lengths = padding(test_text_tokenized)
    test_tags_lengths = padding(truth_tags_tokenized)

    test_truth_data = TensorDataset(torch.LongTensor(test_text_tokenized), torch.LongTensor(truth_tags_tokenized))
    test_truth_loader = DataLoader(test_truth_data, batch_size=batch_size)

    model.eval()
    pred = []
    with torch.no_grad():
        for inputs, labels in test_truth_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 2)
            pred.extend(predicted)

    # CUSTOM PROCEDURE FOR ACCURACY 
    ## for each row in pred (predictions, n_obs x pad_len) and true values, 
    ## compare the first n values (which were present before the padding - number of original values are stored in a separate list - o/p of padding func)
    ## Accuracy = correct predictions / total number of samples
    correct=0
    incorrect=0
    total=0
    for ind in range(len(truth_tags_tokenized)):
        total += test_tags_lengths[ind]
        for _ in range(test_tags_lengths[ind]):
            if pred[ind].tolist()[_] == truth_tags_tokenized[ind][_]:
                correct+=1
            else:
                incorrect+=1

    print('Accuracy of this model is ', round(correct/total*100, 2), '%')


def main(params):
    if params.train:
        model = train(params.training_file)
        torch.save(model.state_dict(), params.model_file)
    else:
        test(params.model_file, params.data_file, params.label_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HMM POS Tagger")
    parser.add_argument("--train", action="store_const", const=True, default=False)
    parser.add_argument("--model_file", type=str, default="model.torch")
    parser.add_argument("--training_file", type=str, default="")
    parser.add_argument("--data_file", type=str, default="")
    parser.add_argument("--label_file", type=str, default="")

    main(parser.parse_args())
