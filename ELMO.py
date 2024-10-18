import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import numpy as np
import re

# Set the device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DataProcessor class for preprocessing the dataset
class DataProcessor:
    def __init__(self, train_file_path, test_file_path, padding=False):
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.padding = padding
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
    
    def process_data(self, split='train', min_count=3):
        stop_words = set(stopwords.words('english'))
        punctuations = set(string.punctuation)
        ps = PorterStemmer()
        
        file_path = self.train_file_path if split == 'train' else self.test_file_path
        data = pd.read_csv(file_path)
        reviews = data['Description']
        labels = data['Class Index']
        
        word_count = {}
        cleaned_sentences = []
        cnt = 0
        
        for idx, desc in enumerate(reviews):
            curr_record = ["<SOS>"]
            for sentence in sent_tokenize(desc):
                sentence = re.sub(r'[\\-]', ' ', sentence)
                sentence = re.sub(r'[^a-zA-Z0-9 ]', '', sentence)
                sentence = word_tokenize(sentence)
                sentence = [word for word in sentence if word not in stop_words and word not in punctuations]
                for word in sentence:
                    word_count[word] = word_count.get(word, 0) + 1
                curr_record.extend(sentence)
            curr_record.append("<EOS>")
            cleaned_sentences.append(curr_record)
            cnt += 1
            if cnt == 20000:
                break
        
        if split == 'train':
            for word, count in word_count.items():
                if count >= min_count:
                    self.word2idx[word] = len(self.word2idx)
        
        for idx, sentence in enumerate(cleaned_sentences):
            cleaned_sentences[idx] = [word if word in self.word2idx else "<UNK>" for word in sentence]
        
        return cleaned_sentences if split == 'train' else cleaned_sentences, self.word2idx

# Custom dataset class for handling data
class CustomDataset(Dataset):
    def __init__(self, sentences, labels, word_to_index, pad=True, max_length=50):
        self.sentences = sentences
        self.labels = labels
        self.word_to_index = word_to_index
        self.pad = pad
        self.max_length = max_length

        self.indexed_sentences = []
        for sentence in self.sentences:
            indexed_sentence = [self.word_to_index.get(word, self.word_to_index["<UNK>"]) for word in sentence]
            self.indexed_sentences.append(torch.tensor(indexed_sentence))

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        indexed_sentence = self.indexed_sentences[idx]
        indexed_sentence_back = indexed_sentence.flip(0)

        if len(indexed_sentence) > self.max_length:
            indexed_sentence = indexed_sentence[:self.max_length]
            indexed_sentence_back = indexed_sentence_back[:self.max_length]

        if self.pad:
            pad_length = max(0, self.max_length - len(indexed_sentence))
            padded_sentence = F.pad(indexed_sentence, (0, pad_length))
            padded_sentence_back = F.pad(indexed_sentence_back, (0, pad_length))
        else:
            padded_sentence = indexed_sentence[:self.max_length]
            padded_sentence_back = indexed_sentence_back[:self.max_length]

        label = self.labels[idx]
        one_hot_label = torch.zeros(4)
        one_hot_label[label - 1] = 1

        return padded_sentence, padded_sentence_back, one_hot_label

# Custom ELMO model
class CustomELMO(nn.Module):
    def __init__(self, vocab_size, embedding_dim, batch_size, embedding_matrix):
        super(CustomELMO, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        self.lstm1 = nn.LSTM(embedding_dim, embedding_dim, batch_first=True, bidirectional=False)
        self.lstm2 = nn.LSTM(embedding_dim, embedding_dim, batch_first=True, bidirectional=False)
        self.linear1 = nn.Linear(embedding_dim, embedding_dim // 2)
        self.linear_out = nn.Linear(embedding_dim // 2, vocab_size)

    def forward(self, X):
        embedded = self.embedding(X)
        lstm1_output, _ = self.lstm1(embedded)
        lstm2_output, _ = self.lstm2(lstm1_output)
        linear1_output = self.linear1(lstm2_output)
        output = self.linear_out(linear1_output)
        return output

# Training function
def train_model(fwd_model, bwd_model, train_dataloader, fwd_opt, bwd_opt, fwd_criterion, bwd_criterion, epochs):
    losses = {'epoch': [], 'train_loss': [], 'valid_loss': []}
    for epoch in range(epochs):
        fwd_model.train()
        bwd_model.train()
        total_loss = 0
        iter = 0
        for (fwd, bwd, label) in tqdm(train_dataloader, desc='Training'):
            fwd = fwd.to(device)
            bwd = bwd.to(device)
            fwd_ip_seq = fwd[:, :-1]
            fwd_target_seq = fwd[:, 1:]
            bwd_ip_seq = bwd[:, :-1]
            bwd_target_seq = bwd[:, 1:]
            fwd_opt.zero_grad()
            bwd_opt.zero_grad()
            fwd_output = fwd_model(fwd_ip_seq)
            bwd_output = bwd_model(bwd_ip_seq)
            fwd_loss = fwd_criterion(fwd_output.reshape(-1, vocab_size), fwd_target_seq.reshape(-1))
            bwd_loss = bwd_criterion(bwd_output.reshape(-1, vocab_size), bwd_target_seq.reshape(-1))
            fwd_loss.backward()
            bwd_loss.backward()
            fwd_opt.step()
            bwd_opt.step()
            total_loss += fwd_loss.item() + bwd_loss.item()
            iter += 1
            if iter % 100 == 0:
                print('Iteration: ', iter, 'Train Loss: ', total_loss / iter)
        train_loss = total_loss / len(train_dataloader)
        print('Train Loss: ', train_loss)
        losses['epoch'].append(epoch)
        losses['train_loss'].append(train_loss)
    
    return losses

# File paths and data preprocessing
train_path = 'train.csv'
test_path = 'test.csv'
preprocessor = DataProcessor(train_path, test_path)
train_sentences, word2idx = preprocessor.process_data('train')
print(len(train_sentences), len(word2idx))
test_sentences = preprocessor.process_data('test')
df1 = pd.read_csv(train_path)
train_Y = df1['Class Index']
df2 = pd.read_csv(test_path)
test_Y = df2['Class Index']

# Create datasets and dataloaders
train_dataset = CustomDataset(train_sentences, train_Y, word2idx)
test_dataset = CustomDataset(test_sentences, test_Y, word2idx)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Train Word2Vec model and create embedding matrix
model = Word2Vec(sentences=train_sentences, vector_size=150, window=5, min_count=1, workers=4)
word_vectors = model.wv
vocab_size = len(word2idx)
emb_dim = 150
emb_matrix = np.zeros((vocab_size, emb_dim))
for word, idx in word2idx.items():
    emb_matrix[idx] = np.zeros(emb_dim) if word == '<PAD>' else word_vectors[word]
emb_matrix = torch.tensor(emb_matrix, dtype=torch.float32)

# Initialize models, optimizers, and loss functions
fwd_model = CustomELMO(vocab_size, emb_dim, 32, emb_matrix)
bwd_model = CustomELMO(vocab_size, emb_dim, 32, emb_matrix)
fwd_opt = optim.Adam(fwd_model.parameters(), lr=0.001)
bwd_opt = optim.Adam(bwd_model.parameters(), lr=0.001)
fwd_criterion = nn.CrossEntropyLoss(ignore_index=0)
bwd_criterion = nn.CrossEntropyLoss(ignore_index=0)

# Train models
epochs = 10
train_model(fwd_model, bwd_model, train_dataloader, fwd_opt, bwd_opt, fwd_criterion, bwd_criterion, epochs)

# Save trained models
torch.save(fwd_model.state_dict(), 'bilstm_f.pt')
torch.save(bwd_model.state_dict(), 'bilstm_b.pt')
