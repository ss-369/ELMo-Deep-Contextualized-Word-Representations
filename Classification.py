from ELMO import Dataprocess, CustomDataset
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from ELMO import CustomELMO

# Set the device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to plot confusion matrix
def plot_confusion_matrix(true_labels, predictions, file_name):
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# Function to print evaluation metrics
def print_evaluation_metrics(true_labels, predictions):
    accuracy = accuracy_score(true_labels, predictions)
    print("Accuracy:", accuracy)
    precision = precision_score(true_labels, predictions, average='weighted')
    print("Precision:", precision)
    recall = recall_score(true_labels, predictions, average='weighted')
    print("Recall:", recall)
    f1 = f1_score(true_labels, predictions, average='weighted')
    print("F1 Score:", f1)

# Custom classifier using ELMo embeddings
class CustomClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, fw_embeddings, bw_embeddings,
                 forward_lstm1, forward_lstm2, backward_lstm1, backward_lstm2, num_classes, requires_grad=True):
        super(CustomClassifier, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.fw_embeddings = nn.Embedding.from_pretrained(torch.tensor(fw_embeddings, dtype=torch.float).to(device), padding_idx=0)
        self.fw_embeddings.weight.requires_grad = False

        self.bw_embeddings = nn.Embedding.from_pretrained(torch.tensor(bw_embeddings, dtype=torch.float).to(device), padding_idx=0)
        self.bw_embeddings.weight.requires_grad = requires_grad

        self.weights = nn.Parameter(torch.tensor([0.33, 0.33, 0.33], dtype=torch.float).to(device), requires_grad=requires_grad)

        self.forward_lstm1 = forward_lstm1
        self.forward_lstm2 = forward_lstm2
        self.backward_lstm1 = backward_lstm1
        self.backward_lstm2 = backward_lstm2

        self.rnn = nn.LSTM(input_size=2 * embedding_dim, hidden_size=embedding_dim,
                           num_layers=2, batch_first=True, bidirectional=False).to(device)
        self.linear1 = nn.Linear(embedding_dim, embedding_dim // 2).to(device)
        self.linear_out = nn.Linear(embedding_dim // 2, num_classes).to(device)

    def forward(self, forward_input, backward_input):
        fw_embed = self.fw_embeddings(forward_input)
        bw_embed = self.bw_embeddings(backward_input)

        fw_lstm1, _ = self.forward_lstm1(fw_embed)
        fw_lstm2, _ = self.forward_lstm2(fw_lstm1)
        bw_lstm1, _ = self.backward_lstm1(bw_embed)
        bw_lstm2, _ = self.backward_lstm2(bw_lstm1)

        embed = torch.cat((fw_embed, bw_embed), dim=2)
        lstm1 = torch.cat((fw_lstm1, bw_lstm1), dim=2)
        lstm2 = torch.cat((fw_lstm2, bw_lstm2), dim=2)

        elmo_out = self.weights[0] * embed + self.weights[1] * lstm1 + self.weights[2] * lstm2
        elmo_max = torch.max(elmo_out, dim=1)[0]
        output, _ = self.rnn(elmo_max)
        output = self.linear1(output)
        output = self.linear_out(output)

        return output

# Classifier with a linear function
class Classifier_Linear(nn.Module):
    def __init__(self, vocab_size, embedding_dim, fw_embeddings, bw_embeddings,
                 forward_lstm1, forward_lstm2, backward_lstm1, backward_lstm2, num_classes, requires_grad=True):
        super(Classifier_Linear, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.fw_embeddings = nn.Embedding.from_pretrained(torch.tensor(fw_embeddings, dtype=torch.float).to(device), padding_idx=0)
        self.fw_embeddings.weight.requires_grad = False

        self.bw_embeddings = nn.Embedding.from_pretrained(torch.tensor(bw_embeddings, dtype=torch.float).to(device), padding_idx=0)
        self.bw_embeddings.weight.requires_grad = True

        self.linear_function = nn.Linear(3 * 2 * embedding_dim, 2 * embedding_dim).to(device)
        self.rnn = nn.LSTM(input_size=2 * embedding_dim, hidden_size=embedding_dim,
                           num_layers=2, batch_first=True, bidirectional=False).to(device)
        self.linear1 = nn.Linear(embedding_dim, embedding_dim // 2).to(device)
        self.linear_out = nn.Linear(embedding_dim // 2, num_classes).to(device)

    def forward(self, forward_input, backward_input):
        fw_embed = self.fw_embeddings(forward_input)
        bw_embed = self.bw_embeddings(backward_input)

        fw_lstm1, _ = self.forward_lstm1(fw_embed)
        fw_lstm2, _ = self.forward_lstm2(fw_lstm1)
        bw_lstm1, _ = self.backward_lstm1(bw_embed)
        bw_lstm2, _ = self.backward_lstm2(bw_lstm1)

        embed = torch.cat((fw_embed, bw_embed), dim=2)
        lstm1 = torch.cat((fw_lstm1, bw_lstm1), dim=2)
        lstm2 = torch.cat((fw_lstm2, bw_lstm2), dim=2)

        elmo_out = self.linear_function(torch.cat((embed, lstm1, lstm2), dim=2))
        elmo_max = torch.max(elmo_out, dim=1)[0]
        output, _ = self.rnn(elmo_max)
        output = self.linear1(output)
        output = self.linear_out(output)

        return output

# Training function for the classifier
def train_classifier(model, train_dataloader, optimizer, criterion, device, epochs=5):
    for epoch in range(epochs):
        print("Epoch: {}".format(epoch))
        model.train()
        train_loss = 0
        iter = 0
        for batch in tqdm(train_dataloader, desc="Training"):
            optimizer.zero_grad()
            batch = tuple(t.to(device) for t in batch)
            forward, backward, labels = batch
            logits = model(forward, backward)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(logits, dim=1)
            train_loss += loss.item()
            iter += 1
            if iter % 100 == 0:
                print("Iteration: {}, Train Loss: {}".format(iter, loss.item()))

# Testing function for the classifier
def test_model(model, test_loader, criterion, split, file_name):
    model.eval()
    model.to(device)
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    true_labels, predictions = [], []
    with torch.no_grad():
        for inputs, backward, labels in test_loader:
            inputs = inputs.to(device)
            backward = backward.to(device)
            labels = labels.to(device)
            outputs = model(inputs, backward)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_labels = labels.argmax(dim=1)
            true_labels.extend(correct_labels.cpu().numpy())
            predictions.extend(predicted.cpu().numpy())
            correct_predictions += (predicted == correct_labels).sum().item()
            total_samples += labels.size(0)

    average_loss = total_loss / len(test_loader)
    accuracy = correct_predictions / total_samples
    print(f"{split} Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}")
    print_evaluation_metrics(true_labels, predictions)
    plot_confusion_matrix(true_labels, predictions, f"elmo_{file_name}_{split}")

# Main script to train and test the classifier
train_path = 'train.csv'
test_path = 'test.csv'
preprocessor = Preprocess(train_path, test_path)
train_sentences, word2idx = preprocessor.process_data('train')
print(len(train_sentences), len(word2idx))
test_sentences = preprocessor.process_data('test')
df1 = pd.read_csv(train_path)
train_Y = df1['Class Index']
df2 = pd.read_csv(test_path)
test_Y = df2['Class Index']
train_dataset = ElmoDataset(train_sentences, train_Y, word2idx)
test_dataset = ElmoDataset(test_sentences, test_Y, word2idx)
vocab_size = len(word2idx)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Load pre-trained forward and backward LSTM models
forward_model = torch.load('bilstm_f.pt')
backward_model = torch.load('bilstm_b.pt')
forward_embeddings = list(forward_model.parameters())[0].cpu().detach().numpy()
backward_embeddings = list(backward_model.parameters())[0].cpu().detach().numpy()
print(forward_embeddings.shape, backward_embeddings.shape)

EMBEDDING_DIM = 150
VOCAB_SIZE = vocab_size
BATCH_SIZE = 32
num_classes = 4

# Instantiate and train the classifier
classifier1 = CustomClassifier(VOCAB_SIZE, EMBEDDING_DIM, forward_embeddings, backward_embeddings,
                               forward_model.lstm1, forward_model.lstm2, backward_model.lstm1, backward_model.lstm2, num_classes, requires_grad=False)
classifier1.to(device)
optimizer1 = optim.Adam(classifier1.parameters(), lr=0.001)
criterion1 = nn.CrossEntropyLoss()

classifier2 = CustomClassifier(VOCAB_SIZE, EMBEDDING_DIM, forward_embeddings, backward_embeddings,
                               forward_model.lstm1, forward_model.lstm2, backward_model.lstm1, backward_model.lstm2, num_classes, requires_grad=True)
classifier2.to(device)
optimizer2 = optim.Adam(classifier2.parameters(), lr=0.001)
criterion2 = nn.CrossEntropyLoss()

# Train classifier2 and evaluate
train_classifier(classifier2, train_dataloader, optimizer2, criterion2, device, epochs=3)
print(classifier2.weights)
test_model(classifier2, train_dataloader, criterion2, "train", "unfrozen_wt")
test_model(classifier2, test_dataloader, criterion2, "test", "unfrozen_wt")

# Save the trained models
torch.save(classifier1.state_dict(), 'frozen_wt.pt')
torch.save(classifier2.state_dict(), 'unfrozen_wt.pt')

# Instantiate, train, and evaluate classifier with a linear function
classifier3 = Classifier_Linear(VOCAB_SIZE, EMBEDDING_DIM, forward_embeddings, backward_embeddings,
                                forward_model.lstm1, forward_model.lstm2, backward_model.lstm1, backward_model.lstm2, num_classes)
classifier3.to(device)
optimizer3 = optim.Adam(classifier3.parameters(), lr=0.001)
criterion3 = nn.CrossEntropyLoss()

train_classifier(classifier3, train_dataloader, optimizer3, criterion3, device, epochs=5)
test_model(classifier3, train_dataloader, criterion3, "train", "Linear_function")
test_model(classifier3, test_dataloader, criterion3, "test", "Linear_function")
