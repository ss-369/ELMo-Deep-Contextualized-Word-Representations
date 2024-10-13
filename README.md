
---

# ELMo: Deep Contextualized Word Representations

## Description

This project implements an **ELMo (Embeddings from Language Models)** architecture from scratch using PyTorch. ELMo generates deep contextualized word embeddings through stacked **Bi-LSTMs**, capturing both the syntactic and semantic properties of words based on their context in a sentence. The model is pre-trained using a bidirectional language modeling objective and is further evaluated on a downstream text classification task.

## Requirements

- **Language**: Python
- **Framework**: PyTorch
- **Dataset**: AG News Classification Dataset
  - Use the **Description** column for training word embeddings.
  - Use the **Label/Index** column for the downstream classification task.

## Features

### 1. ELMo Architecture
- The core ELMo model consists of:
  - Input embedding layer (optionally pretrained with Word2Vec).
  - Stacked **Bi-LSTM** layers (2 layers).
  - Trainable or fixed λs for combining word embeddings from different layers.

### 2. Model Pre-training
- Pre-train the ELMo model using **bidirectional language modeling**. The forward and backward **Bi-LSTM** models predict the next word and the previous word, respectively, to capture contextual word representations.

### 3. Downstream Task: Text Classification
- Evaluate the pretrained ELMo model on a **4-way text classification task** using the AG News Dataset.

## Hyperparameter Tuning

### 1. Trainable λs
- Train the model with λs as trainable parameters.

### 2. Frozen λs
- Randomly initialize λs and freeze them during training.

### 3. Learnable Function
- Learn a custom function to combine the embeddings across different Bi-LSTM layers.

## Evaluation Metrics

- Accuracy
- Precision, Recall, F1 Score
- Confusion Matrix

These metrics are evaluated for both the **pretraining** and **downstream classification** tasks.

## How to Run

### 1. Train ELMo Model
To pre-train the ELMo model on bidirectional language modeling:
```bash
python ELMO.py
```

### 2. Train Classifier for Downstream Task
To train the classifier using the word representations obtained from ELMo:
```bash
python classification.py
```

### Pretrained Models

1. **Bi-LSTM Model**: `bilstm.pt`
2. **Classifier Model**: `classifier.pt`

You can either load these models directly from the directory or provide a link to download them from external storage (e.g., OneDrive).

## Submission

1. **Source Code**:
   - `ELMO.py`: Train the Bi-LSTM on the language modeling task.
   - `classification.py`: Train the classifier on the downstream task using the Bi-LSTM embeddings.

2. **Pretrained Models**:
   - `bilstm.pt`: Pretrained Bi-LSTM model.
   - `classifier.pt`: Pretrained classifier model.

3. **Report (PDF)**:
   - Hyperparameters used for pretraining and the downstream task.
   - Evaluation metrics (accuracy, F1, precision, recall).
   - Analysis of results comparing ELMo with Word2Vec and SVD (from previous assignments).

4. **README**:
   - Instructions on how to execute the code, load the pretrained models, and assumptions made during implementation.

Please upload the pretrained models to an external storage (OneDrive) and include the link here.

## Analysis

- Analyze the performance of the ELMo embeddings compared to Word2Vec and SVD on the downstream classification task.
- Use evaluation metrics such as accuracy, F1, precision, recall, and confusion matrix to compare these models.
- Discuss the impact of hyperparameter tuning on the performance, especially the role of λs.

## Resources

1. [Deep Contextualized Word Representations (ELMo)](https://arxiv.org/abs/1802.05365)
2. [Bidirectional Language Modeling using LSTMs](https://arxiv.org/pdf/1602.02410.pdf)
3. [Text Generation with Bi-LSTM in PyTorch](https://towardsdatascience.com/text-generation-with-bi-lstm-in-pytorch-5fda6e7cc22c)

---

