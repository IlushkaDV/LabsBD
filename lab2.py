import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import nltk
import re
import seaborn as sns
from nltk.tokenize import wordpunct_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, confusion_matrix
)
from collections import Counter

# === Настройки ===
nltk.download('punkt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 1. Загрузка и токенизация текста ===
with open("data.txt", "r", encoding="utf-8") as f:
    raw_text = f.read().lower()

tokens = [w for w in wordpunct_tokenize(raw_text) if re.fullmatch(r"[а-яё\-]+", w)]


# === 2. Подготовка данных ===
def bag_of_words(tokens):
    vocab = list(set(tokens))
    word2idx = {w: i for i, w in enumerate(vocab)}
    X = np.zeros((len(tokens) - 1, len(vocab)))
    y = [word2idx[tokens[i + 1]] for i in range(len(tokens) - 1)]
    for i in range(len(tokens) - 1):
        X[i][word2idx[tokens[i]]] = 1
    return torch.FloatTensor(X), torch.LongTensor(y), word2idx, vocab


def generate_ngrams(tokens, n=5):
    return [tokens[i:i + n] for i in range(len(tokens) - n)]


def prepare_ngram_data(ngrams):
    vocab = list(set(word for gram in ngrams for word in gram))
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}
    X = [[word2idx[word] for word in gram[:-1]] for gram in ngrams]
    y = [word2idx[gram[-1]] for gram in ngrams]
    return torch.LongTensor(X), torch.LongTensor(y), word2idx, idx2word, vocab


# === 3. Определение моделей ===
class FeedForwardModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_heads=2, ff_dim=128, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x).permute(1, 0, 2)  # [seq_len, batch, embed]
        x = self.transformer(x)
        return self.fc(x[-1])


# === 4. Обучение модели ===
def train_model(model, X_train, y_train, X_val, y_val, epochs=200, lr=0.001):
    model.to(device)
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    acc_list = []

    for epoch in range(epochs):
        model.train()
        loss = criterion(model(X_train), y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = torch.argmax(model(X_val), dim=1)
            acc = accuracy_score(y_val.cpu(), preds.cpu())
            prec = precision_score(y_val.cpu(), preds.cpu(), average='macro', zero_division=0)
            rec = recall_score(y_val.cpu(), preds.cpu(), average='macro', zero_division=0)

        acc_list.append(acc)
        print(f"Epoch {epoch + 1}: Loss={loss.item():.4f}, Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}")

    return acc_list


# === 5. Генерация текста ===
def generate_text(model, start_words, word2idx, idx2word, max_len=50):
    model.eval()
    input_ids = [word2idx.get(w, 0) for w in start_words]
    for _ in range(max_len):
        input_tensor = torch.LongTensor([input_ids[-4:]]).to(device)
        with torch.no_grad():
            next_token = torch.argmax(model(input_tensor), dim=1).item()
        input_ids.append(next_token)
    return " ".join([idx2word[i] for i in input_ids])


# === 6. Обучение BoW модели ===
print("\n===== Мешок слов =====")
X_bow, y_bow, word2idx_bow, vocab_bow = bag_of_words(tokens)
X_train_bow, X_val_bow, y_train_bow, y_val_bow = train_test_split(X_bow, y_bow, test_size=0.2, random_state=42)

model_bow = FeedForwardModel(input_size=len(vocab_bow), output_size=len(vocab_bow))
acc_bow = train_model(model_bow, X_train_bow, y_train_bow, X_val_bow, y_val_bow)

# === 7. Обучение Transformer на N-граммах ===
print("\n===== N-граммы + Transformer =====")
ngrams = generate_ngrams(tokens, n=5)
X_ngram, y_ngram, word2idx_ng, idx2word_ng, vocab_ng = prepare_ngram_data(ngrams)
X_train_ng, X_val_ng, y_train_ng, y_val_ng = train_test_split(X_ngram, y_ngram, test_size=0.2, random_state=42)

model_ngram = TransformerModel(len(vocab_ng))
acc_ng = train_model(model_ngram, X_train_ng, y_train_ng, X_val_ng, y_val_ng)

# === 8. Генерация текста ===
print("\n=== Генерация текста ===")
start_words = ngrams[0][:3]
print("Начало:", " ".join(start_words))
print("Продолжение:", generate_text(model_ngram, start_words, word2idx_ng, idx2word_ng))

# === 9. Accuracy-график ===
plt.figure(figsize=(8, 6))
plt.plot(acc_bow, label='BoW')
plt.plot(acc_ng, label='N-grams + Transformer')
plt.title("Accuracy по эпохам")
plt.xlabel("Эпоха")
plt.ylabel("Accuracy")
plt.grid()
plt.legend()
plt.show()


# === 10. Обучение с метриками и графиками ===
def train_model_with_metrics(model, X_train, y_train, X_val, y_val, epochs=200, lr=0.001):
    model.to(device)
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    acc_list, prec_list, rec_list, loss_list = [], [], [], []

    for epoch in range(epochs):
        model.train()
        loss = criterion(model(X_train), y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = torch.argmax(model(X_val), dim=1)
            acc = accuracy_score(y_val.cpu(), preds.cpu())
            prec = precision_score(y_val.cpu(), preds.cpu(), average='macro', zero_division=0)
            rec = recall_score(y_val.cpu(), preds.cpu(), average='macro', zero_division=0)

        acc_list.append(acc)
        prec_list.append(prec)
        rec_list.append(rec)
        loss_list.append(loss.item())

        if (epoch + 1) % 5 == 0:
            print(f"\n📣 Генерация после эпохи {epoch + 1}")
            print("Начало:", " ".join(ngrams[0][:4]))
            print("Продолжение:", generate_text(model, ngrams[0][:4], word2idx_ng, idx2word_ng))

    return acc_list, prec_list, rec_list, loss_list, preds.cpu(), y_val.cpu()


def plot_metrics(acc, prec, rec, losses, title=""):
    epochs = range(1, len(acc) + 1)
    plt.figure(figsize=(12, 8))
    metrics = [(acc, 'Accuracy'), (prec, 'Precision'), (rec, 'Recall'), (losses, 'Loss')]
    colors = ['blue', 'orange', 'green', 'red']

    for i, (data, label) in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
        plt.plot(epochs, data, label=label, color=colors[i - 1])
        plt.title(f"{title} - {label}")
        plt.grid()

    plt.tight_layout()
    plt.show()


def plot_top_confusion_matrix(preds, targets, idx2word, top_n=30):
    word_counts = Counter(targets.numpy())
    top_indices = [w for w, _ in word_counts.most_common(top_n)]
    remap = {old: new for new, old in enumerate(top_indices)}

    mask = [(t.item() in top_indices) and (p.item() in top_indices) for t, p in zip(targets, preds)]
    filtered_preds = preds[mask]
    filtered_targets = targets[mask]

    y_true = [remap[t.item()] for t in filtered_targets]
    y_pred = [remap[p.item()] for p in filtered_preds]
    labels = [idx2word[i] for i in top_indices]

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix — Топ-30 слов")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


# === 11. Финальное переобучение и вывод ===
print("\n===== ПЕРЕОБУЧЕНИЕ TRANSFORMER с метриками =====")
model_ngram = TransformerModel(len(vocab_ng))
acc, prec, rec, losses, preds, targets = train_model_with_metrics(model_ngram, X_train_ng, y_train_ng, X_val_ng,
                                                                  y_val_ng)

plot_metrics(acc, prec, rec, losses, title="Transformer")
plot_top_confusion_matrix(preds, targets, idx2word_ng)
