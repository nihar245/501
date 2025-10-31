# -------------------- Cell 1 (markdown) --------------------
# 1. Sentiment Analysis on IMDB Dataset
# 2. Scope of improvement:
#    2.1 Better network architecture.
#    2.2 measures to address overfitting.
#    2.3 Variable length sequences can be better handled.
#    2.4 Better training

# -------------------- Cell 2 --------------------
# !pip install torch==2.0.1 torchtext==0.15.2

# -------------------- Cell 3 --------------------
# !pip install 'portalocker>=2.0.0'

# -------------------- Cell 4 --------------------
from functools import partial
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim

from torchtext import datasets
from torchtext.data.functional import to_map_style_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# -------------------- Cell 5 --------------------
if torch.cuda.is_available():
    device = torch.device(type='cuda', index=0)
else:
    device = torch.device(type='cpu', index=0)

# -------------------- Cell 6 --------------------
train_data = datasets.IMDB(split='train')

# -------------------- Cell 7 --------------------
eval_data = datasets.IMDB(split='test')

# -------------------- Cell 8 --------------------
mapped_train_data = to_map_style_dataset(train_data)

# -------------------- Cell 9 --------------------
# check
print("Type of Mapped Train Data:", type(mapped_train_data))
print("0th data point", mapped_train_data[0])
print("Type of 0th data point", type(mapped_train_data[0]))
label, review = mapped_train_data[0]
print("Label=", label)
print("Review=", review)
print("Type of Label=", type(label))
print("Type of Review=", type(review))

print("iterating over 1 pair:")
for label, review in mapped_train_data:
    print(label)
    print(review)
    break

# -------------------- Cell 10 --------------------
mapped_eval_data = to_map_style_dataset(eval_data)

# -------------------- Cell 11 --------------------
tokenizer = get_tokenizer("basic_english", language="en")

# -------------------- Cell 12 --------------------
min_word_freq = 2


def build_vocab(mapped_train_data, tokenizer):
    reviews = [review for label, review in mapped_train_data]
    vocab = build_vocab_from_iterator(
        map(tokenizer, reviews),
        specials=["<unk>", "<eos>", "<pad>"],
        min_freq=min_word_freq
    )
    vocab.set_default_index(vocab["<unk>"])
    return vocab


# -------------------- Cell 13 --------------------
vocab = build_vocab(mapped_train_data, tokenizer)

# -------------------- Cell 14 --------------------
vocab_size = vocab.__len__()
print(vocab_size)

# -------------------- Cell 15 --------------------
max_seq_len = 256
max_norm = 1
embed_dim = 300
batch_size = 16
text_pipeline = lambda x: vocab(tokenizer(x))

# -------------------- Cell 16 --------------------
sample = text_pipeline("Hello World")
print(sample)
print(type(sample))

# -------------------- Cell 17 --------------------
def collate_data(batch, text_pipeline):
    reviews, targets = [], []

    for label, review in batch:
        review_tokens_ids = text_pipeline(review)

        if max_seq_len:
            review_tokens_ids = review_tokens_ids[:max_seq_len]

        # append <eos> token id (assumed index 1)
        review_tokens_ids.append(1)
        l = len(review_tokens_ids)

        # prepare padded list of length max_seq_len + 1 (here 257)
        pad_len = max_seq_len + 1
        x = [2] * pad_len  # pad token id assumed to be 2
        x[:l] = review_tokens_ids

        reviews.append(x)
        targets.append(label)

    reviews = torch.tensor(reviews, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return reviews, targets

# -------------------- Cell 18 --------------------
traindl = DataLoader(
    mapped_train_data,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=partial(collate_data, text_pipeline=text_pipeline),
)

evaldl = DataLoader(
    mapped_eval_data,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=partial(collate_data, text_pipeline=text_pipeline),
)

# -------------------- Cell 19 --------------------
for i, (labels, reviews) in enumerate(traindl):
    print(labels.shape, reviews.shape)
    break

# -------------------- Cell 20 --------------------
print(vocab(["<unk>", "<eos>", "<pad>"]))

# -------------------- Cell 21 --------------------
class SentiNN(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size):
        super().__init__()
        self.e = nn.Embedding(input_size, embed_size)
        self.dropout = nn.Dropout(0.2)
        self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.out = nn.Linear(in_features=hidden_size, out_features=2)

    def forward(self, x):
        x = self.e(x)
        x = self.dropout(x)
        outputs, hidden = self.rnn(x)
        hidden.squeeze_(0)
        logits = self.out(hidden)
        return logits

# -------------------- Cell 22 --------------------
embed_size = 128
hidden_size = 256

sentinn = SentiNN(vocab_size, embed_size, hidden_size).to(device)

loss_fn = nn.CrossEntropyLoss(ignore_index=2).to(device)
lr = 0.001
opt = optim.Adam(params=sentinn.parameters(), lr=lr)

# -------------------- Cell 23 --------------------
def train_one_epoch():
    sentinn.train()
    track_loss = 0
    num_correct = 0

    for i, (reviews_ids, sentiments) in enumerate(traindl):
        reviews_ids = reviews_ids.to(device)
        sentiments = sentiments.to(device) - 1
        logits = sentinn(reviews_ids)
        loss = loss_fn(logits, sentiments)

        track_loss += loss.item()
        num_correct += (torch.argmax(logits, dim=1) == sentiments).type(torch.float).sum().item()

        running_loss = round(track_loss / (i + (reviews_ids.shape[0] / batch_size)), 4)
        running_acc = round((num_correct / ((i * batch_size + reviews_ids.shape[0])))*100, 4)

        opt.zero_grad()
        loss.backward()
        opt.step()

    epoch_loss = running_loss
    epoch_acc = running_acc
    return epoch_loss, epoch_acc

# -------------------- Cell 24 --------------------
def eval_one_epoch():
    sentinn.eval()
    track_loss = 0
    num_correct = 0

    for i, (reviews_ids, sentiments) in enumerate(evaldl):

        reviews_ids = reviews_ids.to(device)
        sentiments = sentiments.to(device) - 1
        logits = sentinn(reviews_ids)

        loss = loss_fn(logits, sentiments)

        track_loss += loss.item()
        num_correct += (torch.argmax(logits, dim=1) == sentiments).type(torch.float).sum().item()

        running_loss = round(track_loss / (i + (reviews_ids.shape[0] / batch_size)), 4)
        running_acc = round((num_correct / ((i * batch_size + reviews_ids.shape[0])))*100, 4)

    epoch_loss = running_loss
    epoch_acc = running_acc
    return epoch_loss, epoch_acc

# -------------------- Cell 25 --------------------
n_epochs = 10

for e in range(n_epochs):
    print("Epoch=", e+1, sep="", end=", ")
    epoch_loss, epoch_acc = train_one_epoch()
    print("Train Loss=", epoch_loss, "Train Acc", epoch_acc)
    epoch_loss, epoch_acc = eval_one_epoch()
    print("Eval Loss=", epoch_loss, "Eval Acc", epoch_acc)