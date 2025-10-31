# -------------------- Cell 1 --------------------
import torch
import torch.nn as nn
import random
from collections import Counter
import torch.nn.functional as F

# -------------------- Cell 2 --------------------
corpus = "we use cbow model to learn word embeddings from context".split()
word2idx = {w: i for i, w in enumerate(set(corpus))}
idx2word = {i: w for w, i in word2idx.items()}
vocab_size = len(word2idx)
print("Vocab:", word2idx)

# -------------------- Cell 3 --------------------
window, pairs = 2, []
for i, w in enumerate(corpus):
    ctx = [word2idx[corpus[j]]
           for j in range(max(0, i - window), min(len(corpus), i + window + 1)) if j != i]
    pairs.append((ctx, word2idx[w]))

# -------------------- Cell 4 --------------------
class CBOW(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.in_emb = nn.Embedding(vocab_size, embed_dim)
        self.out_emb = nn.Embedding(vocab_size, embed_dim)

    def forward(self, context):
        v = self.in_emb(context).mean(0)
        scores = torch.matmul(self.out_emb.weight, v)
        return scores

# -------------------- Cell 5 --------------------
embed_dim = 50
model = CBOW(vocab_size, embed_dim)
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
for epoch in range(50):
    total = 0
    random.shuffle(pairs)
    for ctx, tgt in pairs:
        ctx = torch.tensor(ctx)
        tgt = torch.tensor([tgt])

        scores = model(ctx)
        loss = F.cross_entropy(scores.unsqueeze(0), tgt)

        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss={total/len(pairs):.4f}")

# -------------------- Cell 6 --------------------
def predict_center(context_words, topk=3):
    ctx_idx = torch.tensor([word2idx[w] for w in context_words if w in word2idx])
    with torch.no_grad():
        scores = model(ctx_idx)
        probs = F.softmax(scores, dim=0)
        vals, idxs = torch.topk(probs, topk)
    return [(idx2word[i.item()], vals[j].item()) for j, i in enumerate(idxs)]

print(predict_center(["we", "cbow"], topk=5))

