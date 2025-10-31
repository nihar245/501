# -------------------- Cell 1 (markdown) --------------------
# 1. Basic code for machine translation (German -> English) using torchtext, Seq2Seq model using GRU.
# 2. There is a lot of scope of improvement like better architecture, better training and addressing overfitting.
# 3. You can also refer my another code which doesn't use torchtext at https://www.kaggle.com/code/priyankdl/machine-translation-seq-2-seq
# 4. You can also refer my another code which uses Bahdanau Attention mechanism but doesn't use torchtext  at https://www.kaggle.com/code/priyankdl/machine-translation-seq-2-seq-bahdanau-attention

# -------------------- Cell 2 --------------------
# !pip install torchtext==0.15.2 torch==2.0.1

# -------------------- Cell 3 --------------------
# !pip install 'portalocker>=2.0.0'

# -------------------- Cell 4 --------------------
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
from typing import Iterable, List
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

# -------------------- Cell 5 --------------------
if torch.cuda.is_available():
    device = torch.device(type='cuda', index=0)
else:
    device = torch.device(type='cpu', index=0)
print(device)

# -------------------- Cell 6 --------------------
multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"

SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

# -------------------- Cell 7 --------------------
# !pip install -U torchdata
# !pip install -U spacy
# !python -m spacy download en_core_web_sm
# !python -m spacy download de_core_news_sm

# -------------------- Cell 8 --------------------
de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')
en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    if language == 'de':
        for data_sample in data_iter:
            yield de_tokenizer(data_sample[0])
    elif language == 'en':
        for data_sample in data_iter:
            yield en_tokenizer(data_sample[1])

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3

special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))

vocab_de = build_vocab_from_iterator(yield_tokens(train_iter, 'de'),
                                     min_freq=1,
                                     specials=special_symbols,
                                     special_first=True)
vocab_de.set_default_index(UNK_IDX)

train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
vocab_en = build_vocab_from_iterator(yield_tokens(train_iter, 'en'),
                                     min_freq=1,
                                     specials=special_symbols,
                                     special_first=True)
vocab_en.set_default_index(UNK_IDX)

# -------------------- Cell 9 --------------------
print(type(vocab_en))
print("English Vocab Length:", vocab_en.__len__())
print("German Vocab Length:", vocab_de.__len__())

# -------------------- Cell 10 --------------------
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:

        src_sample = src_sample.rstrip("\n")
        tgt_sample = tgt_sample.rstrip("\n")

        src_tokens = de_tokenizer(src_sample)
        tgt_tokens = en_tokenizer(tgt_sample)

        src_ids = vocab_de(src_tokens)
        tgt_ids = vocab_en(tgt_tokens)

        src_ids.append(EOS_IDX)
        tgt_ids.append(EOS_IDX)

        tgt_ids.insert(0, BOS_IDX)

        src_tensor = torch.tensor(src_ids)
        tgt_tensor = torch.tensor(tgt_ids)

        src_batch.append(src_tensor)
        tgt_batch.append(tgt_tensor)

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)
    return src_batch, tgt_batch

# -------------------- Cell 11 --------------------
class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, dropout_p=0.1):
        super().__init__()
        self.e = nn.Embedding(input_size, embed_size)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)

    def forward(self, x):
        x = self.e(x)
        x = self.dropout(x)
        outputs, hidden = self.gru(x)
        return outputs, hidden

# -------------------- Cell 12 --------------------
class Decoder(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size):
        super().__init__()
        self.e = nn.Embedding(output_size, embed_size)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.lin = nn.Linear(hidden_size, output_size)
        self.lsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, prev_hidden):
        x = self.e(x)
        x = self.relu(x)
        output, hidden = self.gru(x, prev_hidden)
        y = self.lin(output)
        y = self.lsoftmax(y)
        return y, hidden

# -------------------- Cell 13 --------------------
def train_one_epoch():
    encoder.train()
    decoder.train()
    track_loss = 0

    train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    train_dataloader = DataLoader(train_iter, batch_size=batch_size, collate_fn=collate_fn)

    for i, (s_ids, t_ids) in enumerate(train_dataloader):

        s_ids = s_ids.to(device)
        t_ids = t_ids.to(device)
        encoder_outputs, encoder_hidden = encoder(s_ids)
        decoder_hidden = encoder_hidden
        yhats, decoder_hidden = decoder(t_ids[:, 0:-1], decoder_hidden)

        gt = t_ids[:, 1:]

        yhats_reshaped = yhats.view(-1, yhats.shape[-1])

        gt = gt.reshape(-1)

        loss = loss_fn(yhats_reshaped, gt)
        track_loss += loss.item()

        opte.zero_grad()
        optd.zero_grad()

        loss.backward()

        opte.step()
        optd.step()

    return track_loss / (i + 1)

# -------------------- Cell 14 --------------------
def eval_one_epoch(e, n_epochs):
    encoder.eval()
    decoder.eval()
    track_loss = 0

    val_iter = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    val_dataloader = DataLoader(val_iter, batch_size=1, collate_fn=collate_fn)

    with torch.no_grad():
        for i, (s_ids, t_ids) in enumerate(val_dataloader):
            s_ids = s_ids.to(device)
            t_ids = t_ids.to(device)
            encoder_outputs, encoder_hidden = encoder(s_ids)
            decoder_hidden = encoder_hidden
            input_id = t_ids[:, 0]
            yhats = []
            if e + 1 == n_epochs:
                pred_sentence = ""
            for j in range(1, t_ids.shape[1]):
                probs, decoder_hidden = decoder(input_id.unsqueeze(1), decoder_hidden)
                yhats.append(probs)
                _, input_id = torch.topk(probs, 1, dim=-1)
                input_id = input_id.squeeze(1, 2)
                if e + 1 == n_epochs:
                    word = vocab_en.lookup_token(input_id.item())
                    pred_sentence += word + " "
                if input_id.item() == 3:
                    break

            if e + 1 == n_epochs:
                src_sentence_tokens = vocab_de.lookup_tokens(s_ids.tolist()[0])
                src_sentence = " ".join(src_sentence_tokens)
                gt_sentence_tokens = vocab_en.lookup_tokens(t_ids[:, 1:].tolist()[0])
                gt_sentence = " ".join(gt_sentence_tokens)
                print("\n-----------------------------------")
                print("Source Sentence:", src_sentence)
                print("GT Sentence:", gt_sentence)
                print("Predicted Sentence:", pred_sentence)

            yhats_cat = torch.cat(yhats, dim=1)
            yhats_reshaped = yhats_cat.view(-1, yhats_cat.shape[-1])
            gt = t_ids[:, 1:j+1]

            gt = gt.view(-1)

            loss = loss_fn(yhats_reshaped, gt)
            track_loss += loss.item()

        if e + 1 == n_epochs:
            print("-----------------------------------")
        return track_loss / (i + 1)

# -------------------- Cell 15 --------------------
embed_size = 300
hidden_size = 512
batch_size = 32

encoder = Encoder(vocab_de.__len__(), embed_size, hidden_size).to(device)
decoder = Decoder(vocab_en.__len__(), embed_size, hidden_size).to(device)

loss_fn = nn.NLLLoss(ignore_index=1).to(device)
lr = 0.001
opte = optim.Adam(params=encoder.parameters(), lr=lr, weight_decay=0.001)
optd = optim.Adam(params=decoder.parameters(), lr=lr, weight_decay=0.001)

n_epochs = 20

for e in range(n_epochs):
    print("Epoch=", e+1, sep="", end=", ")
    print("Train Loss=", round(train_one_epoch(), 4), sep="", end=", ")
    print("Eval Loss=", round(eval_one_epoch(e, n_epochs), 4), sep="")