# -------------------- Cell 1 (markdown) --------------------
# Basic code for machine translation using Seq2Seq (Bahdanau Attention Mechanism Used). Other possibilities include: Better Initialization, Better Architecture, Better Training.
# 1. Prepare Data
#     1.1 Read data
#     1.2 Create normalized pairs (create + normalize (unicode 2 ascii, remove non-letter characters, trim)) (list of lists, each list will be a pair)
#     1.3 Filter pairs
#     1.4 Build vocab (Write Vocab class, Create Vocab objects for each class, and build vocab)
# 2. Define Encoder and Decoder
# 3. Prepare Data and DataLoader
# 4. Training
# 5. Evaluation

# -------------------- Cell 2 --------------------
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from io import open
import unicodedata
import re

# -------------------- Cell 3 --------------------
if torch.cuda.is_available():
    device = torch.device(type='cuda', index=0)
else:
    device = torch.device(type='cpu', index=0)

# -------------------- Cell 4 --------------------
# unicode 2 ascii, remove non-letter characters, trim
def normalizeString(s):
    sres = ""
    for ch in unicodedata.normalize('NFD', s):
        # Return the normal form ('NFD') for the Unicode string s.
        if unicodedata.category(ch) != 'Mn':
            # The function in the first part returns the general
            # category assigned to the character ch as string.
            # "Mn' refers to Mark, Nonspacing
            sres += ch
    sres = re.sub(r"([.!?])", r" \1", sres)
    # inserts a space before any occurrence of ".", "!", or "?" in the string sres.
    sres = re.sub(r"[^a-zA-Z!?]+", r" ", sres)
    # this line of code replaces any sequence of characters in sres
    # that are not letters (a-z or A-Z) or the punctuation marks
    # "!" or "?" with a single space character.
    return sres.strip()

# create list of pairs (list of lists) (no filtering)
def createNormalizedPairs():
    initpairs = []
    for pair in data:
        s1, s2 = pair.split('\t')
        s1 = normalizeString(s1.lower().strip())
        s2 = normalizeString(s2.lower().strip())
        initpairs.append([s1, s2])
    # print(len(initpairs))
    return initpairs

# filter pairs
max_length = 10
def filterPairs(initpairs):
    # filtering conditions in addition to max_length
    eng_prefixes = (
        "i am ", "i m ",
        "he is", "he s ",
        "she is", "she s ",
        "you are", "you re ",
        "we are", "we re ",
        "they are", "they re "
    )

    pairs = []
    for pair in initpairs:
        if len(pair[0].split(" ")) < max_length and len(pair[1].split(" ")) < max_length and pair[0].lower().startswith(eng_prefixes):
            pairs.append(pair)

    print("Number of pairs after filtering:", len(pairs))
    return pairs  # list of lists

# -------------------- Cell 5 --------------------
class Vocab:
    def __init__(self, name):
        self.name = name
        self.word2index = {'SOS': 0, 'EOS': 1}
        self.index2word = {0: 'SOS', 1: 'EOS'}
        self.word2count = {}
        self.nwords = 2

    def buildVocab(self, s):
        for word in s.split(" "):
            if word not in self.word2index:
                self.word2index[word] = self.nwords
                self.index2word[self.nwords] = word
                self.word2count[word] = 1
                self.nwords += 1
            else:
                self.word2count[word] += 1

# -------------------- Cell 6 --------------------
class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, dropout_p=0.1):
        super().__init__()
        self.e = nn.Embedding(input_size, embed_size)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)

    def forward(self, x, lengths):
        x = self.e(x)
        x = self.dropout(x)
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        outputs, hidden = self.gru(x)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        return outputs, hidden

# -------------------- Cell 7 --------------------
class Decoder(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size):
        super().__init__()
        self.e = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout()
        self.gru = nn.GRU(embed_size + hidden_size, hidden_size, batch_first=True)
        self.lin = nn.Linear(hidden_size, output_size)
        self.lsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, context, prev_hidden, lengths):
        x = self.e(x)
        x = self.dropout(x)
        x = torch.cat((x, context), dim=2)
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        output, hidden = self.gru(x, prev_hidden)
        output, _ = pad_packed_sequence(output, batch_first=True)
        y = self.lin(output)
        y = self.lsoftmax(y)
        return y, hidden

# -------------------- Cell 8 --------------------
class Bahdanau_Attention(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size, new_hidden_size):
        super().__init__()
        self.eh2nh = nn.Linear(in_features=encoder_hidden_size, out_features=new_hidden_size)
        self.dh2nh = nn.Linear(in_features=decoder_hidden_size, out_features=new_hidden_size)
        self.score = nn.Linear(in_features=new_hidden_size, out_features=1)

    def forward(self, query, keys):
        query = self.dh2nh(query)
        keys = self.eh2nh(keys)
        att_score = self.score(torch.tanh(query.permute(1,0,2) + keys))
        att_score = att_score.squeeze(2).unsqueeze(1)
        att_weights = F.softmax(att_score, dim=-1)
        context = torch.bmm(att_weights, keys)

        return context, att_weights

# -------------------- Cell 9 --------------------
def get_input_ids(sentence, langobj):
    input_ids = []
    for word in sentence.split(" "):
        input_ids.append(langobj.word2index[word])

    if langobj.name == 'fre':  # translation-direction sensitive
        input_ids.append(langobj.word2index['EOS'])
    else:
        input_ids.insert(0, langobj.word2index['SOS'])
        input_ids.append(langobj.word2index['EOS'])
    return torch.tensor(input_ids)

# -------------------- Cell 10 --------------------
class CustomDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return length

    def __getitem__(self, idx):
        t = pairs[idx][0]  # translation-direction sensitive
        s = pairs[idx][1]  # translation-direction sensitive
        s_input_ids = torch.zeros(max_length + 1, dtype=torch.int64)
        t_input_ids = torch.zeros(max_length + 2, dtype=torch.int64)
        s_input_ids[:len(s.split(" ")) + 1] = get_input_ids(s, fre)  # translation-direction sensitive
        t_input_ids[:len(t.split(" ")) + 2] = get_input_ids(t, eng)  # translation-direction sensitive

        return s_input_ids, t_input_ids, len(s.split(" ")) + 1, len(t.split(" ")) + 1

# -------------------- Cell 11 --------------------
def train_one_epoch():
    encoder.train()
    decoder.train()
    track_loss = 0

    for i, (s_ids, t_ids, s_l, t_l) in enumerate(train_dataloader):
        s_ids = s_ids.to(device)
        t_ids = t_ids.to(device)
        encoder_outputs, encoder_hidden = encoder(s_ids, s_l)
        decoder_hidden = encoder_hidden

        # input_ids = t_ids[:,0]
        yhats = []

        for j in range(0, max_length + 1):
            context, att_weights = ba(decoder_hidden, encoder_outputs)
            probs, decoder_hidden = decoder(t_ids[:, j].unsqueeze(1), context, decoder_hidden, [1] * t_ids.shape[0])
            yhats.append(probs)

        yhats_cat = torch.cat(yhats, dim=1)
        yhats_reshaped = yhats_cat.view(-1, yhats_cat.shape[-1])

        gt = t_ids[:, 1:]
        gt = gt.reshape(-1)

        loss = loss_fn(yhats_reshaped, gt)
        track_loss += loss.item()

        opte.zero_grad()
        optd.zero_grad()
        optba.zero_grad()

        loss.backward()

        # clip_grad_norm_(encoder.parameters(), 1.0)
        # clip_grad_norm_(decoder.parameters(), 1.0)

        opte.step()
        optd.step()
        optba.step()

    return track_loss / len(train_dataloader)

# -------------------- Cell 12 --------------------
def ids2Sentence(ids, vocab):
    sentence = ""
    for id in ids.squeeze():
        if id == 0:
            continue
        word = vocab.index2word[id.item()]
        sentence += word + " "
        if id == 1:
            break
    return sentence

# -------------------- Cell 13 --------------------
# eval loop (written assuming batch_size=1)
def eval_one_epoch(e, n_epochs):
    encoder.eval()
    decoder.eval()
    track_loss = 0
    with torch.no_grad():
        for i, (s_ids, t_ids, s_l, t_l) in enumerate(test_dataloader):
            s_ids = s_ids.to(device)
            t_ids = t_ids.to(device)
            encoder_outputs, encoder_hidden = encoder(s_ids, s_l)
            decoder_hidden = encoder_hidden  # n_dim=3
            input_ids = t_ids[:, 0]
            yhats = []
            if e + 1 == n_epochs:
                pred_sentence = ""
            for j in range(1, max_length + 2):  # j starts from 1
                context, att_weights = ba(decoder_hidden, encoder_outputs)
                probs, decoder_hidden = decoder(input_ids.unsqueeze(1), context, decoder_hidden, [1])
                yhats.append(probs)
                _, input_ids = torch.topk(probs, 1, dim=-1)
                input_ids = input_ids.squeeze(1, 2)  # still a tensor
                if e + 1 == n_epochs:
                    word = eng.index2word[input_ids.item()]  # batch_size=1
                    pred_sentence += word + " "
                if input_ids.item() == 1:  # batch_size=1
                    break

            if e + 1 == n_epochs:
                src_sentence = ids2Sentence(s_ids, fre)  # translation-direction sensitive
                gt_sentence = ids2Sentence(t_ids[:, 1:], eng)  # translation-direction sensitive

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
        return track_loss / len(test_dataloader)

# -------------------- Cell 14 --------------------
# driver code

# read data
data = open("/kaggle/input/eng-fre-trans/eng-fra.txt").read().strip().split('\n')
print("Total number of pairs:", len(data))

# create pairs (create + normalize)
initpairs = createNormalizedPairs()  # list of lists. Each inner list is a pair

# filter pairs
pairs = filterPairs(initpairs)
length = len(pairs)

# create Vocab objects for each language
eng = Vocab('eng')
fre = Vocab('fre')

# build the vocab
for pair in pairs:
    eng.buildVocab(pair[0])
    fre.buildVocab(pair[1])

# print vocab size
print("English Vocab Length:", eng.nwords)
print("French Vocab Length:", fre.nwords)

dataset = CustomDataset()
train_dataset, test_dataset = random_split(dataset, [0.99, 0.01])

batch_size = 32
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

embed_size = 300
hidden_size = 1024

encoder = Encoder(fre.nwords, embed_size, hidden_size).to(device)  # translation-direction sensitive
decoder = Decoder(eng.nwords, embed_size, hidden_size).to(device)  # translation-direction sensitive

ba = Bahdanau_Attention(hidden_size, hidden_size, hidden_size).to(device)

loss_fn = nn.NLLLoss(ignore_index=0).to(device)
lr = 0.001
opte = optim.Adam(params=encoder.parameters(), lr=lr)
optd = optim.Adam(params=decoder.parameters(), lr=lr)
optba = optim.Adam(params=ba.parameters(), lr=lr)

n_epochs = 10

for e in range(n_epochs):
    print("Epoch=", e+1, sep="", end=", ")
    print("Train Loss=", train_one_epoch(), sep="", end=", ")
    print("Eval Loss=", eval_one_epoch(e, n_epochs), sep="")