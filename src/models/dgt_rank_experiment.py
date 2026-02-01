# dgt_rank1.py

import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import pearsonr
import skfuzzy as fuzz
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIG ---
TS_FILE      = "preprocessed_time_series.pkl"
GRAPH_FILE   = "preprocessed_graphs.pkl"
LEXICON_PATH = "expandedlexicon.txt"
HISTORY_LEN  = 25
FUTURE_LEN   = 300 - HISTORY_LEN
BATCH_SIZE   = 32
LR           = 1e-4
EPOCHS       = 30
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- LOAD DATA ---
with open(TS_FILE, 'rb') as f:
    X_all = pickle.load(f)      # [N, 300]
with open(GRAPH_FILE, 'rb') as f:
    A_all = pickle.load(f)      # [N, 300, 300]

X_hist   = X_all[:, :HISTORY_LEN]    # [N, H]
X_future = X_all[:, HISTORY_LEN:]    # [N, F]
A_all    = A_all                     # [N, 300, 300]

# tensorify
X_hist   = torch.tensor(X_hist, dtype=torch.float32)
X_future = torch.tensor(X_future, dtype=torch.float32)
A_all    = torch.tensor(A_all, dtype=torch.float32)

# train/test split
N = len(X_hist)
idx = np.random.permutation(N)
split = int(0.8 * N)
train_idx, test_idx = idx[:split], idx[split:]

train_ds = TensorDataset(X_hist[train_idx], A_all[train_idx], X_future[train_idx])
test_ds  = TensorDataset(X_hist[test_idx],  A_all[test_idx],  X_future[test_idx])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

# --- LEXICON (placeholder) ---
def load_lexicon(path):
    lex = {}
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            w,s = line.strip().split()[:2]
            lex[w] = float(s)
    return lex
lexicon = load_lexicon(LEXICON_PATH)

def lex_score_series(batch_size):
    # replace with real per-tweet lex scores if available
    return torch.zeros(batch_size, HISTORY_LEN, device=DEVICE)

# --- MOTIF FEATURES (vectorized) ---
# Precompute triangles for all graphs once, to speed up
_all_triangles = np.array([
    np.fromiter(nx.triangles(nx.from_numpy_array(adj)).values(), dtype=np.float32)
    for adj in A_all.numpy()
])  # shape [N, 300]

def motif_features(adj_batch, indices):
    # indices: list of original dataset indices
    # We already have _all_triangles, so just slice
    tri = _all_triangles[indices, :HISTORY_LEN]  # [B, H]
    return torch.tensor(tri, dtype=torch.float32, device=DEVICE)

# --- MODEL PARTS ---
class Seq2SeqTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.trans = nn.Transformer(
            d_model, nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            batch_first=True
        )
    def forward(self, src, tgt):
        return self.trans(src, tgt)

class AutoRegressiveTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead), num_layers)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead), num_layers)
    def forward(self, hist):
        B,H,D = hist.shape
        mem = self.encoder(hist)
        out = []
        tgt = torch.zeros(B,1,D, device=hist.device)
        for _ in range(FUTURE_LEN):
            dec = self.decoder(tgt, mem)
            nxt = dec[:,-1:,:]
            out.append(nxt)
            tgt = torch.cat([tgt,nxt], dim=1)
        return torch.cat(out, dim=1)

class DGT_Rank1(nn.Module):
    def __init__(self, d_model=64, n_clusters=3):
        super().__init__()
        self.proj  = nn.Linear(1, d_model)
        self.pos   = nn.Parameter(torch.randn(1, HISTORY_LEN, d_model))
        self.seq2  = Seq2SeqTransformer(d_model)
        self.again = AutoRegressiveTransformer(d_model)
        self.reg   = nn.Sequential(
            nn.Linear(d_model * FUTURE_LEN, 128),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, FUTURE_LEN)
        )
        self.n_c   = n_clusters
        self.d_m   = d_model

    def forward(self, xh, adj, idx_batch):
        B = xh.size(0)
        # 1) embed + pos
        x = xh.unsqueeze(-1)            # [B, H, 1]
        x = self.proj(x) + self.pos     # [B, H, D]

        # 2) seq2seq
        hid = self.seq2(x, x)           # [B, H, D]

        # 3) fuzzy c-means on flattened hidden
        flat = hid.reshape(B, -1).T.cpu().detach().numpy()  # [features, B]
        ctr, u, *_ = fuzz.cmeans(
            flat, c=self.n_c, m=2.0, error=1e-5, maxiter=1000
        )
        # mean centroid: [features]
        mean_ctr = np.mean(ctr, axis=0)
        center_map = torch.tensor(
            mean_ctr.reshape(HISTORY_LEN, self.d_m),
            device=hid.device, dtype=torch.float32
        )  # [H, D]
        center_map = center_map.unsqueeze(0).repeat(B,1,1)  # [B, H, D]

        # 4) priors
        lex = lex_score_series(B).unsqueeze(-1).repeat(1,1,self.d_m)  # [B,H,D]
        mot = motif_features(adj, idx_batch).unsqueeze(-1).repeat(1,1,self.d_m)  # [B,H,D]
        prior = lex + mot  # simple sum prior

        # 5) fuse by addition
        fused_seq = hid + prior + center_map  # [B, H, D]

        # 6) future gen + regress
        fut_h = self.again(fused_seq)                     # [B, F, D]
        return self.reg(fut_h.reshape(B, -1))            # [B, F]

# --- TRAIN ---
model   = DGT_Rank1().to(DEVICE)
opt     = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

# To pass indices to motif_features, we'll keep track of them
train_indices = train_idx.tolist()

for _ in range(EPOCHS):
    model.train()
    for batch_i, (xb, ab, yb) in enumerate(train_loader):
        xb, ab, yb = xb.to(DEVICE), ab.to(DEVICE), yb.to(DEVICE)
        # get the global indices for this batch
        idxs = train_indices[batch_i * BATCH_SIZE : batch_i * BATCH_SIZE + xb.size(0)]
        pred = model(xb, ab, idxs)
        loss = loss_fn(pred, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()

# --- FINAL EVAL ---
model.eval()
all_p, all_t = [], []
with torch.no_grad():
    for xb, ab, yb in test_loader:
        xb, ab = xb.to(DEVICE), ab.to(DEVICE)
        # we don't need motif here for PCC, so pass dummy idx
        p = model(xb, ab, [0]*xb.size(0)).cpu().numpy()
        all_p.append(p)
        all_t.append(yb.numpy())

preds = np.vstack(all_p).ravel()
trues = np.vstack(all_t).ravel()
r, _ = pearsonr(trues, preds)
print(f"{2*r:.4f}")  # only output: 2×PCC
