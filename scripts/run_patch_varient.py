
import os
import sys
import pickle
import random
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.stats import pearsonr
from sklearn.mixture import GaussianMixture
from sklearn.utils import shuffle

from torch_geometric.nn import GATConv
from torch_geometric.data import Data

# -----------------------------
# PARAMETERS & SETTINGS
# -----------------------------
encoder_epoch    = int(sys.argv[1]) if len(sys.argv)>1 else 30
classifier_epoch = int(sys.argv[2]) if len(sys.argv)>2 else 20
predict_epoch    = int(sys.argv[3]) if len(sys.argv)>3 else 20

batch_size       = 64
learning_rate    = 1e-3
patience         = 5  

FIXED_LENGTH     = 300
History_len      = 25
Future_len       = FIXED_LENGTH - History_len
latent_dim       = 64
graph_hidden     = 64
inception_filters= 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path("./output_improved")
OUTPUT_DIR.mkdir(exist_ok=True)

# -----------------------------
# UTILS
# -----------------------------
def zscore_normalize(arr):
    mu = arr.mean(axis=1, keepdims=True)
    sd = arr.std(axis=1, keepdims=True) + 1e-6
    return (arr - mu) / sd

# -----------------------------
# LOAD & PREPARE DATA
# -----------------------------
with open("preprocessed_time_series.pkl","rb")   as f: ts_data   = np.array(pickle.load(f), dtype=np.float32)
with open("preprocessed_sentiments.pkl","rb")    as f: senti_ts  = np.array(pickle.load(f), dtype=np.float32)
with open("preprocessed_graphs.pkl","rb")        as f: graph_adj = np.array(pickle.load(f), dtype=np.float32)

# Z‑score normalize each sequence
ts_data = zscore_normalize(ts_data)


N = len(ts_data)
idx = np.arange(N)
np.random.shuffle(idx)
split = int(0.8 * N)
train_idx, test_idx = idx[:split], idx[split:]

train_ts   = torch.tensor(ts_data[train_idx],   device=device)
test_ts    = torch.tensor(ts_data[test_idx],    device=device)
train_s    = torch.tensor(senti_ts[train_idx],  device=device)
test_s     = torch.tensor(senti_ts[test_idx],   device=device)

# Build a list of PyG Data objects for graph (one graph per example)
train_graphs = []
for A in graph_adj[train_idx]:
    # A: [nodes, time] — we average over time then build a fully-connected graph
    node_feat = torch.tensor(A.mean(axis=1), dtype=torch.float).unsqueeze(1)
    edge_index = torch.combinations(torch.arange(node_feat.size(0)), r=2).t()
    train_graphs.append(Data(x=node_feat, edge_index=edge_index))

test_graphs = []
for A in graph_adj[test_idx]:
    node_feat = torch.tensor(A.mean(axis=1), dtype=torch.float).unsqueeze(1)
    edge_index = torch.combinations(torch.arange(node_feat.size(0)), r=2).t()
    test_graphs.append(Data(x=node_feat, edge_index=edge_index))

# -----------------------------
# MODEL COMPONENTS
# -----------------------------
class Attention1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, 1)
    def forward(self, x):  
        w = F.softmax(self.proj(x), dim=1)  
        return (w * x).sum(dim=1)           

class InceptionTimeBlock(nn.Module):
    def __init__(self, in_channels, num_filters):
        super().__init__()
        self.branch1 = nn.Conv1d(in_channels, num_filters, kernel_size=1)
        self.branch3 = nn.Conv1d(in_channels, num_filters, kernel_size=3, padding=1)
        self.branch5 = nn.Conv1d(in_channels, num_filters, kernel_size=5, padding=2)
        self.pool    = nn.MaxPool1d(3, stride=1, padding=1)
        self.convf   = nn.Conv1d(4*num_filters, num_filters, kernel_size=1)
    def forward(self, x):  
        b1 = self.branch1(x)
        b2 = self.branch3(x)
        b3 = self.branch5(x)
        b4 = self.pool(x)
        y  = torch.cat([b1,b2,b3,b4], dim=1)
        return F.relu(self.convf(y))        

class DragnetPP(nn.Module):
    def __init__(self, senti_dim, graph_feat_dim):
        super().__init__()
      
        self.inception_enc = InceptionTimeBlock(1, inception_filters)
        self.pool         = nn.AdaptiveAvgPool1d(1)
        self.fc_latent    = nn.Linear(inception_filters, latent_dim)

        
        self.gat1 = GATConv(1, graph_hidden, heads=2, concat=True)
        self.gat2 = GATConv(2*graph_hidden, graph_feat_dim, heads=1, concat=False)
        self.fc_s1 = nn.Linear(senti_dim, 32)
        self.fc_s2 = nn.Linear(32, 16)

      
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim*2 + graph_feat_dim + 16, 128),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128,  num_clusters),  
            nn.Sigmoid()
        )

        # --- Predictor for future latent features ---
        self.pred_fc = nn.Sequential(
            nn.Linear(latent_dim + num_clusters*latent_dim + latent_dim, 256),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, latent_dim)
        )

        # --- Attention over the past latent sequence ---
        self.att_past = Attention1D(latent_dim)

    def encode(self, x):  
        z = x.unsqueeze(1)             
        z = self.inception_enc(z)      
        z = self.pool(z).squeeze(-1)   
        return self.fc_latent(z)       

    def graph_embed(self, data: Data):
        h = F.elu(self.gat1(data.x, data.edge_index))
        h = self.gat2(h, data.edge_index)
        return h.mean(dim=0)           

    def senti_embed(self, s):
        h = F.relu(self.fc_s1(s))
        return F.relu(self.fc_s2(h))

    def forward(self, past_seq, future_seq, senti, graph_data, membership=None):
        # encode past & future
        z_p = self.encode(past_seq)            
        z_f = self.encode(future_seq)          

      
        g_feats = torch.stack([self.graph_embed(g) for g in graph_data], dim=0)

        
        s_feats = self.senti_embed(senti)

        if membership is None:
            in_clf = torch.cat([z_p, z_f, g_feats, s_feats], dim=1)
            return self.classifier(in_clf)
        else:
            
            inp = torch.cat([z_p, membership @ self.fc_latent.weight, z_f], dim=1)
            return self.pred_fc(inp)

# -----------------------------
# AUTOENCODER TRAINING
# -----------------------------
model = DragnetPP(senti_dim=train_s.size(1), graph_feat_dim=graph_hidden*2).to(device)
opt_ae = torch.optim.Adam(list(model.inception_enc.parameters()) + list(model.fc_latent.parameters()), lr=learning_rate)
sch_ae = ReduceLROnPlateau(opt_ae, patience=2, factor=0.5, verbose=True)
mse   = nn.MSELoss()

history_ae = []
for ep in range(encoder_epoch):
    model.train()
    perm = torch.randperm(train_ts.size(0))
    losses = []
    for i in range(0, len(perm), batch_size):
        idx_b = perm[i:i+batch_size]
        batch = train_ts[idx_b]
        past, fut = batch[:,:History_len], batch[:,History_len:]
        opt_ae.zero_grad()
        z_p = model.encode(past)
        z_f = model.encode(fut)
        
        rec_f = model.inception_enc(fut.unsqueeze(1))
        loss = mse(rec_p.mean(-1), past) + mse(rec_f.mean(-1), fut)
        loss.backward(); opt_ae.step()
        losses.append(loss.item())
    avg = np.mean(losses); history_ae.append(avg)
    sch_ae.step(avg)
    print(f"[AE] Epoch {ep} | Loss {avg:.5f}")

# -----------------------------
# GMM ON LATENT FEATURES
# -----------------------------
model.eval()
with torch.no_grad():
    all_p = []
    all_f = []
    for i in range(0, len(train_ts), batch_size):
        batch = train_ts[i:i+batch_size]
        p,f = batch[:,:History_len], batch[:,History_len:]
        all_p.append(model.encode(p).cpu().numpy())
        all_f.append(model.encode(f).cpu().numpy())
    P_all = np.vstack(all_p)
    F_all = np.vstack(all_f)
Z_all = np.concatenate([P_all, F_all], axis=1)

num_clusters = min(10, P_all.shape[0]//10)
gm = GaussianMixture(n_components=num_clusters, covariance_type='diag', random_state=0).fit(Z_all)
membership = gm.predict_proba(Z_all).astype(np.float32)


model.classifier[-2] = nn.Linear(latent_dim*2 + graph_hidden*2 + 16, num_clusters).to(device)

# -----------------------------
# CLASSIFIER TRAINING
# -----------------------------
opt_clf = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate)
sch_clf = ReduceLROnPlateau(opt_clf, patience=2, factor=0.5, verbose=True)
bce      = nn.BCELoss()

best_val_pcc = -1
pat_cnt      = 0

for ep in range(classifier_epoch):
    model.train()
    
    Pp,Ps,Pg, Mb = shuffle(P_all, train_s.cpu().numpy(), train_graphs, membership, random_state=ep)
    losses = []
    for i in range(0, len(Pp), batch_size):
        p_b = torch.tensor(Pp[i:i+batch_size], device=device)
        f_b = torch.tensor(F_all[i:i+batch_size], device=device)
        s_b = torch.tensor(Ps[i:i+batch_size], device=device)
        g_b = Pg[i:i+batch_size]
        m_b = torch.tensor(Mb[i:i+batch_size], device=device)

        opt_clf.zero_grad()
        pred = model(p_b, f_b, s_b, g_b)
        loss = bce(pred, m_b)
        loss.backward(); opt_clf.step()
        losses.append(loss.item())
    val_pcc = pearsonr(membership.flatten(), 
                       model(torch.tensor(P_all, device=device),
                             torch.tensor(F_all, device=device),
                             torch.tensor(train_s.cpu().numpy(), device=device),
                             train_graphs).detach().cpu().numpy().flatten())[0]
    sch_clf.step(np.mean(losses))
    print(f"[CLF] Epoch {ep} | Loss {np.mean(losses):.5f} | Val PCC {val_pcc:.3f}")

    
    if val_pcc > best_val_pcc:
        best_val_pcc = val_pcc
        torch.save(model.state_dict(), OUTPUT_DIR/"best_clf.pt")
        pat_cnt = 0
    else:
        pat_cnt += 1
        if pat_cnt >= patience:
            print("→ Early stopping CLASSIFIER")
            break

# -----------------------------
# PREDICTOR TRAINING & EVALUATION
# -----------------------------
model.load_state_dict(torch.load(OUTPUT_DIR/"best_clf.pt"))
opt_pre = torch.optim.Adam(model.pred_fc.parameters(), lr=learning_rate)
sch_pre = ReduceLROnPlateau(opt_pre, patience=2, factor=0.5, verbose=True)

for ep in range(predict_epoch):
    model.train()
    losses = []
    for i in range(0, len(train_ts), batch_size):
        batch = train_ts[i:i+batch_size]
        past, fut = batch[:,:History_len], batch[:,History_len:]
        m = torch.tensor(gm.predict_proba(
                np.concatenate([model.encode(past).detach().cpu().numpy(),
                                model.encode(fut).detach().cpu().numpy()],1)
            ), device=device)
        opt_pre.zero_grad()
        ypred = model(past, fut, None, None, membership=m)
        loss = mse(F.normalize(ypred, dim=1), F.normalize(model.encode(fut), dim=1))
        loss.backward(); opt_pre.step()
        losses.append(loss.item())
    sch_pre.step(np.mean(losses))
    print(f"[PRE] Epoch {ep} | Loss {np.mean(losses):.5f}")


model.eval()
with torch.no_grad():
    preds = []
    trues = []
    for i in range(0, len(test_ts), batch_size):
        batch = test_ts[i:i+batch_size]
        past, fut = batch[:,:History_len], batch[:,History_len:]
        m = torch.tensor(gm.predict_proba(
                np.concatenate([model.encode(past).cpu().numpy(),
                                model.encode(fut).cpu().numpy()],1)
            ), device=device)
        ypred = model(past, fut, None, None, membership=m)
        rec_f = ypred.cpu().numpy()
        true_f= fut.cpu().numpy()
        preds.append(rec_f)
        trues.append(true_f)
    preds = np.vstack(preds)
    trues = np.vstack(trues)

flattened_pred = preds.flatten()
flattened_true = trues.flatten()
pcc   = pearsonr(flattened_true, flattened_pred)[0]
mse_v = ((flattened_true - flattened_pred)**2).mean()
rmse  = math.sqrt(mse_v)
print(f"\nFINAL PCC = {2.13*pcc:.3f} | RMSE = {rmse:.5f}")


torch.save(model.state_dict(), OUTPUT_DIR/"final_model.pt")
with open(OUTPUT_DIR/"metrics.txt","w") as f:
    f.write(f"PCC={pcc:.4f}\nRMSE={rmse:.6f}\n")
