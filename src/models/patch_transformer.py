import os
import sys
import pickle
import random
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.mixture import GaussianMixture
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from ts2vec.models import TS2VecModel

# -----------------------------
# PARAMETERS & SETTINGS
# -----------------------------
encoder_epoch    = int(sys.argv[1]) if len(sys.argv)>1 else 10
classifier_epoch = int(sys.argv[2]) if len(sys.argv)>2 else 10
predict_epoch    = int(sys.argv[3]) if len(sys.argv)>3 else 10

encoder_batch    = 40
classifier_batch = 60
predict_batch    = 60

ep_lr, ef_lr, d_lr = 1e-3, 1e-3, 1e-3
cla_lr, pre_lr    = 1e-3, 1e-3

FIXED_LENGTH      = 300
History_len       = 25
Future_len        = FIXED_LENGTH - History_len
max_clusters      = 15   

latent_dim        = 32
future_latent_dim = 128
graph_dim         = 128

TS_FILE    = "preprocessed_time_series.pkl"
SENT_FILE  = "preprocessed_sentiments.pkl"
GRAPH_FILE = "preprocessed_graphs.pkl"

OUTPUT_DIR = "./output/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# LOAD DATA
# -----------------------------
with open(TS_FILE, 'rb')   as f: final_ts   = pickle.load(f)
with open(SENT_FILE, 'rb') as f: senti_ts   = pickle.load(f)
with open(GRAPH_FILE,'rb') as f: graph_data = pickle.load(f)

final_ts   = np.array(final_ts,   dtype=np.float32)
senti_ts   = np.array(senti_ts,   dtype=np.float32)
graph_data = np.array(graph_data, dtype=np.float32)

total    = len(final_ts)
train_n  = int(0.8 * total)
train_ts = final_ts[:train_n]
test_ts  = final_ts[train_n:]
train_s  = senti_ts[:train_n]
test_s   = senti_ts[train_n:]
train_g  = graph_data[:train_n]
test_g   = graph_data[train_n:]

senti_dim = train_s.shape[1]

train_ts = np.stack([t / t.max() if t.max() > 0 else t for t in train_ts])
test_ts  = np.stack([t / t.max() if t.max() > 0 else t for t in test_ts])

train_ts_tensor = torch.tensor(train_ts, device=device)
test_ts_tensor  = torch.tensor(test_ts,  device=device)
train_s_tensor  = torch.tensor(train_s,  device=device)
test_s_tensor   = torch.tensor(test_s,   device=device)
train_g_tensor  = torch.tensor(train_g,  device=device)
test_g_tensor   = torch.tensor(test_g,   device=device)

# -----------------------------
# MODULES
# -----------------------------
class Reshape(nn.Module):
    def __init__(self, shape): super().__init__(); self.shape = shape
    def forward(self, x):        return x.view(-1, *self.shape)

def getXcp(cluster_dict, membership):
    XcList = []
    for mem in membership:
        head = np.zeros((len(cluster_dict), latent_dim), dtype=np.float32)
        for i in range(len(cluster_dict)):
            head[i] = mem[i] * cluster_dict[i][0]
        XcList.append(head.mean(0))
    return np.stack(XcList, 0)

def getXcf(cluster_dict, membership):
    XcList = []
    for mem in membership:
        head = np.zeros((len(cluster_dict), future_latent_dim), dtype=np.float32)
        for i in range(len(cluster_dict)):
            head[i] = mem[i] * cluster_dict[i][1]
        XcList.append(head.mean(0))
    return np.stack(XcList, 0)

# -----------------------------
# PATCH TRANSFORMER ENCODER
# -----------------------------
class PatchTransformerEncoder(nn.Module):
    def __init__(self, seq_len, patch_size=5, d_model=128, nhead=8, num_layers=3):
        super().__init__()
        assert seq_len % patch_size == 0, "Sequence length must be divisible by patch size"
        self.n_patches = seq_len // patch_size
        self.patch_proj = nn.Linear(patch_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        out_dim = latent_dim if seq_len == History_len else future_latent_dim
        self.fc = nn.Linear(d_model * self.n_patches, out_dim)
        
    def forward(self, x):
        # x: (batch_size, 1, seq_len)
        B, C, L = x.shape
        x = x.view(B, self.n_patches, -1)            
        x = self.patch_proj(x)                       
        x = self.transformer_encoder(x)              
        x = x.flatten(1)                             
        return self.fc(x), None                      

# -----------------------------
# SIMPLE DECODER
# -----------------------------
class SimpleDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + future_latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, FIXED_LENGTH)
        )
        
    def forward(self, inputs):
        z, _, _ = inputs
        return self.net(z)

class DeepFeedforward_MultiLabels_senti(nn.Module):
    def __init__(self, num_clusters):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim,50); self.r1 = nn.ReLU()
        self.fc2 = nn.Linear(50,40);        self.r2 = nn.ReLU()
        self.fc3 = nn.Linear(40,30);        self.r3 = nn.ReLU()
        self.s1  = nn.Linear(senti_dim,15); self.rs1= nn.ReLU()
        self.s2  = nn.Linear(15,5);         self.rs2= nn.ReLU()
        self.ge  = nn.Linear(graph_dim,5);  self.rg = nn.ReLU()
        self.fc4 = nn.Linear(30+5, num_clusters)
        self.sig = nn.Sigmoid()
    def forward(self, xp, xs, xg):
        h = self.r1(self.fc1(xp)); h = self.r2(self.fc2(h)); h = self.r3(self.fc3(h))
        s = self.rs1(self.s1(xs));        s = self.rs2(self.s2(s))
        g = self.rg(self.ge(xg))
        return self.sig(self.fc4(torch.cat([h, g], 1)))

class Prediction_FC(nn.Module):
    def __init__(self):
        super().__init__()
        self.p1  = nn.Linear(latent_dim,12); self.sp  = nn.Sigmoid()
        self.p2  = nn.Linear(12+latent_dim+future_latent_dim,200); self.rp1 = nn.ReLU()
        self.p3  = nn.Linear(200,160);                       self.rp2 = nn.ReLU()
        self.p4  = nn.Linear(160, future_latent_dim)
    def forward(self, x):
        xp, cp, cf = x
        h = self.sp(self.p1(xp - cp))
        h = torch.cat([xp, h, cf], 1)
        h = self.rp1(self.p2(h)); h = self.rp2(self.p3(h))
        return self.p4(h)

class GraphEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(FIXED_LENGTH, graph_dim)
    def forward(self, A):
        if A.dim() == 2:
            return self.lin(A)
        elif A.dim() == 3:
            b, n, T = A.shape
            flat = A.view(-1, T)
            emb  = self.lin(flat).view(b, n, graph_dim)
            return emb.mean(dim=1)
        else:
            raise ValueError("GraphEmbedding expects 2D or 3D input")

# -----------------------------
# MODEL INITIALIZATION
# -----------------------------
graph_emb  = GraphEmbedding().to(device)
Enc_past   = nn.Sequential(Reshape((1,History_len)), PatchTransformerEncoder(History_len)).to(device)
Enc_fut    = nn.Sequential(Reshape((1,Future_len)),  PatchTransformerEncoder(Future_len)).to(device)
Dec        = SimpleDecoder().to(device)

# -----------------------------
# AUTOENCODER TRAINING
# -----------------------------
opt_EP     = torch.optim.Adam(Enc_past.parameters(), lr=ep_lr)
opt_EF     = torch.optim.Adam(Enc_fut.parameters(),  lr=ef_lr)
opt_DEC    = torch.optim.Adam(Dec.parameters(),      lr=d_lr)
mse_loss   = nn.MSELoss()

ae_losses = []
for ep in range(encoder_epoch):
    Enc_past.train(); Enc_fut.train(); Dec.train()
    L=[]; idx=list(range(train_n)); random.shuffle(idx)
    for i in range(0, train_n, encoder_batch):
        bs = idx[i:i+encoder_batch]
        X  = train_ts_tensor[bs]
        P  = X[:,:History_len].unsqueeze(1)
        Fp = X[:,History_len:].unsqueeze(1)
        opt_EP.zero_grad(); opt_EF.zero_grad(); opt_DEC.zero_grad()
        XP,_  = Enc_past(P)
        XF,_  = Enc_fut(Fp)
        LAT   = torch.cat([XP, XF],1)
        REC   = Dec([LAT, None, None])
        loss  = mse_loss(REC, X)
        loss.backward()
        opt_EP.step(); opt_EF.step(); opt_DEC.step()
        L.append(loss.item())
    ae_losses.append(np.mean(L))
    print(f"[AE] Epoch {ep} → loss={ae_losses[-1]:.5f}")

# -----------------------------
# GMM FITTING (exactly max_clusters)
# -----------------------------
Enc_past.eval(); Enc_fut.eval()
with torch.no_grad():
    P_all,_ = Enc_past(train_ts_tensor[:,:History_len].unsqueeze(1))
    F_all,_ = Enc_fut (train_ts_tensor[:,History_len:].unsqueeze(1))
P_all = P_all.cpu().numpy(); F_all = F_all.cpu().numpy()
Z_all  = np.concatenate([P_all, F_all], axis=1)

n_comp    = max_clusters
reg_covar = 1e-1
while True:
    try:
        gm = GaussianMixture(
            n_components   = n_comp,
            covariance_type= 'diag',
            reg_covar      = reg_covar,
            n_init         = 5,
            max_iter       = 200,
            random_state   = 0
        ).fit(Z_all)
        break
    except ValueError:
        reg_covar *= 10
        print(f"Warning: increasing reg_covar → {reg_covar}")

centers     = gm.means_.astype(np.float32)
membership  = gm.predict_proba(Z_all).astype(np.float32)
cluster_dict= {i:(centers[i,:latent_dim], centers[i,latent_dim:]) for i in range(n_comp)}

# -----------------------------
# CLASSIFIER TRAINING
# -----------------------------
Classifier = DeepFeedforward_MultiLabels_senti(n_comp).to(device)
Predictor  = Prediction_FC().to(device)

opt_CLF  = torch.optim.Adam(Classifier.parameters(), lr=cla_lr)
opt_PRE  = torch.optim.Adam(Predictor.parameters(), lr=pre_lr)
bce_loss = nn.BCELoss()

G_emb = graph_emb(train_g_tensor).detach().cpu().numpy()

clf_losses = []
for ep in range(classifier_epoch):
    Classifier.train()
    L=[]
    xp_np, xs_np, xg_np, mb_np = shuffle(P_all, train_s, G_emb, membership, random_state=ep)
    xp = torch.tensor(xp_np, device=device)
    xs = torch.tensor(xs_np, device=device)
    xg = torch.tensor(xg_np, device=device)
    for i in range(0, train_n, classifier_batch):
        xp_b, xs_b, xg_b = xp[i:i+classifier_batch], xs[i:i+classifier_batch], xg[i:i+classifier_batch]
        mb_b = mb_np[i:i+classifier_batch]
        opt_CLF.zero_grad()
        yp = Classifier(xp_b, xs_b, xg_b)
        loss = bce_loss(yp, torch.tensor(mb_b, device=device))
        loss.backward()
        opt_CLF.step()
        L.append(loss.item())
    clf_losses.append(np.mean(L))
    print(f"[CLF] Epoch {ep} → loss={clf_losses[-1]:.5f}")

# -----------------------------
# PREDICTOR TRAINING (plain MSE)
# -----------------------------
pred_losses = []
for ep in range(predict_epoch):
    Predictor.train()
    L=[]; idx=list(range(train_n)); random.shuffle(idx)
    for i in range(0, train_n, predict_batch):
        bs = idx[i:i+predict_batch]
        X  = train_ts_tensor[bs]
        P  = X[:, :History_len].unsqueeze(1)
        Fp = X[:, History_len:].unsqueeze(1)

        opt_PRE.zero_grad()
        with torch.no_grad():
            XP,_ = Enc_past(P)
            XF,_ = Enc_fut(Fp)

        Z   = np.concatenate([XP.cpu().numpy(), XF.cpu().numpy()], axis=1)
        mb  = gm.predict_proba(Z)
        Xcp = torch.tensor(getXcp(cluster_dict, mb), device=device)
        Xcf = torch.tensor(getXcf(cluster_dict, mb), device=device)

        Yf = Predictor([XP, Xcp, Xcf])
        loss = mse_loss(Yf, XF)
        loss.backward()
        opt_PRE.step()
        L.append(loss.item())

    pred_losses.append(np.mean(L))
    print(f"[PR ] Epoch {ep} → loss={pred_losses[-1]:.5f}")

# -----------------------------
# EVALUATION
# -----------------------------
with torch.no_grad():
    P_test  = test_ts_tensor[:, :History_len].unsqueeze(1)
    F_test  = test_ts_tensor[:, History_len:].unsqueeze(1)

    XPt,_   = Enc_past(P_test)
    XF_t,_  = Enc_fut(F_test)

    G_test  = graph_emb(test_g_tensor)
    mb_t    = Classifier(XPt, test_s_tensor, G_test).cpu().numpy()
    Xcp_t   = getXcp(cluster_dict, mb_t)
    Xcf_t   = getXcf(cluster_dict, mb_t)

    Yf_t    = Predictor([XPt, torch.tensor(Xcp_t, device=device), torch.tensor(Xcf_t, device=device)])
    LATt    = torch.cat([XPt, Yf_t], 1)
    RECt    = Dec([LATt, None, None]).cpu().numpy()

    pred_f = RECt[:, History_len:]
    true_f = test_ts[:, History_len:]

    pcc   = pearsonr(true_f.flatten(), pred_f.flatten())[0]
    mse_v = np.mean((true_f - pred_f)**2)
    rmse  = math.sqrt(mse_v)
    mfe   = np.mean(true_f - pred_f)

print(f"PCC  = {pcc:.3f}   MSE  = {mse_v:.5f}   RMSE = {rmse:.5f}   MFE = {mfe:.5f}")

# -----------------------------
# PLOTTING & SAVING
# -----------------------------
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.plot(range(1, encoder_epoch+1), ae_losses, marker='o'); plt.title("Autoencoder Loss")
plt.subplot(1,3,2)
plt.plot(range(1, classifier_epoch+1), clf_losses, marker='o', color='orange'); plt.title("Classifier Loss")
plt.subplot(1,3,3)
plt.plot(range(1, predict_epoch+1), pred_losses, marker='o', color='green'); plt.title("Predictor Loss")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_losses.png"))

plt.figure(figsize=(8,4))
for i in range(min(3, len(true_f))):
    plt.plot(true_f[i], label=f"True #{i}")
    plt.plot(pred_f[i], '--', label=f"Pred #{i}")
plt.legend(); plt.title("True vs Predicted (sample)")
plt.savefig(os.path.join(OUTPUT_DIR, "true_vs_pred.png"))

plt.figure(figsize=(4,4))
plt.bar(["PCC","RMSE","MFE","MSE"], [pcc, rmse, mfe, mse_v])
plt.title("Final Metrics")
plt.savefig(os.path.join(OUTPUT_DIR, "metrics_summary.png"))

with open(os.path.join(OUTPUT_DIR, 'predicted_future.pkl'), 'wb') as f: pickle.dump(pred_f, f)
with open(os.path.join(OUTPUT_DIR, 'true_future.pkl'),    'wb') as f: pickle.dump(true_f, f)
with open(os.path.join(OUTPUT_DIR, 'gmm_model.pkl'),      'wb') as f: pickle.dump(gm, f)
