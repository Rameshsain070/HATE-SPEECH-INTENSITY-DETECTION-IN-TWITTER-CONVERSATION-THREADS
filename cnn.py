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
with open(TS_FILE,  'rb') as f: final_ts   = pickle.load(f)
with open(SENT_FILE,'rb') as f: senti_ts   = pickle.load(f)
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

train_ts = np.stack([t / t.max() if t.max()>0 else t for t in train_ts])
test_ts  = np.stack([t / t.max() if t.max()>0 else t for t in test_ts])

train_ts_tensor = torch.tensor(train_ts, device=device)
test_ts_tensor  = torch.tensor(test_ts,  device=device)
train_s_tensor  = torch.tensor(train_s,  device=device)
test_s_tensor   = torch.tensor(test_s,   device=device)
train_g_tensor  = torch.tensor(train_g,  device=device)
test_g_tensor   = torch.tensor(test_g,   device=device)

# -----------------------------
# UTILITIES
# -----------------------------
class Reshape(nn.Module):
    def __init__(self, shape): super().__init__(); self.shape = shape
    def forward(self, x):        return x.view(-1, *self.shape)

def getXcp(cluster_dict, membership):
    arr = np.stack([
        np.stack([mem[i]*cluster_dict[i][0] for i in range(len(cluster_dict))]).mean(0)
        for mem in membership
    ],0)
    return arr.astype(np.float32)

def getXcf(cluster_dict, membership):
    arr = np.stack([
        np.stack([mem[i]*cluster_dict[i][1] for i in range(len(cluster_dict))]).mean(0)
        for mem in membership
    ],0)
    return arr.astype(np.float32)

# -----------------------------
# TIME-SERIES VAE
# -----------------------------
class TimeSeriesVAE(nn.Module):
    def __init__(self, seq_len, in_ch, hidden_dims, z_dim):
        super().__init__()
        self.seq_len = seq_len
        # encoder
        layers, c = [], in_ch
        for h in hidden_dims:
            layers += [nn.Conv1d(c,h,3,padding=1),
                       nn.BatchNorm1d(h), nn.ReLU(),
                       nn.MaxPool1d(2)]
            c = h
        self.encoder = nn.Sequential(*layers)
        red = seq_len//(2**len(hidden_dims))
        enc_out = c*red
        self.fc_mu     = nn.Linear(enc_out, z_dim)
        self.fc_logvar = nn.Linear(enc_out, z_dim)
        # decoder
        self.dec_input = nn.Linear(z_dim, enc_out)
        dec_layers = []
        rev = hidden_dims[::-1]
        for i,h in enumerate(rev):
            outc = in_ch if i==len(rev)-1 else rev[i+1]
            dec_layers += [nn.ConvTranspose1d(h,outc,4,2,1),
                           nn.BatchNorm1d(outc),
                           nn.ReLU() if i<len(rev)-1 else nn.Identity()]
        self.decoder = nn.Sequential(*dec_layers)
    def reparam(self, mu, lv):
        std = torch.exp(0.5*lv)
        return mu + torch.randn_like(std)*std
    def forward(self, x):
        B = x.size(0)
        h = self.encoder(x).view(B,-1)
        mu,lv = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparam(mu,lv)
        d = self.dec_input(z).view(B,-1,self.seq_len//(2**len(self.encoder)))
        recon = self.decoder(d)
        Lr = recon.size(-1)
        if Lr<self.seq_len:
            recon = torch.cat([recon,
                x.new_zeros((B,1,self.seq_len-Lr))],-1)
        return z,mu,lv,recon
    def loss(self, recon, x, mu, lv):
        return F.mse_loss(recon,x) + \
            (-0.5*torch.mean(1+lv-mu.pow(2)-lv.exp()))

# -----------------------------
# DECODER / PRIOR / PREDICTOR / GRAPH
# -----------------------------
class SimpleDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim+future_latent_dim,512), nn.ReLU(),
            nn.Linear(512,1024), nn.ReLU(),
            nn.Linear(1024,FIXED_LENGTH)
        )
    def forward(self, tup):
        z,_,_,_ = tup
        return self.net(z)

class PriorTransformer(nn.Module):
    def __init__(self, dxp, dxs, dxg, nc,
                 dm=128, nh=4, nl=2):
        super().__init__()
        self.lxp = nn.Linear(dxp, dm)
        self.lxs = nn.Linear(dxs, dm)
        self.lxg = nn.Linear(dxg, dm)
        enc = nn.TransformerEncoderLayer(dm, nh,
             dim_feedforward=2*dm, batch_first=True)
        self.tr = nn.TransformerEncoder(enc, nl)
        self.fc = nn.Linear(dm,nc)
    def forward(self,xp,xs,xg):
        t1 = self.lxp(xp).unsqueeze(1)
        t2 = self.lxs(xs).unsqueeze(1)
        t3 = self.lxg(xg).unsqueeze(1)
        s  = torch.cat([t1,t2,t3],1)
        h  = self.tr(s).mean(1)
        return torch.sigmoid(self.fc(h))

class Prediction_FC(nn.Module):
    def __init__(self):
        super().__init__()
        self.p1 = nn.Linear(latent_dim,12); self.sp = nn.Sigmoid()
        self.p2 = nn.Linear(12+latent_dim+future_latent_dim,200); self.rp1=nn.ReLU()
        self.p3 = nn.Linear(200,160); self.rp2=nn.ReLU()
        self.p4 = nn.Linear(160,future_latent_dim)
    def forward(self,x):
        xp,cp,cf = x
        h = self.sp(self.p1(xp-cp))
        h = torch.cat([xp,h,cf],1)
        h = self.rp1(self.p2(h)); h=self.rp2(self.p3(h))
        return self.p4(h)

class GraphEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(FIXED_LENGTH, graph_dim)
    def forward(self,A):
        if A.dim()==2:
            return self.lin(A)
        b,n,T = A.shape
        flat = A.view(-1,T)
        emb  = self.lin(flat).view(b,n,graph_dim)
        return emb.mean(1)

# -----------------------------
# INSTANTIATE & OPTS
# -----------------------------
history_vae = TimeSeriesVAE(History_len,1,(32,64),latent_dim).to(device)
future_vae  = TimeSeriesVAE(Future_len,1,(32,64),future_latent_dim).to(device)
Dec         = SimpleDecoder().to(device)
Classifier  = PriorTransformer(latent_dim, senti_dim, graph_dim, max_clusters).to(device)
Predictor   = Prediction_FC().to(device)
graph_emb   = GraphEmbedding().to(device)

opt_h = torch.optim.Adam(history_vae.parameters(), lr=ep_lr)
opt_f = torch.optim.Adam(future_vae.parameters(),  lr=ef_lr)
opt_d = torch.optim.Adam(Dec.parameters(),         lr=d_lr)
opt_c = torch.optim.Adam(Classifier.parameters(),  lr=cla_lr)
opt_p = torch.optim.Adam(Predictor.parameters(),   lr=pre_lr)

mse = nn.MSELoss()
bce = nn.BCELoss()

# -----------------------------
# 1) TRAIN VAEs
# -----------------------------
for ep in range(encoder_epoch):
    history_vae.train(); future_vae.train()
    tot=0; idx=torch.randperm(train_n)
    for i in range(0,train_n,encoder_batch):
        bs = idx[i:i+encoder_batch]
        X  = train_ts_tensor[bs]
        hX = X[:,:History_len].unsqueeze(1)
        fX = X[:,History_len:].unsqueeze(1)
        opt_h.zero_grad(); opt_f.zero_grad()
        zh,mu_h,lv_h,rec_h = history_vae(hX)
        zf,mu_f,lv_f,rec_f = future_vae(fX)
        l_h = history_vae.loss(rec_h,hX,mu_h,lv_h)
        l_f = future_vae.loss(rec_f,fX,mu_f,lv_f)
        loss = l_h + l_f
        loss.backward()
        opt_h.step(); opt_f.step()
        tot += loss.item()
    print(f"[VAE] Epoch {ep} loss={tot/(train_n/encoder_batch):.5f}")

# -----------------------------
# 2) LATENTS & GMM
# -----------------------------
history_vae.eval(); future_vae.eval()
with torch.no_grad():
    H_all=[]; F_all=[]
    for i in range(0,train_n,encoder_batch):
        bs=list(range(i,min(i+encoder_batch,train_n)))
        hX = train_ts_tensor[bs,:History_len].unsqueeze(1)
        fX = train_ts_tensor[bs,History_len:].unsqueeze(1)
        zh,_,_,_ = history_vae(hX)
        zf,_,_,_ = future_vae(fX)
        H_all.append(zh.cpu().numpy())
        F_all.append(zf.cpu().numpy())
    H_all = np.concatenate(H_all,0)
    F_all = np.concatenate(F_all,0)
Z_all = np.concatenate([H_all,F_all],1)

reg = 1e-1
while True:
    try:
        gm = GaussianMixture(
            n_components=max_clusters,
            covariance_type='diag',
            reg_covar=reg,
            n_init=5,
            max_iter=200,
            random_state=0
        ).fit(Z_all)
        break
    except ValueError:
        reg *= 10

centers      = gm.means_.astype(np.float32)
membership   = gm.predict_proba(Z_all).astype(np.float32)
cluster_dict = {i:(centers[i,:latent_dim],centers[i,latent_dim:])
                for i in range(max_clusters)}

# -----------------------------
# 3) PRIOR TRAIN
# -----------------------------
G_emb = graph_emb(train_g_tensor).detach().cpu().numpy()
for ep in range(classifier_epoch):
    Classifier.train(); losses=[]
    xp,xs,xg,mb0 = shuffle(H_all, train_s, G_emb,
                           membership, random_state=ep)
    xp = torch.tensor(xp, device=device)
    xs = torch.tensor(xs, device=device)
    xg = torch.tensor(xg, device=device)
    for i in range(0,train_n,classifier_batch):
        xb = xp[i:i+classifier_batch]
        sb = xs[i:i+classifier_batch]
        gb = xg[i:i+classifier_batch]
        mbatch = mb0[i:i+classifier_batch]
        opt_c.zero_grad()
        y = Classifier(xb,sb,gb)
        loss = bce(y, torch.tensor(mbatch,
                     dtype=torch.float32,
                     device=device))
        loss.backward(); opt_c.step()
        losses.append(loss.item())
    print(f"[CLF] Epoch {ep} loss={np.mean(losses):.5f}")

# -----------------------------
# 4) PREDICTOR TRAIN
# -----------------------------
for ep in range(predict_epoch):
    Predictor.train(); losses=[]; idx=list(range(train_n)); random.shuffle(idx)
    for i in range(0,train_n,predict_batch):
        bs = idx[i:i+predict_batch]
        X  = train_ts_tensor[bs]
        hX = X[:,:History_len].unsqueeze(1)
        fX = X[:,History_len:].unsqueeze(1)
        opt_p.zero_grad()
        with torch.no_grad():
            zh,_,_,_ = history_vae(hX)
            zf,_,_,_ = future_vae(fX)
        Zt = np.concatenate([zh.cpu().numpy(),
                             zf.cpu().numpy()],1)
        mbp = gm.predict_proba(Zt)
        Xcp = torch.tensor(getXcp(cluster_dict,mbp),
                           dtype=torch.float32,
                           device=device)
        Xcf = torch.tensor(getXcf(cluster_dict,mbp),
                           dtype=torch.float32,
                           device=device)
        Yf  = Predictor([zh, Xcp, Xcf])
        loss = mse(Yf, zf)
        loss.backward(); opt_p.step()
        losses.append(loss.item())
    print(f"[PR] Epoch {ep} loss={np.mean(losses):.5f}")

# -----------------------------
# 5) EVAL & SAVE
# -----------------------------
with torch.no_grad():
    P_test = test_ts_tensor[:,:History_len].unsqueeze(1)
    F_test = test_ts_tensor[:,History_len:].unsqueeze(1)
    zh,_,_,_ = history_vae(P_test)
    zf,_,_,_ = future_vae(F_test)
    gte      = graph_emb(test_g_tensor).detach().cpu().numpy()
    mbt      = Classifier(zh, test_s_tensor, gte).cpu().numpy()
    Xcp_t    = getXcp(cluster_dict, mbt)
    Xcf_t    = getXcf(cluster_dict, mbt)
    Ypred    = Predictor([zh,
                         torch.tensor(Xcp_t,
                                      dtype=torch.float32,
                                      device=device),
                         torch.tensor(Xcf_t,
                                      dtype=torch.float32,
                                      device=device)])
    LAT      = torch.cat([zh, Ypred],1)
    REC      = Dec([LAT,None,None]).cpu().numpy()
    pred_f   = REC[:,History_len:]
    true_f   = test_ts[:,History_len:]
    pcc      = pearsonr(true_f.flatten(), pred_f.flatten())[0]
    mse_v    = np.mean((true_f-pred_f)**2)
    rmse     = math.sqrt(mse_v)
    mfe      = np.mean(true_f-pred_f)

print(f"PCC={pcc:.3f} RMSE={rmse:.5f} MFE={mfe:.5f}")
