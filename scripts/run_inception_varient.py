
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
encoder_epoch    = int(sys.argv[1]) if len(sys.argv)>1 else 20
classifier_epoch = int(sys.argv[2]) if len(sys.argv)>2 else 20
predict_epoch    = int(sys.argv[3]) if len(sys.argv)>3 else 20

encoder_batch    = 64
classifier_batch = 64
predict_batch    = 64

ep_lr, ef_lr, d_lr = 1e-3, 1e-3, 1e-3
cla_lr, pre_lr     = 1e-3, 1e-3
weight_decay       = 1e-5

FIXED_LENGTH      = 300
History_len       = 25
Future_len        = FIXED_LENGTH - History_len
max_clusters      = 20

latent_dim        = 64
future_latent_dim = 256
graph_dim         = 256

TS_FILE    = "preprocessed_time_series.pkl"
SENT_FILE  = "preprocessed_sentiments.pkl"
GRAPH_FILE = "preprocessed_graphs.pkl"
OUTPUT_DIR = "./output/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# DATA LOADING
# -----------------------------
with open(TS_FILE,   'rb') as f: final_ts   = pickle.load(f)
with open(SENT_FILE, 'rb') as f: senti_ts   = pickle.load(f)
with open(GRAPH_FILE,'rb') as f: graph_data = pickle.load(f)

final_ts   = np.array(final_ts,   dtype=np.float32)
senti_ts   = np.array(senti_ts,   dtype=np.float32)
graph_data = np.array(graph_data, dtype=np.float32)

total    = len(final_ts)
train_n  = int(0.8 * total)
val_n    = int(0.1 * train_n)
train_n -= val_n

train_ts = final_ts[:train_n]
val_ts   = final_ts[train_n:train_n+val_n]
test_ts  = final_ts[train_n+val_n:]

train_s = senti_ts[:train_n]
val_s   = senti_ts[train_n:train_n+val_n]
test_s  = senti_ts[train_n+val_n:]

train_g = graph_data[:train_n]
val_g   = graph_data[train_n:train_n+val_n]
test_g  = graph_data[train_n+val_n:]

def norm_stack(arr):
    return np.stack([t / t.max() if t.max()>0 else t for t in arr])

train_ts = norm_stack(train_ts)
val_ts   = norm_stack(val_ts)
test_ts  = norm_stack(test_ts)

train_ts_tensor = torch.tensor(train_ts, device=device)
val_ts_tensor   = torch.tensor(val_ts,   device=device)
test_ts_tensor  = torch.tensor(test_ts,  device=device)

train_s_tensor = torch.tensor(train_s, device=device)
val_s_tensor   = torch.tensor(val_s,   device=device)
test_s_tensor  = torch.tensor(test_s,  device=device)

train_g_tensor = torch.tensor(train_g, device=device)
val_g_tensor   = torch.tensor(val_g,   device=device)
test_g_tensor  = torch.tensor(test_g,  device=device)

senti_dim = train_s.shape[1]

# -----------------------------
# UTILITIES: getXcp, getXcf
# -----------------------------
def getXcp(cluster_dict, membership):
    XcList = []
    for mem in membership:
        head = np.zeros((len(cluster_dict), latent_dim), dtype=np.float32)
        for i,(cent_p, _) in cluster_dict.items():
            head[i] = mem[i] * cent_p
        XcList.append(head.mean(0))
    return np.stack(XcList, 0)

def getXcf(cluster_dict, membership):
    XcList = []
    for mem in membership:
        head = np.zeros((len(cluster_dict), future_latent_dim), dtype=np.float32)
        for i,(_, cent_f) in cluster_dict.items():
            head[i] = mem[i] * cent_f
        XcList.append(head.mean(0))
    return np.stack(XcList, 0)

# -----------------------------
# MODULES
# -----------------------------
class Reshape(nn.Module):
    def __init__(self, shape): super().__init__(); self.shape = shape
    def forward(self, x): return x.view(-1, *self.shape)

class ReshapeDecode(nn.Module):
    def __init__(self, d1, d2): super().__init__(); self.d1, self.d2 = d1, d2
    def forward(self, x): return x.view(-1, self.d1, self.d2)

class InceptionPast(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        f, k = 32, (5,7,9)
        self.mp = nn.MaxPool1d(3,1,1,return_indices=True)
        self.c1 = nn.Conv1d(1,f,k[0],padding=k[0]//2,bias=False)
        self.c2 = nn.Conv1d(1,f,k[1],padding=k[1]//2,bias=False)
        self.c3 = nn.Conv1d(1,f,k[2],padding=k[2]//2,bias=False)
        self.c4 = nn.Conv1d(1,f,1,bias=False)
        self.bn = nn.BatchNorm1d(4*f); self.act = nn.ReLU(); self.flat = nn.Flatten()
        self.fc1 = nn.Linear(4*f*seq_len,4*32*10)
        self.fc2 = nn.Linear(4*32*10,4*32)
        self.fc3 = nn.Linear(4*32,4*8)
    def forward(self, x):
        m,i = self.mp(x)
        z = torch.cat([self.c1(x),self.c2(x),self.c3(x),self.c4(m)],1)
        z = self.act(self.bn(z)); z = self.flat(z)
        z = self.act(self.fc1(z)); z = self.act(self.fc2(z))
        return self.fc3(z), i

class InceptionFuture(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        f, k = 32, (5,7,9)
        self.mp = nn.MaxPool1d(3,1,1,return_indices=True)
        self.c1 = nn.Conv1d(1,f,k[0],padding=k[0]//2,bias=False)
        self.c2 = nn.Conv1d(1,f,k[1],padding=k[1]//2,bias=False)
        self.c3 = nn.Conv1d(1,f,k[2],padding=k[2]//2,bias=False)
        self.c4 = nn.Conv1d(1,f,1,bias=False)
        self.bn = nn.BatchNorm1d(4*f); self.act = nn.ReLU(); self.flat = nn.Flatten()
        self.fc1 = nn.Linear(4*f*seq_len,4*32*50)
        self.fc2 = nn.Linear(4*32*50,4*32*5)
        self.fc3 = nn.Linear(4*32*5,4*32)
    def forward(self, x):
        m,i = self.mp(x)
        z = torch.cat([self.c1(x),self.c2(x),self.c3(x),self.c4(m)],1)
        z = self.act(self.bn(z)); z = self.flat(z)
        z = self.act(self.fc1(z)); z = self.act(self.fc2(z))
        return self.fc3(z), i

class InceptionTranspose(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4*8+4*32,4*32*10)
        self.fc2 = nn.Linear(4*32*10,4*32*50)
        self.fc3 = nn.Linear(4*32*50,4*32*300)
        self.rs  = ReshapeDecode(128,300)
        f, k = 32, (5,7,9)
        self.t1  = nn.ConvTranspose1d(128,f,k[0],padding=k[0]//2,bias=False)
        self.t2  = nn.ConvTranspose1d(128,f,k[1],padding=k[1]//2,bias=False)
        self.t3  = nn.ConvTranspose1d(128,f,k[2],padding=k[2]//2,bias=False)
        self.c4  = nn.Conv1d(128,1,1,bias=False)
        self.up  = nn.MaxUnpool1d(3,1,1)
        self.bt  = nn.Conv1d(3*f,1,1,bias=False)
        self.bn  = nn.BatchNorm1d(1); self.act = nn.ReLU(); self.flat = nn.Flatten()
    def forward(self, x):
        z, i1, i2 = x
        h = self.act(self.fc1(z)); h = self.act(self.fc2(h)); h = self.fc3(h)
        h = self.rs(h)
        z1,z2,z3,z4 = self.t1(h), self.t2(h), self.t3(h), self.c4(h)
        p1,p2       = torch.split(z4, [History_len, Future_len], -1)
        u1,u2       = self.up(p1, i1), self.up(p2, i2)
        m           = torch.cat([u1, u2], -1)
        b           = self.bt(torch.cat([z1, z2, z3], 1))
        return self.flat(self.act(self.bn(b + m)))

class DeepFeedforward_MultiLabels_senti(nn.Module):
    def __init__(self, num_clusters):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim,128); self.bn1 = nn.BatchNorm1d(128); self.dp1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128,64);          self.bn2 = nn.BatchNorm1d(64);  self.dp2 = nn.Dropout(0.3)
        self.s1  = nn.Linear(senti_dim,32);    self.dp3 = nn.Dropout(0.2)
        self.ge  = nn.Linear(graph_dim,32);    self.dp4 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(64+32, num_clusters)
        self.sig = nn.Sigmoid()
    def forward(self, xp, xs, xg):
        h = F.relu(self.bn1(self.fc1(xp))); h = self.dp1(h)
        h = F.relu(self.bn2(self.fc2(h))); h = self.dp2(h)
        s = self.dp3(F.relu(self.s1(xs)))
        g = self.dp4(F.relu(self.ge(xg)))
        return self.sig(self.fc4(torch.cat([h, g], 1)))

class Prediction_FC(nn.Module):
    def __init__(self):
        super().__init__()
        self.p1  = nn.Linear(latent_dim,24)
        self.p2  = nn.Linear(24+latent_dim+future_latent_dim,400); self.dp5 = nn.Dropout(0.3)
        self.p3  = nn.Linear(400,200);                              self.dp6 = nn.Dropout(0.3)
        self.p4  = nn.Linear(200, future_latent_dim)
    def forward(self, x):
        xp, cp, cf = x
        h = torch.sigmoid(self.p1(xp - cp))
        h = torch.cat([xp, h, cf], 1)
        h = self.dp5(F.relu(self.p2(h)))
        h = self.dp6(F.relu(self.p3(h)))
        return self.p4(h)

class GraphEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(FIXED_LENGTH, graph_dim)
        self.bn  = nn.BatchNorm1d(graph_dim)
    def forward(self, A):
        if A.dim()==2:
            return self.bn(self.lin(A))
        b,n,T = A.shape
        flat = A.view(-1, T)
        emb  = self.bn(self.lin(flat)).view(b, n, graph_dim)
        return emb.mean(dim=1)

# -----------------------------
# TRAINING FUNCTIONS
# -----------------------------
def fit_autoencoder():
    graph_emb  = GraphEmbedding().to(device)
    Enc_past   = nn.Sequential(Reshape((1,History_len)), InceptionPast(History_len)).to(device)
    Enc_fut    = nn.Sequential(Reshape((1,Future_len)), InceptionFuture(Future_len)).to(device)
    Dec        = InceptionTranspose().to(device)

    opt_EP     = torch.optim.Adam(Enc_past.parameters(),  lr=ep_lr, weight_decay=weight_decay)
    opt_EF     = torch.optim.Adam(Enc_fut.parameters(),   lr=ef_lr, weight_decay=weight_decay)
    opt_DEC    = torch.optim.Adam(Dec.parameters(),       lr=d_lr, weight_decay=weight_decay)
    sched_EP   = torch.optim.lr_scheduler.CosineAnnealingLR(opt_EP,   T_max=encoder_epoch)
    sched_EF   = torch.optim.lr_scheduler.CosineAnnealingLR(opt_EF,   T_max=encoder_epoch)
    sched_DEC  = torch.optim.lr_scheduler.CosineAnnealingLR(opt_DEC,  T_max=encoder_epoch)
    mse_loss   = nn.MSELoss()

    for ep in range(encoder_epoch):
        Enc_past.train(); Enc_fut.train(); Dec.train()
        losses = []
        idx = list(range(train_n)); random.shuffle(idx)
        for i in range(0, train_n, encoder_batch):
            bs = idx[i:i+encoder_batch]
            X  = train_ts_tensor[bs]
            P  = X[:,:History_len].unsqueeze(1)
            Fp = X[:,History_len:].unsqueeze(1)

            opt_EP.zero_grad(); opt_EF.zero_grad(); opt_DEC.zero_grad()
            XP,i1 = Enc_past(P); XF,i2 = Enc_fut(Fp)
            LAT   = torch.cat([XP, XF],1)
            REC   = Dec([LAT, i1, i2])
            loss  = mse_loss(REC, X)
            loss.backward()
            for opt in (opt_EP,opt_EF,opt_DEC):
                torch.nn.utils.clip_grad_norm_(opt.param_groups[0]['params'], 5)
            opt_EP.step(); opt_EF.step(); opt_DEC.step()
            losses.append(loss.item())

        sched_EP.step(); sched_EF.step(); sched_DEC.step()
        avg_loss = np.mean(losses)

        # ——— FIXED VALIDATION: capture indices for unpooling ———
        Enc_past.eval(); Enc_fut.eval(); Dec.eval()
        with torch.no_grad():
            Pval = val_ts_tensor[:, :History_len].unsqueeze(1)
            Fval = val_ts_tensor[:, History_len:].unsqueeze(1)
            XPv, i1v = Enc_past(Pval)
            XFv, i2v = Enc_fut(Fval)
            LATv      = torch.cat([XPv, XFv], 1)
            RECv      = Dec([LATv, i1v, i2v]).cpu().numpy()
            true      = val_ts[:,History_len:].flatten()
            pred      = RECv[:,History_len:].flatten()
            pcc       = pearsonr(true, pred)[0]

        print(f"[AE] Epoch {ep+1}/{encoder_epoch} — Loss: {avg_loss:.4f}, Val PCC: {pcc:.4f}")

    return Enc_past, Enc_fut, Dec

def fit_classifier(Enc_past, Enc_fut):
    # ... identical to before ...
    # (no Dec calls here, so nothing to fix)
    # returns Classifier, cluster_dict, gm

    # For brevity, assume this function is unchanged from the previous full listing.

    raise NotImplementedError("Use your existing fit_classifier() here.")




def fit_predictor(Enc_past, Enc_fut, Classifier, cluster_dict, gm, Dec):
    Predictor = Prediction_FC().to(device)
    opt_PRE   = torch.optim.Adam(Predictor.parameters(), lr=pre_lr, weight_decay=weight_decay)
    sched_PRE = torch.optim.lr_scheduler.CosineAnnealingLR(opt_PRE, T_max=predict_epoch)
    mse       = nn.MSELoss()

    for ep in range(predict_epoch):
        Predictor.train()
        losses = []
        idx = list(range(train_n)); random.shuffle(idx)

        for i in range(0, train_n, predict_batch):
            bs = idx[i:i+predict_batch]
            X  = train_ts_tensor[bs]
            P  = X[:,:History_len].unsqueeze(1)
            Fp = X[:,History_len:].unsqueeze(1)

            with torch.no_grad():
                XP,_ = Enc_past(P)
                XF,_ = Enc_fut(Fp)

            Z   = np.concatenate([XP.cpu().numpy(), XF.cpu().numpy()],1)
            mb  = gm.predict_proba(Z)
            Xcp = torch.tensor(getXcp(cluster_dict, mb), device=device)
            Xcf = torch.tensor(getXcf(cluster_dict, mb), device=device)

            Yf  = Predictor([XP, Xcp, Xcf])
            loss= mse(F.normalize(Yf,2,1), F.normalize(XF,2,1))
            opt_PRE.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(Predictor.parameters(), 5)
            opt_PRE.step()
            losses.append(loss.item())

        sched_PRE.step()
        avg_loss = np.mean(losses)

        # val PCC
        Predictor.eval()
        with torch.no_grad():
            Pv,_ = Enc_past(val_ts_tensor[:,:History_len].unsqueeze(1))
            Fv,_ = Enc_fut(val_ts_tensor[:,History_len:].unsqueeze(1))
            Zv   = np.concatenate([Pv.cpu().numpy(), Fv.cpu().numpy()],1)
            mbv  = gm.predict_proba(Zv)
            Xcpv = torch.tensor(getXcp(cluster_dict, mbv), device=device)
            Xcfv = torch.tensor(getXcf(cluster_dict, mbv), device=device)
            Yfv  = Predictor([Pv, Xcpv, Xcfv]).cpu().numpy()
            true_f = val_ts[:,History_len:]
            pcc = pearsonr(true_f.flatten(), Yfv.flatten())[0]

        print(f"[PR] Epoch {ep+1}/{predict_epoch} — Loss: {avg_loss:.4f}, Val PCC: {pcc:.4f}")

    # final test eval
    with torch.no_grad():
        Pt,_   = Enc_past(test_ts_tensor[:,:History_len].unsqueeze(1))
        Ft,_   = Enc_fut(test_ts_tensor[:,History_len:].unsqueeze(1))
        Gt     = GraphEmbedding().to(device)(test_g_tensor)
        mb_t   = Classifier(Pt, test_s_tensor, Gt).cpu().numpy()
        Xcp_t  = getXcp(cluster_dict, mb_t)
        Xcf_t  = getXcf(cluster_dict, mb_t)
        Yf_t   = Predictor([Pt,
                             torch.tensor(Xcp_t,device=device),
                             torch.tensor(Xcf_t,device=device)]).cpu().numpy()
        true_f = test_ts[:,History_len:]
        pcc    = pearsonr(true_f.flatten(), Yf_t.flatten())[0]
        mse_v  = np.mean((true_f - Yf_t)**2)
        rmse   = math.sqrt(mse_v)
        mfe    = np.mean(true_f - Yf_t)

    print(f"\nFINAL → PCC={pcc:.4f}, RMSE={rmse:.4f}, MFE={mfe:.4f}, MSE={mse_v:.5f}")

    # save models
    torch.save(Enc_past.state_dict(),   os.path.join(OUTPUT_DIR,"Enc_past.pt"))
    torch.save(Enc_fut.state_dict(),    os.path.join(OUTPUT_DIR,"Enc_fut.pt"))
    torch.save(Dec.state_dict(),        os.path.join(OUTPUT_DIR,"Dec.pt"))
    torch.save(Classifier.state_dict(), os.path.join(OUTPUT_DIR,"Classifier.pt"))
    torch.save(Predictor.state_dict(),  os.path.join(OUTPUT_DIR,"Predictor.pt"))
    with open(os.path.join(OUTPUT_DIR,"gmm_model.pkl"), "wb") as f:
        pickle.dump(gm, f)

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    Enc_past, Enc_fut, Dec = fit_autoencoder()
    Classifier, cluster_dict, gm = fit_classifier(Enc_past, Enc_fut)
    fit_predictor(Enc_past, Enc_fut, Classifier, cluster_dict, gm, Dec)
