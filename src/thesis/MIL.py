#%%

import gc
import random
import numpy as np
import pandas as pd
import torch
import optuna
import lightning as L
import mlflow
from typing import Optional
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold,RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torchmetrics import Accuracy, F1Score
from torcheval.metrics import MulticlassAUROC, MulticlassAUPRC
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import EarlyStopping
from torchmetrics import Accuracy, F1Score, Precision, Recall  
from sklearn.metrics import confusion_matrix, classification_report  
#%%

#%%
# =============== Reproducibility ===============
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
L.seed_everything(SEED, workers=True)
#%%
#============== MLflow config ====================
MLFLOW_TRACKING_URI = "file:./mlruns"   # change if you use a server
MLFLOW_EXPERIMENT   = "Multiple Instance Learning_ADIS"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)   # creates or activates by name

#%%
# =============== Paths ===============
cell_path = '/home/skshastry/Project/rosmap_brain_blood/data/MIL_cell.json'
tcr_path  = '/home/skshastry/Project/rosmap_brain_blood/data/MIL_tcr.json'
meta_path = '/home/skshastry/Project/rosmap_brain_blood/data/metadata.csv'

#%%
# =============== Load data ===============
cell_emb = pd.read_json(cell_path, lines=True)
tcr_data = pd.read_json(tcr_path,  lines=True)
metadata = pd.read_csv(meta_path, index_col=0)
#%%
tcr_data
#%%
cell_emb
 #%%
# Merge
merged_data = pd.merge(cell_emb, tcr_data, on='patient_id', how='inner')
merged_data = pd.merge(merged_data, metadata, on='patient_id', how='inner')

# Ensure label is numeric
if not np.issubdtype(merged_data['diagnosis'].dtype, np.integer):
    le = LabelEncoder()
    merged_data['diagnosis'] = le.fit_transform(merged_data['diagnosis'])

# Prepare features
features = merged_data.drop(columns=['diagnosis']).copy()

#%%
X_tcr_all = np.stack(merged_data['tcr_emb'].apply(np.asarray)).astype(np.float32)     # (N, d_tcr)
X_age_all = merged_data['Age'].to_numpy(np.float32)[:, None]                          # (N, 1)

# Gender -> one-hot robustly
g_raw = merged_data['Gender']
if g_raw.dtype == object:
    g_idx = g_raw.map({'F': 0, 'M': 1}).fillna(0).to_numpy(np.int64)                  # fallback 0
else:
    g_idx = np.nan_to_num(g_raw.to_numpy(np.float32), nan=0.0).astype(np.int64)       # ensure 0/1
X_sex_all = np.eye(2, dtype=np.float32)[g_idx]                                        # (N, 2)

X_bagcov_all = np.concatenate([X_tcr_all, X_age_all, X_sex_all], axis=1).astype(np.float32)
tcr_dim = X_tcr_all.shape[1]
age_col = tcr_dim
SEX_START = age_col + 1
SEX_END=SEX_START + 2
cont_cols = np.arange(0, SEX_START)        # 0 .. 15406 (cells+TCR+age)
sex_cols  = np.arange(SEX_START, SEX_END)  
# --- instance tensor split ---
X_cell_emb = np.stack(merged_data['cell_emb'].apply(np.asarray)).astype(np.float32)   # (N, T, 513)
X_embed = X_cell_emb[..., :512]                                                    # (N, T, 512)

# counts per cell type
cnt_feat_np = X_cell_emb[..., 512].astype(np.float32)                                  # (N, T)
counts_np = np.maximum(cnt_feat_np, 0.0)
count_feat = np.log1p(counts_np).astype(np.float32)
X_inst_all = np.concatenate([X_embed, count_feat[..., None]], axis=-1)  # (N, T, 513)

y_all = merged_data['diagnosis'].to_numpy(np.int64)

# calculating dimensions
in_dim=X_inst_all.shape[-1]
cov_dim=X_bagcov_all.shape[-1]

#no of unique classes to be predicted
n_cls = len(np.unique(y_all))
counts_np
#==================================== Gated Attention model===========================
#%%
class GatedAttentionMIL(nn.Module):
    """
    Gated attention MIL.
    H: (B, T, D) 
    mask: [B,T] True if T present
    Returns:
        -> bag Z: (B, D) bag embedding, attention A: (B, T)
    """
    def __init__(self,h_dim:int, attn_dim: int = 64):
        super().__init__()
        self.V = nn.Linear(h_dim, attn_dim, bias=True)
        self.U = nn.Linear(h_dim, attn_dim, bias=True)
        self.w = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, H: Tensor, mask:Optional[Tensor]=None):
        Vh = torch.tanh(self.V(H))            # (B,T,D) (captures the content)
        Uh = torch.sigmoid(self.U(H))         # (B,T,D) (acts like a gate(decides which parts are important))
        scores = self.w(Vh * Uh).squeeze(-1)  # (B,T) (cell type imp for the bag)
        if mask is not None:
            scores=scores.masked_fill(~mask, float('-inf')) # prevents missing cell types from recieving attention
        A = torch.softmax(scores, dim=1)        # (B,T) attention scores for each cell type
        Z = torch.einsum('bth,bt->bh', H, A)  # (B,D) weigthed sum pooling per pateint
        return Z, A
#%%
#========================classification model=============================
    
class Classification(nn.Module):
    def __init__(self, d_model: int, n_cls: int, dropout_rate: float,
                 nlayers: int = 3, activation=nn.ReLU):
        super().__init__()
        layers = []
        for _ in range(nlayers - 1):
            layers.append(nn.Linear(d_model,d_model ))
            layers.append(activation())
            layers.append(nn.LayerNorm(d_model))
            layers.append(nn.Dropout(dropout_rate))
        self.backbone = nn.Sequential(*layers)
        self.out = nn.Linear(d_model, n_cls)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return self.out(x)

#%%

# ==============Lightning wrapper that uses (attention + Classification) ==============
    
class MIL(L.LightningModule):
    def __init__(
        self,
        in_dim: int,
        attn_dim:int,
        n_cls: int,
        cls_nlayers: int = 3,
        cls_dropout: float = 0.2,
        cls_activation = nn.ReLU,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.0,
        cov_dim:int= 0 ):

        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.multiROC = MulticlassAUROC(num_classes=n_cls,average='macro')
        self.multiPR = MulticlassAUPRC(num_classes=n_cls,average='macro')
        self.accuracy   = Accuracy(task='multiclass', num_classes=n_cls)
        self.f1         = F1Score(task='multiclass', num_classes=n_cls, average='macro')   
        self.prec_macro = Precision(task='multiclass', num_classes=n_cls, average='macro') 
        self.rec_macro  = Recall(task='multiclass', num_classes=n_cls, average='macro')   

        self.pool = GatedAttentionMIL(h_dim=in_dim, attn_dim=attn_dim)
        cls_in= in_dim + (cov_dim if cov_dim else 0)

        self.classifier = Classification(
            d_model=cls_in,
            n_cls=n_cls,
            dropout_rate=cls_dropout,
            nlayers=cls_nlayers,
            activation=cls_activation,
        )

    def forward(self, H: Tensor,cov:Optional[Tensor],mask:Optional[Tensor]):
        # H: [B, N, D]
        z, a = self.pool(H,mask=mask)   
        if cov is not None and cov.numel()>0:
            z= torch.cat([z,cov],dim=1)
        logits=self.classifier(z)                         # z: [B, D], a: [B, N]         # [B, C]
        return logits, a

    def _step(self, batch, stage: str):
        H, cov, mask, y = batch              # << EXPECT THIS BATCH SHAPE
        logits, attn = self(H, cov, mask)
        loss = self.criterion(logits, y)
        probs = logits.softmax(-1)
        preds = probs.argmax(dim=-1)
        if stage =="val":
            self.multiROC.update(probs, y); self.multiPR.update(probs, y)

        self.log(f"{stage}_loss", loss, prog_bar=(stage!='test'))
        self.log(f"{stage}_acc",  self.accuracy(preds, y), prog_bar=True)
        self.log(f"{stage}_f1",   self.f1(preds, y),       prog_bar=True)
        self.log(f"{stage}_prec_macro", self.prec_macro(preds, y))
        self.log(f"{stage}_rec_macro",  self.rec_macro(preds, y))
        return loss

    def training_step(self, batch, batch_idx):   return self._step(batch, "train")
    def validation_step(self, batch, batch_idx): return self._step(batch, "val")

    def test_step(self, batch, idx):
            H, cov, mask, y = batch
            logits, attn = self(H, cov, mask)
            loss = self.criterion(logits, y)
            probs = logits.softmax(-1)
            preds = probs.argmax(dim=-1)
            # collect for CM
            if not hasattr(self, "_test_preds"):
                self._test_preds, self._test_targets = [], []
            self._test_preds.append(preds.detach().cpu())
            self._test_targets.append(y.detach().cpu())
            self.multiROC.update(probs, y); self.multiPR.update(probs, y)

            self.log("test_loss", loss, prog_bar=True)
            self.log("test_acc",  self.accuracy(preds, y), prog_bar=True)
            self.log("test_f1",   self.f1(preds, y),       prog_bar=True)
            self.log("test_prec_macro", self.prec_macro(preds, y))
            self.log("test_rec_macro",  self.rec_macro(preds, y))
            return loss    
    
    def on_validation_epoch_end(self):
        auroc = self.multiROC.compute()
        aupr    = self.multiPR.compute()
        self.log("val_auroc", auroc, prog_bar=True)
        self.log("val_auprc",    aupr,    prog_bar=True)
        self.multiROC.reset()
        self.multiPR.reset()
    
    def on_test_start(self):  # init once at test begin
        self._test_preds = []
        self._test_targets = []
        self.multiROC.reset()
        self.multiPR.reset()

    def on_test_epoch_end(self):
        """
        - Aggregate AUROC/AUPRC across the whole test epoch.
        - Print confusion matrix & per-class report.
        - Reset metric objects and buffers.
        """
        auroc = self.multiROC.compute()
        self.log("test_auroc", auroc, prog_bar=True)
        self.multiROC.reset()

        auprc = self.multiPR.compute()
        self.log("test_auprc", auprc, prog_bar=True)
        self.multiPR.reset()

        y_true = torch.cat(self._test_targets).numpy()
        y_pred = torch.cat(self._test_preds).numpy()

        cm = confusion_matrix(y_true, y_pred)
        print("Confusion matrix:\n", cm)
        print(classification_report(y_true, y_pred, digits=3))

        # now clear buffers
        self._test_preds.clear()
        self._test_targets.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(                                      
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
#================================= Helper function for data=======================
class BagDataset(Dataset):

    def __init__(self,X_inst_all,counts_np,X_bagcov_all,y_all):
        H = torch.as_tensor(X_inst_all, dtype=torch.float32)
        y = torch.as_tensor(y_all,      dtype=torch.long)

        self.H=H
        self.y=y
        self.cov= (None if X_bagcov_all is None
                   else torch.as_tensor(X_bagcov_all, dtype=torch.float32))
        if counts_np is None:
            self.mask=None
        else:
            m = torch.as_tensor(counts_np, dtype=torch.float32)               # (N, T)
            self.mask = (m > 0).to(torch.bool)
    
    def __len__(self): return len(self.y)

    def __getitem__(self, i) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        H_i = self.H[i]                         # (T, D)
        cov_i = None if self.cov is None else self.cov[i]      # (C,) or None
        mask_i = None if self.mask is None else self.mask[i]   # (T,) or None
        y_i = self.y[i]                         # ()
        return H_i, cov_i, mask_i, y_i
    
#=========================Scaling=========================
def scale_inner(X_cov,tr_idx,va_idx):
    """
    Inner fold scaling
    scales only columns - tcr embeddings and age
    Gender - one hot encoded
    """
    scaler=StandardScaler()
    Xtr=X_cov[tr_idx].copy()
    Xva=X_cov[va_idx].copy()
    Xtr[:,cont_cols] = scaler.fit_transform(Xtr[:,cont_cols])
    Xva[:,cont_cols] = scaler.transform(Xva[:,cont_cols])
    return Xtr,Xva

def scale_outer(X_cov,tro_idx,vao_idx):
    """
    Outer fold
    scales only columns - tcr embeddings and age
    Gender - one hot encoded
    """
    scaler=StandardScaler()
    Xtrv=X_cov[tro_idx].copy()
    Xte=X_cov[vao_idx].copy()
    Xtrv[:,cont_cols] = scaler.fit_transform(Xtrv[:,cont_cols])
    Xte[:,cont_cols] = scaler.transform(Xte[:,cont_cols])
    return Xtrv,Xte

#======================= Trainer Helper function============================

def make_trainer(max_epochs, logger=None, use_early_stopping=None, precision=16,monitor='val_auroc',mode='max',patience=7,extra_callbacks=None):
    '''
    
    Create a Lightning Trainer with consistent, memory-safe defaults.

    Args:
        max_epochs (int): Max number of epochs to train.
        logger: Optional Lightning logger (e.g., MLFlowLogger). Use None to disable logging.
        use_early_stopping (bool): If True, attach EarlyStopping(monitor="val_loss_pat").
            Note: only enable this when you pass a validation dataloader to `fit(...)`.
        precision (int|str): Numerical precision for training (e.g., 16 for FP16).

    Returns:
        lightning.pytorch.Trainer: Configured trainer instance.
    
    '''
    callbacks = [EarlyStopping(monitor="val_auroc", patience=5)] if use_early_stopping else None
    trainer=L.Trainer(
        logger=logger,
        max_epochs=max_epochs,
        accelerator='gpu',
        devices=1,
        precision=precision,              
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        callbacks=callbacks,
        log_every_n_steps=50,
    )
    return trainer

#=============================Nested CV with HPO- OPTUNA===========================================
outer_folds=5
n_inner=3
outer_cv=StratifiedKFold(n_splits=outer_folds, shuffle=True,random_state=SEED)

outer_results = []
for outer_fold, (trainval_idx, test_idx) in enumerate(outer_cv.split(X_inst_all, y_all), start=1):
    print(f"\n=== Outer Fold {outer_fold}/{outer_folds} ===")

    # ------- Inner Optuna search (3-fold) -------
    def objective(trial):
        params = {
            "attn_dim":     trial.suggest_int("attn_dim", 256,256),
            "cls_nlayers":  trial.suggest_int("cls_nlayers", 1, 1),
            "cls_dropout":  trial.suggest_float("cls_dropout", 0.2, 0.5),
            "lr":           trial.suggest_float("lr", 5e-4, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-3, log=True),
            "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.1),
            "batch_size":   trial.suggest_categorical("batch_size", [4, 8, 16]),
            "max_epochs":   trial.suggest_int("max_epochs", 30, 40),
            "patience":     trial.suggest_int("patience", 5, 10),
        }

        inner_cv = StratifiedKFold(n_splits=n_inner,shuffle=True, random_state=SEED)
        scores = []
        for tr_rel, va_rel in inner_cv.split(trainval_idx, y_all[trainval_idx]):
            tr_ids = trainval_idx[tr_rel]
            va_ids = trainval_idx[va_rel]
            Xcov_tr, Xcov_va= scale_inner(X_bagcov_all, tr_ids, va_ids)

            train_ds = BagDataset(X_inst_all[tr_ids],counts_np[tr_ids], Xcov_tr,y_all[tr_ids])
            val_ds   = BagDataset(X_inst_all[va_ids], counts_np[va_ids], Xcov_va,y_all[va_ids] )

            train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True,
                                    num_workers=0, pin_memory=False)
            val_loader   = DataLoader(val_ds,   batch_size=params["batch_size"], shuffle=False,
                                    num_workers=0, pin_memory=False)

            model = MIL(
                in_dim=X_inst_all.shape[-1],   
                attn_dim=params["attn_dim"],  
                n_cls=n_cls,
                cls_nlayers=params["cls_nlayers"],
                cls_dropout=params["cls_dropout"],
                lr=params["lr"],
                weight_decay=params["weight_decay"],
                label_smoothing=params["label_smoothing"],
                cov_dim=X_bagcov_all.shape[1]
            )

            mlf_logger = MLFlowLogger(
                experiment_name='Multiple Instance Learning_ADIS',
                run_name=f"outer{outer_fold}_trial{trial.number}",
                tracking_uri='file:./mlruns'
            )
        
        #guarantees each trial/fold leaves a “clean slate” before the next one start (cleans up GPU memory)
            trainer = make_trainer(
                        max_epochs=params["max_epochs"],
                        logger=mlf_logger,
                        use_early_stopping=True,
                        monitor='val_auroc',
                        mode='max',
                        patience=params["patience"]
                    )
            
            try:
                trainer.fit(model, train_loader, val_loader)
                val_metrics = trainer.validate(model, val_loader, verbose=False)[0]
            finally:
                # full cleanup after each inner fold  
                try:
                    trainer.teardown()
                except Exception:
                    pass
                del model, trainer, train_loader, val_loader
                torch.cuda.empty_cache()
                gc.collect()

            
            auroc=float(val_metrics.get("val_auroc", np.nan))
            scores.append(auroc)
        return float(np.nanmean(scores))
    

    study = optuna.create_study(direction="maximize", study_name=f"outer{outer_fold}_inner{n_inner}")
    study.optimize(objective, n_trials=40, show_progress_bar=False)
    best_params = study.best_params
    print("Best inner params:", best_params, " (avg val auroc:", study.best_value, ")")

    # ------- Retrain on full outer train+val with best params; test on outer test -------
    Xcov_trv, Xcov_te = scale_outer(X_bagcov_all, trainval_idx, test_idx)

    trainval_ds = BagDataset(X_inst_all[trainval_idx], counts_np[trainval_idx], Xcov_trv,y_all[trainval_idx] )
    test_ds     = BagDataset(X_inst_all[test_idx],counts_np[test_idx],Xcov_te, y_all[test_idx])

    trainval_loader = DataLoader(trainval_ds, batch_size=best_params["batch_size"], shuffle=True,
                                 num_workers=0, pin_memory=False)
    test_loader     = DataLoader(test_ds,     batch_size=best_params["batch_size"], shuffle=False,
                                 num_workers=0, pin_memory=False)

    final_model = MIL(
        in_dim=X_inst_all.shape[-1],
        attn_dim=best_params["attn_dim"],
        n_cls=n_cls,
        cls_nlayers=best_params["cls_nlayers"],
        cls_dropout=best_params["cls_dropout"],
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
        label_smoothing=best_params["label_smoothing"],
        cov_dim=X_bagcov_all.shape[1]
    )

    trainer = make_trainer(
            max_epochs=best_params["max_epochs"],
            logger=None,
            use_early_stopping=False,
            monitor='val_auroc',
            mode='max',
            patience=best_params["patience"]
            )
    
  # cleanup(GPU) after each outer fold
    try:
        trainer.fit(final_model, trainval_loader,test_loader)
        test_metrics = trainer.test(final_model, test_loader, verbose=False)[0]
    finally: 
        try:
            trainer.teardown()
        except Exception:
            pass
        del final_model, trainer, trainval_loader, test_loader
        torch.cuda.empty_cache()
        gc.collect()


    outer_results.append({
        "acc":   float(test_metrics.get("test_acc",   np.nan)),
        "f1":    float(test_metrics.get("test_f1",    np.nan)),
        "auroc": float(test_metrics.get("test_auroc", np.nan)),
        "auprc": float(test_metrics.get("test_auprc", np.nan)),
    })
    print(f"Outer {outer_fold} → acc={outer_results[-1]['acc']:.3f}, "
          f"f1={outer_results[-1]['f1']:.3f}, "
          f"auroc={outer_results[-1]['auroc']:.3f}, "
          f"auprc={outer_results[-1]['auprc']:.3f}")

# ===================Metrics Summary==========================================
    
accs   = np.array([r["acc"]   for r in outer_results], dtype=float)
f1s    = np.array([r["f1"]    for r in outer_results], dtype=float)
aurocs = np.array([r["auroc"] for r in outer_results], dtype=float)
auprcs = np.array([r["auprc"] for r in outer_results], dtype=float)

print("\n=== Final Nested CV (outer) ===")
print(f"ACC   mean±std: {np.nanmean(accs):.4f} ± {np.nanstd(accs):.4f}")
print(f"F1    mean±std: {np.nanmean(f1s):.4f} ± {np.nanstd(f1s):.4f}")
print(f"AUROC mean±std: {np.nanmean(aurocs):.4f} ± {np.nanstd(aurocs):.4f}")
print(f"AUPRC mean±std: {np.nanmean(auprcs):.4f} ± {np.nanstd(auprcs):.4f}")