#%%
import gc
import random
import numpy as np
import pandas as pd
import torch
import optuna
import lightning as L
import mlflow
import ast
import csv
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
from model import GatedAttentionMIL,Classification
import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"

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
MLFLOW_EXPERIMENT   = "Multiple Instance Learning_ADIS(StratifiedKFold=5)"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)   # creates or activates by name
#%%
# =============== Paths ===============
cell_path = '/home/skshastry/Project/rosmap_brain_blood/data/MIL_scRNA.json'
tcr_path  = '/home/skshastry/Project/rosmap_brain_blood/data/MIL_tcr.json'
#%%
# =============== Loading and pre-processing data ===============
cell_emb = pd.read_json(cell_path, lines=True)
tcr_data = pd.read_json(tcr_path,  lines=True)
covariates=cell_emb[['Age','Gender']]
scRNA_emb=cell_emb.drop(columns=['Age','Gender','diagnosis'])
#%%
#===========processing metadata for input============================
age=covariates['Age'].to_numpy(np.int64)[:,None]
g=covariates['Gender'].to_numpy(np.int64)
gender=np.eye(2,dtype=np.float32)[g]
meta_fin=np.concatenate([age,gender],axis=1).astype(np.float32)
#%%
# Function to convert string to float array
def parse_tcr_emb(bag):
    """
    bag: list of strings (each string represents a TCR embedding)
    Returns: list of np.arrays (float32)
    """
    parsed_bag = [np.fromstring(s.strip('[]'), sep=',', dtype=np.float32) for s in bag]
    return np.stack(parsed_bag)  # shape = (N_cells, D_tcr)
#%%
sc_input = scRNA_emb['sc_emb_list'].apply(lambda x: np.asarray(x)).tolist()
tcr_input=tcr_data['tcr_emb_list'].apply(lambda x: np.asarray(x)).tolist()
#%%
tcr_input_numeric = [parse_tcr_emb(bag) for bag in tcr_input]
sc_input_numeric = [parse_tcr_emb(bag) for bag in sc_input]
#%%
def safe_eval(x):
    if isinstance(x, str):
        return ast.literal_eval(x)
    return x  # already a list

cell_types_scRNA = cell_emb['cell_type_list'].apply(safe_eval).to_list()
cell_types_tcr  = tcr_data['cell_type_list'].apply(safe_eval).to_list()
#%%
#%%

pids = cell_emb['patient_id'].tolist()
meta_fin_tensor = torch.tensor(meta_fin, dtype=torch.float32)
y=cell_emb['diagnosis'].to_numpy(np.int64)
#%%
cont_cols = np.arange(0, 1) 
tcr_dim=tcr_input_numeric[0].shape[-1]
sc_dim=sc_input_numeric[0].shape[-1]
n_cls=3
cov_dim=meta_fin_tensor.shape[-1]
#%%
# ==============Lightning wrapper that uses (Gated attention + Classification) ==============
    
class PatientWrapper(L.LightningModule):
    def __init__(
        self,
        sc_dim: int,
        tcr_dim:int,
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

        self.pool_scRNA = GatedAttentionMIL(dim=sc_dim, attn_dim=attn_dim)
        self.pool_tcr = GatedAttentionMIL(dim=tcr_dim,attn_dim=attn_dim)
        cls_in= sc_dim + tcr_dim + (cov_dim if cov_dim else 0)

        self.classifier = Classification(
            d_model=cls_in,
            n_cls=n_cls,
            dropout_rate=cls_dropout,
            nlayers=cls_nlayers,
            activation=cls_activation,
        )
        self._attention_scRNA=[]
        self._attention_tcr=[]
        self._attention_pid=[]

    def forward(self, H: Tensor,T:Tensor,cov:Optional[Tensor]):
        z_sc, a_sc = self.pool_scRNA(H)
        z_tcr,a_tcr=self.pool_tcr(T)

        z= torch.cat([z_sc,z_tcr,cov],dim=1)

        logits=self.classifier(z)                         # z: [B, D], a: [B, N]         # [B, C]
        return logits, a_sc,a_tcr

    def _step(self, batch, stage: str):
        losses = []
        for H, T, cov, y, pid,cell_type_scRNA,cell_type_tcr in batch:
            H = H.unsqueeze(0).to(self.device)
            T = T.unsqueeze(0).to(self.device)
            cov = cov.unsqueeze(0).to(self.device) if cov is not None else None
            y = y.unsqueeze(0).to(self.device)

            logits, a_sc, a_tcr = self(H, T, cov)
            loss = self.criterion(logits, y)
            losses.append(loss)
            z_sc, _ = self.pool_scRNA(H)
            z_tcr, _ = self.pool_tcr(T)
            z = torch.cat([z_sc, z_tcr, cov], dim=1)

            if stage in ["val", "test"]:
                self._attention_scRNA.append((pid,cell_type_scRNA ,a_sc.detach().cpu().numpy()))
                self._attention_tcr.append((pid,cell_type_tcr, a_tcr.detach().cpu().numpy()))

            probs = logits.softmax(-1)
            preds = probs.argmax(dim=-1)

            if stage == "test" and not hasattr(self, "_test_preds"):
                self._test_preds, self._test_targets = [], []

            if stage == "test":
                self._test_preds.append(preds.detach().cpu())
                self._test_targets.append(y.detach().cpu())
                print(f"Patient {pid}, Z_sc shape: {z_sc.shape}")
                print(f"Patient {pid}, Z_tcr shape: {z_tcr.shape}")
                print(f"Patient {pid}, Z shape: {z.shape}")
                
            if stage != "test" or stage == "test":
                self.multiROC.update(probs, y)
                self.multiPR.update(probs, y)
                self.log(f"{stage}_loss", loss, prog_bar=True)
                self.log(f"{stage}_acc", self.accuracy(preds, y), prog_bar=True)
                self.log(f"{stage}_f1", self.f1(preds, y), prog_bar=True)
                self.log(f"{stage}_prec_macro", self.prec_macro(preds, y))
                self.log(f"{stage}_rec_macro", self.rec_macro(preds, y))

        return torch.mean(torch.stack(losses))

    def training_step(self, batch, batch_idx):   return self._step(batch, "train")
    def validation_step(self, batch, batch_idx): return self._step(batch, "val")
    def test_step(self, batch, batch_idx): return self._step(batch, "test")   
    
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

        #confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print("Confusion matrix:\n", cm)
        print(classification_report(y_true, y_pred, digits=3))
        # ================= Save attention scores as CSV =================
        # scRNA
        # scRNA

        # --------- scRNA ---------
        with open("attention_scRNA.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["patient_id", "cell_type", "attention"])

            for pid, ctypes_sc, attn in self._attention_scRNA:
                # attn: [num_cells, attn_dim] or [num_cells, 1]
                attn_per_cell = attn.flatten() 
                for ct, att_val in zip(ctypes_sc,attn_per_cell):
                    writer.writerow([pid, ct, att_val])
                    

        # --------- TCR ---------
        with open("attention_TCR.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["patient_id", "cell_type", "attention"])

            for pid, ctypes_tcr, attn in self._attention_tcr:
                attn_per_cell = attn.flatten()
                for ct, attn_val_tcr in zip(ctypes_tcr,attn_per_cell):
                    writer.writerow([pid, ct, attn_val_tcr])

        self._test_preds.clear()
        self._test_targets.clear()
        self._attention_scRNA.clear()
        self._attention_tcr.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(                                      
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
#================================= Helper function for data=======================
class BagDataset(Dataset):

    def __init__(self,X_scRNA,X_tcr,X_bagcov_all,y,pids,cell_type_sc,cell_type_tcr):
        y = torch.as_tensor(y, dtype=torch.long)
        self.H=X_scRNA
        self.T=X_tcr
        self.y=y
        self.cell_type_scRNA=cell_type_sc
        self.cell_type_tcr=cell_type_tcr
        self.cov= (None if X_bagcov_all is None
                   else torch.as_tensor(X_bagcov_all, dtype=torch.float32))
        self.pids=pids
    
    def __len__(self): return len(self.y)

    def __getitem__(self, i) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        H_i = torch.tensor(self.H[i], dtype=torch.float32)  
        T_i= torch.tensor(self.T[i], dtype=torch.float32)                      
        cov_i = None if self.cov is None else self.cov[i] 
        y_i = self.y[i]  
        pid=self.pids[i]
        cell_types_sc=self.cell_type_scRNA[i] 
        cell_types_tcr=self.cell_type_tcr[i]                 
        return H_i,T_i, cov_i, y_i,pid,cell_types_sc,cell_types_tcr

def mil_collate_fn(batch):
    """
    batch: list of tuples returned by __getitem__ in Dataset
           Each tuple: (H_i, T_i, cov_i, y_i, patient_id)
    """
    return batch
#=========================Scaling=========================
def scale_inner(X_cov,tr_idx,va_idx):
    """
    Inner fold scaling
    scales only columns -age
    Gender - one hot encoded
    """
    scaler=StandardScaler()
    Xtr = X_cov[tr_idx].cpu().numpy().copy() 
    Xva = X_cov[va_idx].cpu().numpy().copy()
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
    Xtrv=X_cov[tro_idx].numpy().copy()
    Xte=X_cov[vao_idx].numpy().copy()
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
outer_cv=StratifiedKFold(n_splits=outer_folds,shuffle=True,random_state=SEED)

outer_results = []
for outer_fold, (trainval_idx, test_idx) in enumerate(outer_cv.split(sc_input_numeric, y), start=1):
    print(f"\n=== Outer Fold {outer_fold}/{outer_folds} ===")

    # ------- Inner Optuna search (3-fold) -------
    def objective(trial):
        params = {
            "attn_dim":     trial.suggest_int("attn_dim", 64,264),
            "cls_nlayers":  trial.suggest_int("cls_nlayers", 1, 2),
            "cls_dropout":  trial.suggest_float("cls_dropout", 0.5, 0.6),
            "lr":           trial.suggest_float("lr", 5e-4, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-3, log=True),
            "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.1),
            "batch_size":   trial.suggest_categorical("batch_size", [4, 8, 16]),
            "patience":     trial.suggest_int("patience", 5, 10),
        }

        inner_cv = StratifiedKFold(n_splits=n_inner,shuffle=True, random_state=SEED)
        scores = []
        for tr_rel, va_rel in inner_cv.split(trainval_idx, y[trainval_idx]):
            tr_ids = trainval_idx[tr_rel]
            va_ids = trainval_idx[va_rel]
            Xcov_tr, Xcov_va= scale_inner(meta_fin_tensor, tr_ids, va_ids)
            Xcov_tr_tensor = torch.tensor(Xcov_tr, dtype=torch.float32)
            Xcov_va_tensor = torch.tensor(Xcov_va, dtype=torch.float32)

            train_ds = BagDataset(
                        [sc_input_numeric[i] for i in tr_ids],
                        [tcr_input_numeric[i] for i in tr_ids],
                        Xcov_tr_tensor,
                        y[tr_ids],
                        [pids[i] for i in tr_ids],
                        [cell_types_scRNA[i] for i in tr_ids],
                        [cell_types_tcr[i] for i in tr_ids]
                    )

            val_ds = BagDataset(
                    [sc_input_numeric[i] for i in va_ids],
                    [tcr_input_numeric[i] for i in va_ids],
                    Xcov_va_tensor,
                    y[va_ids],
                    [pids[i] for i in va_ids],
                    [cell_types_scRNA[i] for i in va_ids],
                    [cell_types_tcr[i] for i in va_ids]
                )


            train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True,
                                    num_workers=0, pin_memory=False,collate_fn=mil_collate_fn)
            val_loader   = DataLoader(val_ds,   batch_size=params["batch_size"], shuffle=False,
                                    num_workers=0, pin_memory=False,collate_fn=mil_collate_fn)

            model = PatientWrapper(
                sc_dim=sc_dim,
                tcr_dim=tcr_dim,   
                attn_dim=params["attn_dim"],  
                n_cls=n_cls,
                cls_nlayers=params["cls_nlayers"],
                cls_dropout=params["cls_dropout"],
                lr=params["lr"],
                weight_decay=params["weight_decay"],
                label_smoothing=params["label_smoothing"],
                cov_dim=cov_dim
            )

            mlf_logger = MLFlowLogger(
                experiment_name=MLFLOW_EXPERIMENT,
                run_name=f"outer{outer_fold}_trial{trial.number}",
                tracking_uri=MLFLOW_TRACKING_URI
            )
        
        #guarantees each trial/fold leaves a “clean slate” before the next one start (cleans up GPU memory)
            trainer = make_trainer(
                        max_epochs=40,
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
    Xcov_trv, Xcov_te = scale_outer(meta_fin_tensor, trainval_idx, test_idx)
    Xcov_tr_tensor = torch.tensor(Xcov_trv, dtype=torch.float32)
    Xcov_te_tensor = torch.tensor(Xcov_te, dtype=torch.float32)

    trainval_ds = BagDataset(
                [sc_input_numeric[i] for i in trainval_idx],
                [tcr_input_numeric[i] for i in trainval_idx], 
                Xcov_tr_tensor,
                y[trainval_idx],
                [pids[i] for i in trainval_idx],
                [cell_types_scRNA[i] for i in trainval_idx],
                [cell_types_tcr[i] for i in trainval_idx]
                )
    test_ds   = BagDataset(
        [sc_input_numeric[i] for i in test_idx],
        [tcr_input_numeric[i] for i in test_idx] , 
        Xcov_te_tensor,
        y[test_idx],
        [pids[i] for i in test_idx],
        [cell_types_scRNA[i] for i in test_idx],
        [cell_types_tcr[i] for i in test_idx]
        )

    trainval_loader = DataLoader(trainval_ds, batch_size=best_params["batch_size"], shuffle=True,
                                 num_workers=0, pin_memory=False,collate_fn=mil_collate_fn)
    test_loader     = DataLoader(test_ds,     batch_size=best_params["batch_size"], shuffle=False,
                                 num_workers=0, pin_memory=False,collate_fn=mil_collate_fn)

    final_model = PatientWrapper(
        sc_dim=sc_dim,
        tcr_dim=tcr_dim,
        attn_dim=best_params["attn_dim"],
        n_cls=n_cls,
        cls_nlayers=best_params["cls_nlayers"],
        cls_dropout=best_params["cls_dropout"],
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
        label_smoothing=best_params["label_smoothing"],
        cov_dim=cov_dim
    )

    trainer = make_trainer(
            max_epochs=40,
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