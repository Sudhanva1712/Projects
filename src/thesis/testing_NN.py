# %%
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import gc
import random
import numpy as np
import pandas as pd
import torch
import optuna
import lightning as L
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight
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
# =============== Reproducibility ===============
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
L.seed_everything(SEED, workers=True)

# =============== Paths ===============
#%%
cell_data = pd.read_json('/home/skshastry/Project/rosmap_brain_blood/test_data_final.json')
cell_data
#%%

# Ensure label is numeric
if not np.issubdtype(cell_data['status'].dtype, np.integer):
    le = LabelEncoder()
    cell_data['status'] = le.fit_transform(cell_data['status'])

cell_data


#%%
# ---- Build aligned arrays ONCE ----
X_age_all   = cell_data['Age_numeric'].to_numpy(np.float32)[:, None]                                   # (N,1)

g = cell_data['Sex'].to_numpy(np.float32)[:, None]
g1d = g.reshape(-1)                         # (N,)
g_idx = g1d.astype(np.int64) 

X_sex_all = np.eye(2, dtype=np.float32)[g_idx] 
X_sex_all
#%%  # shape (N, 2), one-hot
X_cell_emb=np.stack(cell_data['cell_emb'].apply(np.asarray).to_list()).astype(np.float32)
N = X_cell_emb.shape[0]
X_cell_flat=X_cell_emb.reshape(N,-1).astype(np.float32)

#Input array ( features- cell_embeddings, tcr embeddings,age ,sex)
X_all = np.concatenate([X_cell_flat, X_age_all,X_sex_all], axis=1).astype(np.float32)
D_cell=X_cell_flat.shape[1]
AGE_COL=D_cell
SEX_START  = AGE_COL + 1               
SEX_END    = SEX_START + 2
cont_cols = np.arange(0, SEX_START)        # 0 .. 15406 (cells+TCR+age)
sex_cols  = np.arange(SEX_START, SEX_END)  # 15407, 1540

#Target (diagnosis lables)
y_all = cell_data['status'].to_numpy(np.int64)
class_labels = np.unique(y_all)
weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=y_all)
weights_tensor = torch.tensor(weights, dtype=torch.float32)
weights_tensor
#%%

# =============== Model ===============
class Classification(nn.Module):
    def __init__(self, d_model: int, n_cls: int, dropout_rate: float, nlayers: int = 1, activation=nn.ReLU):
        super().__init__()
        layers = []
        for _ in range(nlayers - 1):
            layers.append(nn.Linear(d_model, d_model))
            layers.append(activation())
            layers.append(nn.LayerNorm(d_model))
            layers.append(nn.Dropout(dropout_rate))
        self.backbone = nn.Sequential(*layers)     # stores the hidden layers in structured way
        self.out = nn.Linear(d_model, n_cls)

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        return self.out(x)

class PatientClassifier(L.LightningModule):
    def __init__(self, d_model: int, n_cls: int, dropout_rate: float, nlayers: int = 1,
                 lr: float = 1e-3,                                              
                 label_smoothing: float = 0.0,        # To make sure model does not overfit (acts on the loss/targets)                
                 weight_decay: float = 0.0,
                 class_weights:float=0.0):           # L2 reg -prevents overfittings (acts on model parameters (w))
        super().__init__()
        self.save_hyperparameters()
        self.model = Classification(d_model=d_model, n_cls=n_cls, dropout_rate=dropout_rate, nlayers=nlayers)
        self.criterion = nn.CrossEntropyLoss(weight=weights_tensor,label_smoothing=label_smoothing)
        self.multiROC = MulticlassAUROC(num_classes=n_cls,average='macro')
        self.multiPR = MulticlassAUPRC(num_classes=n_cls,average='macro')
        self.accuracy   = Accuracy(task='multiclass', num_classes=n_cls)
        self.f1         = F1Score(task='multiclass', num_classes=n_cls, average='macro')   
        self.prec_macro = Precision(task='multiclass', num_classes=n_cls, average='macro') 
        self.rec_macro  = Recall(task='multiclass', num_classes=n_cls, average='macro')   


    def on_fit_start(self):                                           

        if isinstance(self.criterion, nn.CrossEntropyLoss) and self.criterion.weight is not None:
            self.criterion.weight = self.criterion.weight.to(self.device)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        loss = self.criterion(logits, y)
        acc = self.accuracy(logits, y)
        f1  = self.f1(logits, y)
        prec=self.prec_macro(logits,y)
        recall=self.rec_macro(logits,y)

        self.log("train_loss_pat", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc_pat",  acc,  on_epoch=True, prog_bar=True)
        self.log("train_f1_pat",   f1,   on_epoch=True, prog_bar=True)
        self.log("train_prec_macro", prec, prog_bar=False)
        self.log("train_rec_macro",  recall,  prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        probs = logits.softmax(-1)
        preds=probs.argmax(dim=-1)

        val_loss = self.criterion(logits, y)
        acc_val = self.accuracy(logits, y)
        f1_val  = self.f1(logits, y)
        prec=self.prec_macro(logits,y)
        recall=self.rec_macro(logits,y)
        self.multiROC.update(probs,y)
        self.multiPR.update(probs,y)

        self.log("val_loss_pat", val_loss, prog_bar=True)
        self.log("val_acc_pat",  acc_val,  prog_bar=True)
        self.log("val_f1_pat",   f1_val,   prog_bar=True)
        self.log("val_prec_macro", prec, prog_bar=False)
        self.log("val_rec_macro",  recall,  prog_bar=False)
        return val_loss
    
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

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        probs = logits.softmax(-1)
        preds=probs.argmax(dim=-1)

        self._test_preds.append(preds.detach().cpu())
        self._test_targets.append(y.detach().cpu())

        acc = self.accuracy(logits, y)
        f1  = self.f1(logits, y)
        self.multiROC.update(probs, y)
        self.multiPR.update(probs, y)
        prec = self.prec_macro(logits, y)   # <- macro precision
        rec  = self.rec_macro(logits, y)  

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc",  acc,  prog_bar=True)
        self.log("test_f1",   f1,   prog_bar=True)
        self.log("test_prec_macro",  prec, prog_bar=False)
        self.log("test_rec_macro",   rec,  prog_bar=False)
        return loss

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

class PatientDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)
#%%
# ---- helper: consistent Trainer  with safe defaults ----
def make_trainer(max_epochs, logger=None, use_early_stopping=True, precision=16,monitor='val_auroc',mode='max',patience=7,extra_callbacks=None):
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
    return L.Trainer(
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
#%%
# =============== Nested CV with Optuna ===============
n_outer = 5
outer_cv = StratifiedKFold(n_splits=n_outer,shuffle=True, random_state=SEED)
num_classes = len(np.unique(y_all))

#%%

outer_acc = []
outer_auroc=[]
outer_metrics = {"acc": [], "f1": [], "auroc": [], "auprc": []}
#outer loops CV
for fold_idx, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(X_all, y_all), start=1):
    print(f"\nOuter Fold {fold_idx}/{n_outer}")

    X_outer_train, X_outer_test = X_all[outer_train_idx], X_all[outer_test_idx]
    y_outer_train, y_outer_test = y_all[outer_train_idx], y_all[outer_test_idx]

    def objective(trial):
        lr           = trial.suggest_float('lr', 1e-4, 3e-3, log=True)
        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
        #max_epochs   = trial.suggest_int('max_epochs', 10, 30)
        batch_size=trial.suggest_categorical('batch_size',[4,8,16])
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)

        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED) # Avoids noise
        inner_scores = []
        #Inner loop CV
        for tr_idx, va_idx in inner_cv.split(X_outer_train, y_outer_train):
            X_tr, X_va = X_outer_train[tr_idx], X_outer_train[va_idx]
            y_tr, y_va = y_outer_train[tr_idx], y_outer_train[va_idx]


            scaler = StandardScaler()
            X_tr=X_tr.copy()
            X_va=X_va.copy()
            X_tr[:,cont_cols] = scaler.fit_transform(X_tr[:,cont_cols])
            X_va[:,cont_cols] = scaler.transform(X_va[:,cont_cols])

            train_loader = DataLoader(
                PatientDataset(X_tr, y_tr),
                batch_size=batch_size, shuffle=True,
                num_workers=0, pin_memory=False, persistent_workers=False 
            )
            val_loader   = DataLoader(
                PatientDataset(X_va, y_va),
                batch_size=batch_size, shuffle=False,
                num_workers=0, pin_memory=False, persistent_workers=False  
            )

            model = PatientClassifier(
                d_model=X_tr.shape[1],
                n_cls=num_classes,
                dropout_rate=dropout_rate,
                nlayers=1,
                lr=lr,          
                label_smoothing=0.05,                  
                weight_decay=weight_decay             
            )

            #logging during search
            mlf_logger = MLFlowLogger(
                experiment_name='NN_cv_optuna',
                run_name=f"outer{fold_idx}_trial{trial.number}",
                tracking_uri='file:./mlruns'
            )

            trainer = make_trainer(max_epochs=80, logger=mlf_logger,use_early_stopping=True)

#guarantees each trial/fold leaves a “clean slate” before the next one start (cleans up GPU memory)
            try:
                trainer.fit(model, train_loader, val_loader)
                metrics = trainer.validate(model, val_loader, verbose=False)
                inner_scores.append(float(metrics[0]["val_auroc"]))
            finally:
                # full cleanup after each inner fold  
                try:
                    trainer.teardown()
                except Exception:
                    pass
                del model, trainer, train_loader, val_loader, scaler
                torch.cuda.empty_cache()
                gc.collect()

        return float(np.mean(inner_scores))

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    best_params = study.best_trial.params
    print(f"Best params (outer fold {fold_idx}): {best_params}")

    # ---- Retrain on full outer-train with best params; evaluate on outer-test
    scaler = StandardScaler()
    X_outer_train=X_outer_train.copy()
    X_outer_test=X_outer_test.copy()
    X_outer_train[:, cont_cols] = scaler.fit_transform(X_outer_train[:,cont_cols])
    X_outer_test[:, cont_cols]  = scaler.transform(X_outer_test[:, cont_cols])

    print(X_outer_train[:, cont_cols].mean(), X_outer_train[:, cont_cols].std())
    print([np.unique(X_outer_train[:, c]) for c in np.atleast_1d(sex_cols)])


    train_loader = DataLoader(
        PatientDataset(X_outer_train, y_outer_train),
        batch_size=best_params['batch_size'], shuffle=True,
        num_workers=0, pin_memory=False, persistent_workers=False  
    )
    test_loader  = DataLoader(
        PatientDataset(X_outer_test,  y_outer_test),
        batch_size=best_params['batch_size'], shuffle=False,
        num_workers=0, pin_memory=False, persistent_workers=False  
    )

    best_model = PatientClassifier(
        d_model=X_outer_train.shape[1],
        n_cls=num_classes,
        dropout_rate=best_params['dropout_rate'],
        nlayers=1,
        lr=best_params['lr'],
        label_smoothing=0.05,
        weight_decay=best_params['weight_decay']
    )

    trainer = make_trainer(max_epochs=80, logger=None,use_early_stopping=False)  
# cleanup(GPU) after each outer fold
    try:
        trainer.fit(best_model, train_loader)
        test_metrics = trainer.test(best_model, test_loader, verbose=False)[0]
    finally:
        try:
            trainer.teardown()
        except Exception:
            pass
        del best_model, trainer, train_loader, test_loader, scaler
        torch.cuda.empty_cache()
        gc.collect()

    outer_metrics["acc"].append(float(test_metrics.get("test_acc")))
    outer_metrics["f1"].append(float(test_metrics.get("test_f1")))
    outer_metrics["auroc"].append(float(test_metrics.get("test_auroc")))
    outer_metrics["auprc"].append(float(test_metrics.get("test_auprc")))

    outer_acc.append(float(test_metrics['test_acc']))
    outer_auroc.append(float(test_metrics['test_auroc']))

    print(
        f"Outer fold {fold_idx} "
        f"acc: {test_metrics.get('test_acc', float('nan')):.4f} | "
        f"F1: {test_metrics.get('test_f1', float('nan')):.4f} | "
        f"ROC-AUC: {test_metrics.get('test_auroc', float('nan')):.4f} | "
        f"PR-AUC: {test_metrics.get('test_auprc', float('nan')):.4f}"
    )
# ---- Final performance
outer_acc = np.array(outer_acc, dtype=float)
outer_auroc = np.array(outer_auroc, dtype=float)

print(f"\nFinal Nested CV Accuracy: {outer_acc.mean():.4f} ± {outer_acc.std():.4f}")
print(f"\nFinal Nested CV AUROC: {outer_auroc.mean():.4f} ± {outer_auroc.std():.4f}")
