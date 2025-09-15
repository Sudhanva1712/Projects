from torch import nn, Tensor
import torch

class GatedAttentionMIL(nn.Module):
    """m
    Gated attention MIL.
    H: (B, T, D) 
    mask: [B,T] True if T present
    Returns:
        -> bag Z: (B, D) bag embedding, attention A: (B, T)
    """
    def __init__(self,dim:int, attn_dim: int = 64):
        super().__init__()
        self.V = nn.Linear(dim, attn_dim, bias=True)
        self.U = nn.Linear(dim, attn_dim, bias=True)
        self.w = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, H: Tensor):
        Vh = torch.tanh(self.V(H))            # (B,T,D) (captures the content)
        Uh = torch.sigmoid(self.U(H))         # (B,T,D) (acts like a gate(decides which parts are important))
        scores = self.w(Vh * Uh).squeeze(-1)  # (B,T) (cell type imp for the bag)
        A = torch.softmax(scores, dim=1)        # (B,T) attention scores for each cell type
        Z = torch.einsum('bth,bt->bh', H, A)  # (B,D) weigthed sum pooling per pateint
        return Z, A
#%%
#========================classification model=============================
    
class Classification(nn.Module):
    def __init__(self, d_model: int, n_cls: int, dropout_rate: float,
                 nlayers: int , activation=nn.ReLU):
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
#========================Logistic-Regression=============================                 

class LogisticRegression(nn.Module):
    def __init__(self,d_model:int,n_cls:int):
        super().__init__()
        self.linear = nn.Linear(d_model, n_cls) 
    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)