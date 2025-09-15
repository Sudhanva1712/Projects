#%%
import scanpy as sc
import torch
import pandas as pd
from scgpt.tasks.cell_emb import get_batch_cell_embeddings,embed_data
#%%
import torch, torchtext
print(torch.__version__, torchtext.__version__)
print(torch.cuda.is_available())
#%%

#%%
adata=sc.read_h5ad('/home/skshastry/Project/rosmap_brain_blood/data/scRNA_emb/combined_data.h5ad')

#%%
adata.var['features']=adata.var.index
obs_to_save=['orig.ident','predicted.celltype.l2']
adata.obs
#%%
adata=embed_data(
    adata_or_file=adata,
    model_dir='/home/skshastry/Project/rosmap_brain_blood/pretrained_model',
    gene_col='features',
    max_length=3000,
    batch_size=64,
    obs_to_save=obs_to_save, 
    device='cuda',
    use_fast_transformer=False,
    return_new_adata=True
)
#%%
adata.write_h5ad("cell_embeddings.h5ad")
#%%
cell_emb=sc.read_h5ad("/home/skshastry/Project/rosmap_brain_blood/data/cell_embeddings.h5ad")
cell_emb.X
#%%
emb_df=pd.DataFrame(
    cell_emb.X,
    index=cell_emb.obs.index,
    columns=[f"Dim{i+1}" for i in range(cell_emb.X.shape[1])]
)
#%%
emb_df[['orig.ident','cell_type']]=cell_emb.obs[['orig.ident','predicted.celltype.l2']]

emb_df = (
    emb_df
    .reset_index()                                     # turns the index into a column named “index”
    .rename(columns={'index': 'cell_id'})             # rename “index” → “cell_id”
)
#%%
emb_df.to_csv('cell_embeddings.csv',index=False)
#%%