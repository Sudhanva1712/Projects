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
adata=sc.read_h5ad('/home/skshastry/Project/rosmap_brain_blood/GSE226602_rna_raw_counts.h5ad')
adata.obs
#%%

adata.var['features']=adata.var_names
obs_to_save=['ident']
adata.var['features']
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
adata.write_h5ad("cell_embeddings_testing.h5ad")
#%%
cell_emb=sc.read_h5ad("/home/skshastry/Project/rosmap_brain_blood/cell_embeddings_testing.h5ad")
cell_emb
#%%
emb_df=pd.DataFrame(
    cell_emb.X,
    index=cell_emb.obs_names,
    columns=[f"Dim{i+1}" for i in range(cell_emb.X.shape[1])]
)
emb_df
#%%
emb_df[['patient_id','cell_type','Sex','Age','status']]=cell_emb.obs[['patient_id','full_clustering','Sex_binary','Age_interval','Status_on_day_collection_summary']]

#%%
emb_df = (
    emb_df
    .reset_index()                                     # turns the index into a column named “index”
    .rename(columns={'index': 'cell_id'})             # rename “index” → “cell_id”
)
emb_df
count=emb_df.groupby('status')['patient_id'].nunique()
emb_df
#%%
emb_df.to_csv('cell_embeddings_testing.csv',index=False)
#%%