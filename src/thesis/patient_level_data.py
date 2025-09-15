#%%
import pandas as pd
import numpy as np
#%%
data=pd.read_csv("/home/skshastry/Project/rosmap_brain_blood/data/final_input.csv",index_col=0)
#%%
tcr=pd.read_csv('/home/skshastry/Project/rosmap_brain_blood/data/tcr_processed/tcr_valid_embeddings_with_sample_id.csv')
tcr
#%%
#-----------------------------------------scRNA embeddings-----------------------------------
data_scRNA=data.drop(columns=['tcr_emb'])
#%%
data_scRNA['embedding_list']=data_scRNA['embedding_list'].apply(lambda x: eval(x) if isinstance(x,str) else x)

#%%
unique_cell_type=data_scRNA['cell_type'].unique()
# cell_type_count=data_scRNA.groupby(['patient_id','cell_type']).size().unstack(fill_value=0)
# cell_type_count.to_csv('/home/skshastry/Project/rosmap_brain_blood/data/count.csv')
#%%

def cell_type_mean_embeddings(df_patient):
    cell_type_mean_embedding = {}

    for cell_type in unique_cell_type:
        # Filter for cell type
        subset = df_patient[df_patient['cell_type'] == cell_type]

        # Mean cell embedding
        if not subset.empty:
            cell_emb_stack = np.stack(subset['embedding_list'])
            cell_emb_mean = cell_emb_stack.mean(axis=0)
            count = len(subset)
        else:
            cell_emb_mean = np.zeros(512)
            count = 0
        # Combine all
        combined = np.concatenate([
            cell_emb_mean,         # (512,)
            [count]                # (1,)
        ])
        cell_type_mean_embedding[cell_type] = combined

    # Stack to final [30, 531]
    return np.stack([cell_type_mean_embedding[ct] for ct in unique_cell_type])

#%%
# Group and apply
patient_matrices = data_scRNA.groupby("patient_id").apply(cell_type_mean_embeddings)
''
#%%
patient_arrays=np.stack(patient_matrices.tolist())
#patient_df=patient_arrays.reshape(patient_arrays.shape[0],-1)
patient_arrays
#%%
patient_ids=patient_matrices.index
feature_list = [row.tolist() for row in patient_arrays]

df_patient=pd.DataFrame({'patient_id':patient_ids,'cell_emb':feature_list})
#%%
df_patient.to_json("/home/skshastry/Project/rosmap_brain_blood/data/cell_type_embd.json", orient='records', lines=True)

#%%

#----------------------------------TCR embeddings----------------------------------------------------------
data_tcr=data.drop(columns=['embedding_list'])
import ast

def clean_tcr_emb(x):
    try:
        if isinstance(x, str):
            x = ast.literal_eval(x)
        x = np.array(x)
        if x.shape == (16,) and np.issubdtype(x.dtype, np.number):
            return x
    except:
        pass
    return np.zeros(16)  # fallback for invalid entries

data_tcr['tcr_emb'] = data_tcr['tcr_emb'].apply(clean_tcr_emb)

#%%
tcr_df=data_tcr.groupby('patient_id')['tcr_emb'].apply(
    lambda emb:np.mean(np.stack(emb),axis=0)
).reset_index()
tcr_df['tcr_emb'] = tcr_df['tcr_emb'].apply(lambda x: x.tolist())

#%%
tcr_df.to_json("/home/skshastry/Project/rosmap_brain_blood/data/tcr_embeddings.json", orient='records', lines=True)
#%%
#------------------------------------METADATA---------------------------------------------

meta=pd.read_csv('/home/skshastry/Project/rosmap_brain_blood/data/patient_meta/patient_metadata.csv')
meta
#%%
meta=meta.rename(columns={
    'DA':'patient_id',
    'diagnosis (0=control, 1=MCI, 2=Mild dementia)':'diagnosis',
    'Gender (0=F, 1=M)':'Gender'

})
meta=meta[['patient_id','diagnosis','Gender','Age']]
meta
#%%
meta['patient_id']=meta['patient_id'].str.extract(r'(DA-\d{4})')
# %%
meta.to_csv('/home/skshastry/Project/rosmap_brain_blood/data/metadata.csv')
# %%
