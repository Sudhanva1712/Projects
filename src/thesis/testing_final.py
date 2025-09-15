#%%
import scanpy as sc
import numpy as np
import pandas as pd
#%%
# adata = sc.read_h5ad('/home/skshastry/Project/rosmap_brain_blood/haniffa21.processed.h5ad')
# rna = adata[:, adata.var['feature_types'] == 'Gene Expression'].copy()
# rna.layers['raw'].data
# rna.layers['counts'] = rna.layers['raw'].copy()
# rna.X = rna.layers['counts'].copy()
# del rna.layers['raw']
# #%%
# sc.experimental.pp.highly_variable_genes(rna, n_top_genes=2000, batch_key='Site')
# #%%
# rna = rna[:, rna.var.highly_variable].copy()
# rna
# #%%
# rna.obs['Status_on_day_collection_summary'].cat.categories
# drop_conditions = ['LPS_10hours', 'LPS_90mins', 'Non_covid','Asymptomatic', 'Critical','Moderate']
# rna = rna[~rna.obs['Status_on_day_collection_summary'].isin(drop_conditions)]
# #%%
# rna.obs['Sex_binary'] = (rna.obs['Sex'] == 'Male').astype(int)
# rna.obs[['Sex','Sex_binary']]
# #%%
# rna.write('/home/skshastry/Project/rosmap_brain_blood/processed.h5ad')
 #%%
data_test=pd.read_csv('/home/skshastry/Project/rosmap_brain_blood/cell_embeddings_testing.csv')
data_test_vv=data_test.copy()
# %%
embcol=[a for a in data_test_vv if a.startswith('Dim')]
data_test_vv['embedding_list']=data_test_vv[embcol].values.tolist()
#%%
data_test_vv=data_test_vv.drop(columns=embcol)
data_test_vv
# %%
unique_cell_type=data_test_vv['cell_type'].unique()

# %%
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
            [count]               # (1,)
        ])
        cell_type_mean_embedding[cell_type] = combined

    # Stack to final [30, 531]
    return np.stack([cell_type_mean_embedding[ct] for ct in unique_cell_type])

patient_matrices = data_test_vv.groupby("patient_id").apply(cell_type_mean_embeddings)
patient_matrices
# %%
patient_arrays=np.stack(patient_matrices.tolist())
#patient_df=patient_arrays.reshape(patient_arrays.shape[0],-1)
patient_arrays.shape
# %%
patient_ids=patient_matrices.index
feature_list = [row.tolist() for row in patient_arrays]

df_patient=pd.DataFrame({'patient_id':patient_ids,'cell_emb':feature_list})
#%%
metadata=data_test.copy()
#%%

metadata.columns

# %%
# Select relevant columns
meta = metadata[['patient_id', 'cell_type', 'Sex', 'Age', 'status']]

#%%
# Group by patient_id and aggregate
dd = meta.groupby('patient_id').agg({  # or mode if needed
    'Sex': 'first',                           # or mode
    'Age': 'first',                           # or median
    'status': 'first'                         # patient-level label
})
import re

final =df_patient.merge(dd,on='patient_id',how='inner')

def parse_interval_midpoint(age_str):
    match = re.findall(r'\d+', str(age_str))  # Extract all numbers
    if len(match) == 2:
        low, high = map(int, match)
        return (low + high) / 2
    return None

final['Age_numeric'] = final['Age'].apply(parse_interval_midpoint)
#%%
final.to_json('/home/skshastry/Project/rosmap_brain_blood/test_data_final.json')
# %%
