#%%
import pandas as pd
import numpy as np
#=========================================scRNA data processing===============================
#%%
data=pd.read_csv("/home/skshastry/Project/rosmap_brain_blood/data/final_input.csv",index_col=0)
# %%
tcr_data=data[['patient_id','tcr_emb','cell_type']]
#%%
tcr_data_filtered=tcr_data[tcr_data['tcr_emb'].apply(lambda x: x!='0')]
#%%
tcr_data_filtered.reset_index(inplace=True)
tcr_data_filtered=tcr_data_filtered.drop(columns=['index'])
#%%
# Group by patient_id and create separate lists for 'tcr_emb' and 'cell_type'
tcr_emb_list = tcr_data_filtered.groupby('patient_id')['tcr_emb'].apply(list).reset_index(name='tcr_emb_list')
cell_type_list = tcr_data_filtered.groupby('patient_id')['cell_type'].apply(list).reset_index(name='cell_type_list')

# Merge both lists (tcr_emb and cell_type) into a single DataFrame
merged_data_tcr = pd.merge(tcr_emb_list, cell_type_list, on='patient_id')

#%%

sc_data=data.drop(columns=['tcr_emb','cell_id'])
sc_emb_list=sc_data.groupby('patient_id')['embedding_list'].apply(list).reset_index(name='sc_emb_list')
sc_cell_list=sc_data.groupby('patient_id')['cell_type'].apply(list).reset_index(name='cell_type_list')
# %%
merged_scRNA=pd.merge(sc_emb_list,sc_cell_list, on='patient_id')
#%%

metadata=pd.read_csv('/home/skshastry/Project/rosmap_brain_blood/data/metadata.csv',index_col=0)
meta_scRNA=merged_scRNA.merge(metadata,on='patient_id',how='inner')
meta_scRNA.to_json('/home/skshastry/Project/rosmap_brain_blood/data/MIL_scRNA.json',orient='records', lines=True)
# %%
merged_data_tcr.to_json('/home/skshastry/Project/rosmap_brain_blood/data/MIL_tcr.json',orient='records', lines=True)
# %%
