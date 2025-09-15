#%%
import pandas as pd
from pathlib import Path


# Read the metadata
meta_data = pd.read_csv('/home/skshastry/Project/rosmap_brain_blood/data/patient_metadata.csv')
tcr=pd.read_csv("/home/skshastry/Project/rosmap_brain_blood/data/tcr_processed/tcr_valid_embeddings_with_sample_id.csv")
# Clean the data by removing rows with missing values in the "DA" column
meta_cleaned = meta_data.dropna(subset=["DA"])
tcr_celaned=tcr.dropna(subset=["sample_id"])

#%%
# Select only the relevant columns
meta_final = meta_cleaned[["DA", "diagnosis (0=control, 1=MCI, 2=Mild dementia)","Age","Gender (0=F, 1=M)"]]
meta_final['DA'] = meta_final['DA'].str.split(' ').str[0]

#removing a patient
meta_final = meta_final[meta_final['DA'] != 'DA-1802']
tcr_fin=tcr_celaned[tcr_celaned['sample_id'] != 'DA-1802']
tcr_fin
#%%

#renaming the columns in TCR embeddings
tcr_fin.rename(columns={
                'sample_id':'patient_id',
               'cell_id':'cell_id'
               },inplace=True)
tcr_col=[c for c in tcr_fin if c.isdigit()]
tcr_fin['tcr_emb']=tcr_fin[tcr_col].values.tolist()
tcr_fin=tcr_fin.drop(columns=tcr_col)
tcr_info=tcr_fin[['patient_id','cell_id','tcr_emb']]
tcr_final=tcr_info.copy()
#%%
# Rename the columns
meta_final.rename(columns={
    "diagnosis (0=control, 1=MCI, 2=Mild dementia)": "diagnosis",
    "DA" :"patient_id",
    "Age" : "age",
    "Gender (0=F, 1=M)": "Gender"
}, inplace=True)
meta_final


#%%
embeddings=pd.read_csv("/home/skshastry/Project/rosmap_brain_blood/data/cell_embeddings.csv")
embeddings.rename(columns={
    'orig.ident':'patient_id'
},inplace=True)
embd=embeddings.copy()
#%%
emb_cols=[a for a in embd if a.startswith('Dim')]
# 3) Create a new column that is the list of all Dim values for each row
embd['embedding_list'] = embd[emb_cols].values.tolist()

# 4) (Optional) drop the old Dim columns if you no longer need them
embd = embd.drop(columns=emb_cols)

# 5) If you just want the patient_id + list column:
res = embd[['patient_id', 'embedding_list','cell_type','cell_id']]
res["cell_id"]=(
    res['cell_id'].str.split("_",n=1,expand=True)[1]
)
final_cell_emb=res.copy()
final_cell_emb
#%%
# Merge first two DataFrames(scRNA and Metadata (cell_type,sex and age))
tmp = final_cell_emb.merge(meta_final, on="patient_id", how="left")
final_df = tmp.reset_index(drop=True)
final_df

#%%
# concatinating the TCR embeddings with cell code of T cells()
tcr_unique = tcr_final.drop_duplicates(
    subset=['patient_id','cell_id'],
    keep='first'
)
tcr_unique_fin=tcr_unique.copy()
tcr_unique_fin
#%%
# merge combined data(scRNA and metadata) with TCR embeddings
fin = final_cell_emb.merge(tcr_unique_fin, on=["cell_id",'patient_id'], how="left")
fin
#%%
# list the columns that came from tcr_fin (excluding the key)
tcr_cols = [c for c in tcr_unique.columns if c != ("cell_id",'patient_id')]

# fill NaN only in those
fin[tcr_cols] = fin[tcr_cols].fillna(0)


#%%
#writing the file to .h5ad format for scGPT input
fin.to_csv("/home/skshastry/Project/rosmap_brain_blood/data/final_input.csv")
#%%
