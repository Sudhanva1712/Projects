#%%
import pandas as pd
from pathlib import Path
import anndata as ad

# Read the metadata
meta_data = pd.read_csv('/home/skshastry/Project/rosmap_brain_blood/data/patient_metadata.csv')
tcr=pd.read_csv("/home/skshastry/Project/rosmap_brain_blood/data/tcr_valid_embeddings_with_sample_id.csv")
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

#%%

#renaming the columns in TCR embeddings
tcr_fin.rename(columns={
                'sample_id':'orig.ident',
               'cell_id':'cell_code'
               },inplace=True)

#%%
# Rename the columns
meta_final.rename(columns={
    "diagnosis (0=control, 1=MCI, 2=Mild dementia)": "diagnosis",
    "DA" :"orig.ident",
    "Age" : "age",
    "Gender (0=F, 1=M)": "Gender"
}, inplace=True)

#%%
# Define the directory where your .h5ad files are stored
all_patient=ad.read_h5ad("/home/skshastry/Project/rosmap_brain_blood/data/combined_data.h5ad")
df=all_patient.obs.index.to_series().str.split('_',expand=True)
df.columns=['sample','cell_code']
all_patient.obs=all_patient.obs.join(df)
#%%
obs_df = all_patient.obs.copy()
obs_df["cell_id"] = obs_df.index 
#%%
# Merge first two DataFrames(scRNA and Metadata (cell_type,sex and age))
tmp = obs_df.merge(meta_final, on="orig.ident", how="left")

#%%
# concatinating the TCR embeddings with cell code of T cells()
tcr_unique = tcr_fin.drop_duplicates(
    subset=['orig.ident','cell_code'],
    keep='first'
)

# merge combined data(scRNA and metadata) with TCR embeddings
fin = tmp.merge(tcr_unique, on=["cell_code",'orig.ident'], how="inner")
matched_ids = fin["cell_id"].astype(str).unique().tolist()
# 4) Subset your AnnData to **only** those cells
adata_matched = all_patient[all_patient.obs_names.isin(matched_ids)].copy()
#%%
adata_matched.obs = fin
adata_matched.obs["diagnosis"].unique()
print(adata_matched)
# 1) After all your merges and before writing:

# Force both obs and var indexes to be string dtype
adata_matched.obs.index = adata_matched.obs.index.astype(str)
adata_matched.var.index = adata_matched.var.index.astype(str)



#%%
#writing the file to .h5ad format for scGPT input
adata_matched.write_h5ad("/home/skshastry/Project/rosmap_brain_blood/data/scRNA_TCR_data.h5ad")
#%%