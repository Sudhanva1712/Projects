#%%
from tcrvalid.load_models import *
from tcrvalid.physio_embedding import SeqArrayDictConverter
import pandas as pd
#%%
#get the model for TRB
model_name='1_2_full_40'
model = load_named_models(model_name,chain='TRB',as_keras=True)[model_name]
#%%
# get mapping object to convert protein sequence 
# to physicochemical representation
mapping = SeqArrayDictConverter()
#%%
# convert seq to physicochemical - list could have many sequences
# sequences are "CDR2-CDR3" formatted
df=pd.read_csv("/home/skshastry/Project/rosmap_brain_blood/data/tcr_valid_input_trb.csv")
mask = df['cdr3'].str.len() <= 28

# Filter df before feeding sequences to the embedding
filtered_df = df[mask].reset_index(drop=True)

# Now use only these for embedding:
sequences_short = filtered_df['cdr3'].tolist()


#%%
x = mapping.seqs_to_array(sequences_short,maxlen=28)
#%%
# get TCR-VALID representation
z,_,_ = model.predict(x)
#%%
emd_df=pd.DataFrame(z)
emd_df['sample_id']=filtered_df['sample_id'].values
emd_df['cell_id']=filtered_df['barcode'].values

#%%
mm=emd_df['sample_id']=='DA-1381'
emd_df.loc[mm,['cell_id','sample_id']].shape
#%%
emd_df.to_csv("tcr_valid_embeddings_with_sample_id.csv", index=False)

#%%