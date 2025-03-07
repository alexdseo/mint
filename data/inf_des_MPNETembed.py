import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def name_des(df):
    # Save only text that is human readable
    lst_text = []
    for i in range(len(df)):
        text = df['name'][i]
        if df['description'][i] is not np.nan:
            text += '. ' + df['description'][i]  # add . after the name, then description
        else:
            print(i, 'nan here')
        # Append to list
        lst_text += [text]

    return lst_text

if __name__ == "__main__":
    # Set seed
    np.random.seed(1996)
    # # batch numbers
    # batch = [str(x) for x in list(range(1, 12))]
    # Load pre-trained model
    # SBERT # Use mpnet
    sbert = SentenceTransformer('all-mpnet-base-v2', device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    # Get embeddings
    df = pd.read_csv('edamam_inference_des_sample.csv', low_memory=False, lineterminator='\n')
    # get name and description
    df_nd = name_des(df)
    # SBERT embeddings #Run with gpu
    embeddings_mpnet = sbert.encode(df_nd)
    # export
    np.save('edamam_inference_des_MPNET_embedding.npy', embeddings_mpnet)