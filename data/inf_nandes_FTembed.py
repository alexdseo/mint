import pandas as pd
import numpy as np
from tqdm import tqdm
import fasttext

if __name__ == "__main__":
    # Set seed
    np.random.seed(1996)
    # # batch numbers
    # batch = [str(x) for x in list(range(1, 12))]
    # Read trained model
    model = fasttext.load_model("model_nir_nopp.bin")
    # Get embeddings
    df = pd.read_csv('edamam_inference_nandes_sample.csv', low_memory=False, lineterminator='\n')
    food_name = list(df['name'])
    # tokenize with fasttext
    word_tokens = [fasttext.tokenize(sent) for sent in food_name]
    # get the embeddings
    ft_we = list()
    for menu in word_tokens:
        tmp = []
        for w in menu:
            try:
                tmp.append(model.get_word_vector(w))
            except KeyError:
                continue
        tmp = np.mean(tmp, axis=0)
        ft_we.append(tmp)
    # array
    ft_we = np.array(ft_we)
    # export
    np.save('edamam_inference_nandes_FT_embedding.npy', ft_we)
