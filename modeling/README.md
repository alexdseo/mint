# Modeling

## Pipeline

Here we explain how to run the MINT pipeline, in order to make the food category and nutrition desnity score prediction using only menu item's name.

- `utils.py`: 
> This python script includes traing-test split for multiple folds, and the equation to calculate nutrition density scores, $RRR$ and $RRR-macro$.
- `clustering.py`: 
> This python script includes the functions to cluster the sentence embeddings created by running [`create_embeddings.py`](https://github.com/alexdseo/mint/blob/main/data/create_embeddings.py). Therefore, [`create_embeddings.py`](https://github.com/alexdseo/mint/blob/main/data/create_embeddings.py) needs to be run prior to running this script.  We use combination of UMAP and HDBSCAN to cluster them, where you can also try different hyperparmeters to see the difference in results by exploring tuning option. The clustering example for on of the folds is shown below.
- `nn_architectures.py`: 
> This python script includes the neural network architectures for food category prediction model and nutrition prediction model. These models will be called in `models.py`.
- `models.py`:
> This python script includes the functions to call the MINT and its comparisons that were used in our ablation study. `clustering.py` needs to be run prior to running this script. 

![clustering example](https://github.com/alexdseo/mint/blob/main/figures/clustering.png)
