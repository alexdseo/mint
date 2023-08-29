# Modeling

## Pipeline

Here we explain how to run the MINT pipeline, to make the food category and nutrition density score prediction using only the menu item's name.

- `utils.py`: 
> This Python script includes a training-test split for multiple folds, and the equation to calculate nutrition density scores, $RRR$ and $RRR-macro$.
- `clustering.py`: 
> This Python script includes the functions to cluster the sentence embeddings created by running [`create_embeddings.py`](https://github.com/alexdseo/mint/blob/main/data/create_embeddings.py). Therefore, [`create_embeddings.py`](https://github.com/alexdseo/mint/blob/main/data/create_embeddings.py) needs to be run prior to running this script.  We use a combination of UMAP and HDBSCAN to cluster them, where you can also try different hyperparameters to see the difference in results by exploring the tuning option. The clustering example for one of the folds is shown below.
- `nn_architectures.py`: 
> This Python script includes the neural network architectures for the food category prediction model and nutrition prediction model. These models will be called in `models.py`.
- `models.py`:
> This Python script includes the functions to call the MINT and its comparisons used in our ablation study. `clustering.py` needs to be run prior to running this script.

> 2 arguments required: 'nutrition density score', either RRR or RRR_m1. 'folds', from kf1 to kf5 that user want to test on.

![clustering example](https://github.com/alexdseo/mint/blob/main/figures/clustering.png)
