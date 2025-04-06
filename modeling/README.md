# Modeling

The model pipeline consists of 3 steps, first producing embeddings through trained food-specific language model (run `../data/embeddings.py`), then clustering the training data to create the food category pseudo-labels. 

Create food category pseudo-labels on the training dataset with option for hyperparameter tuning (`False` uses tuned parameters; `True` runs the tuning process):
```
python clustering.py False
```

After clustering process, train MINT on desired nutrient density score (options: RRR, NRF9.3, NRF6.3, LIM, WHO, FSA). Choose one of the 5 folds (options: kf1, kf2, kf3, kf4, kf5) to test on for cross-validation and reproducing results in the manuscript. PyTorch and Tensorflow both available to use for training MINT:
```
python training_torch.py RRR kf1
python training_tf.py RRR kf1
```

<!-- Confidence interval and error analysis for checking the robustness of MINT predictions are included in this [notebook](https://gist.github.com/alexdseo/27babc1fc313d412630bf07b54b64c2f). -->
