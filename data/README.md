# Data

## Data Preparations

MINT is trained with a high-quality dataset that contains generic food items - canonical foods, including everything from individual raw foods to complex meals – with their list of ingredients and full nutrient composition information. This dataset was curated and shared for research purposes by the nutrition data company [Edamam Inc](https://www.edamam.com/). 
The dataset is based on recipe and food composition data from multiple sources, including the U.S. Department of Agriculture's (USDA) Food Data Central, and curated to ensure the accuracy of these data. This dataset is preferred due to its unbiased representation of food items, deprived of any retailer-specific biases, and its comprehensive coverage across various food categories. The dataset can be accessed through their API, or contact the company for the full dataset, if needed for research purposes. In [`files`](https://github.com/alexdseo/mint/tree/main/data/files), samples of this training dataset are included.

We also utilize [Recipe1M+](http://im2recipe.csail.mit.edu/), a large-scale structured dataset containing over 1 million food names, ingredients, and recipes. The dataset is used to train the language models and produce embeddings that will be used to train the MINT. The full dataset(`layer1.json`) is available for access through their [website](http://im2recipe.csail.mit.edu/). Model weights for **RecipeFT**, a FastText model trained with Recipe1M+, can be downloaded [here](https://drive.google.com/drive/folders/16yGJUie7fu2ZdIwoRbHEGyQU4uLj9jlH) and **RecipeBERT**, a BERT model trained with Recipe1M+, can be downloaded through our huggingface repository [here](https://huggingface.co/alexdseo/RecipeBERT).

Produce all embeddings:
```
cd data
python embeddings.py
```

We leverage ChatGPT API to generate synthetic data labels, including AMDD (Appetizers, Main-dish, Dessert, and Drink) labels and one-line menu descriptions. This synthetic data is used to train the MINT model and classify the menu types.

Generate synthetic data (`OpenAI API key needed`):
```
python LLM_data_augmentation.py
```

## Creating Metrics

A large-scale real-world menu item dataset comprising approximately 70 million menu items from 600,000 restaurants that was sourced in 2020 from Edamam is used as the inference dataset in this project. Our preliminary investigation reveals that the dataset contains restaurants with incomplete menu entries with a minimal number of items, as well as entries from local delis and liquor stores that list an extensive inventory. Therfore, we remove 1% of the dataset from both ends of the national distribution of the menu counts per restaurant. Specifically, restaurants with fewer than 6 menu items or more than 692 menu items are discarded.

Detect outliers in the inference data:
```
python outlier_detection.py
```

After getting the prediction of the nutrient density on the inference dataset using trained MINT, we estimate restaurant-level nutrient density (RND) and food environment nutrient density (FEND). 

Estimate RND and FEND based on RRR (options: RRR, NRF9.3, NRF6.3, LIM, WHO, FSA):
```
python estimate_RND.py RRR
python estimate_FEND.py RRR
```

Our analysis workflow is shown below, along with examples of ground truth RND scores of national fast food chains.

<p align="center">
  <img src="https://github.com/alexdseo/mint/blob/main/figures/intro_fig.png" alt="workflow" style="width:750px;"/>
</p>

