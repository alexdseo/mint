# Data

## Required dataset

Here we describe about the required datasets to create MINT. The first dataset, [Recipe1M+](http://im2recipe.csail.mit.edu/), is a large-scale structured dataset containing over 1 million food names, ingredients, and recipes. We will use this dataset to train the FastText model, so we can create word embeddings for our main dataset. The full dataset(`layer1.json`) is available for access through their [website](http://im2recipe.csail.mit.edu/). 

Then we would need our main dataset, high-quality generic food items with their list of ingredients and full nutrient composition information. This dataset was curated and shared for research purposes by the nutrition data company [Edamam Inc](https://www.edamam.com/). Edamam dataset will be used to train the nutrition prediction model using their menu item name and nutrition information, and to create sentence embeddings using their menu item name and its list of ingredients. The sentence embeddings will be created by utilizing pre-trained MPNet. The dataset can be accessed through their API or, if you need it for research purpose, contact them for the full dataset.

In this folder, we included samples of the Edamam dataset `edamam_ingredients_sample.csv`, which includes the menu items and their ingredients, and `edamam_nutrition_sample.csv`, which includes the menu items and their nutrition information.

You can run [`create_embeddings.py`](https://github.com/alexdseo/mint/blob/main/data/create_embeddings.py) to create word embeddings of each menu item (averaged by all words) from the Edamam dataset, and also to create sentence embeddings using menu item names and its list of ingredients from the Edamam dataset. Before running this file, you have to get access to the [Recipe1M+](http://im2recipe.csail.mit.edu/) dataset, in order to train the FastText model. We specifically used the `layer1.json` dataset, which includes all of the texts for each menu item. 

<!---
## Language Models

- Model weights for **RecipeFT** can be downloaded [here](https://drive.google.com/drive/folders/16yGJUie7fu2ZdIwoRbHEGyQU4uLj9jlH)
- **RecipeBERT** can be downloaded through our huggingface repository [here](https://huggingface.co/alexdseo/RecipeBERT)
--->

## Dataset for the application

After training the MINT using the `Recipe1M+` and the `Edamam` dataset, you can make an inference on the menu item's nutrition density score from different datasets. In our paper, we made a real-world application using [Spoonacular](https://spoonacular.com/food-api) dataset, which contains chain restaurant menu item's nutrient values, published by each restaurant as required by the Affordable Care 468 Act (ACA). Their dataset is available for access through their [API](https://spoonacular.com/food-api).

We then used the restaurant location data from [Los Angeles County Restaurant and Market Inventory](https://data.lacounty.gov/), which contains the locations for all registered restaurants in Los Angeles County. Some of the chain restaurants with more than 10 locations in LA county but was not listed on `Spoonacular`'s database, were manually extracted by us from each of the restaurant's official website. This dataset is listed in this folder: `extracted_menu_items.csv`.

Using these datasets, and trained MINT, you can evaluate the chain restaurant food environment -- the physical spaces in which people access and consume food-- in LA County:

![Application to LA County data](https://github.com/alexdseo/mint/blob/main/figures/heatmap.png)

For this map, the median $RRR−macro$ of the menu was used to evaluate each chain restaurant. Which, were averaged across all chain restaurants in neighborhoods to evaluate the food environment. (a) MINT restaurant quality predictions and (b) Ground truth restaurant quality.
