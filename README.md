# Predicting the Nutritional Quality of a Food Environments

MINT: Menu Item to NutrienT is a new pipeline to capture the nutritional quality of a restaurant through menu item names. This repository contains code to reproduce the results using MINT that was introduced in our paper "What’s On the Menu? Towards Predicting Nutritional Quality of Food Environments", submitted to ICDE 2024.

## Description

Unhealthy diets are a leading cause of major chronic diseases including obesity, diabetes, cancer, and heart disease. Food environments--the physical spaces in which people access and consume food--have the potential to profoundly impact diet and related diseases. While large chain restaurants are required to report the nutrient composition of their menu items, the vast majority of restaurants do not, creating a substantial knowledge gap. We take a step to address this problem by developing MINT: Menu Item to NutrienT model. This model utilizes a novel data source on generic food items, along with state-of-the-art word embedding and deep learning methods, to predict the nutritional density of never-before-seen menu items based only on their name text with $R^2=0.77$. Details of our model MINT is described below:


![MINT pipeline](https://github.com/alexdseo/mint/blob/main/figures/model_diagram.png)

(a) Word embedding model. We begin by extracting food names, ingredients, and recipes from the Recipe1M+ dataset, and use the concatenation of this text to train a FastText word embedding model. The menu item embedding is the average embedding of each word in the menu item. 

(b) Food category prediction model. We embed Edamam training data containing menu items concatenated with item ingredients using a pre-trained MPNet model. We then cluster the training data using HDBSCAN, which we treat as a ground truth food category. We then use the FastText model to embed food items alone and train a model to predict the most likely cluster associated with each food name. 

(c) Nutrition Score Model. The Edamam dataset is used to train a nutrition score model in which food names embedded with FastText, are fed into models to predict the *Nutrient density scores*. The model first trained on the entire dataset is then fine-tuned on the ground-truth categories to create MINT.


## Installation

Our code was tested on `Python 3.9`, to install other requirements:
```setup
pip install -r requirements.txt
```

## Usage

MINT is trained with a high-quality dataset that contains generic food items - canonical foods, including everything from individual raw foods to complex meals – which includes their ingredients and nutrient composition information. For more details about the datasets used to create MINT, please see [`data`](https://github.com/alexdseo/mint/blob/main/data/README.md).

MINT consists of 2 prediction models: the food category prediction model and the nutrition quality prediction model. After getting all the datasets required to run MINT, to make predictions on nutrition quality/individual nutrients based on menu item names, please see [`modeling`](https://github.com/alexdseo/mint/tree/main/modeling/README.md). 
