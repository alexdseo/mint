<p align="center">
  <img src="https://github.com/alexdseo/mint/blob/main/figures/mint_logo.png" alt="drawing" style="width:350px;"/>
</p>

<div align="center">
  <h2>Menu Item to NutrienT</h2>
</div>

--- 

MINT: Menu Item to NutrienT is an open-source machine learning pipeline that predicts the nutritional density of a menu item through their names. MINT includes building a food-specific language model and generating ingredient-contextualized cluster labels to enhance the prediction performance. MINT employs a multi-expert system enabling specialized prediction for distinct food categories. The goal of the MINT is to predict the nutrient density, which serves as a basis for defining restaurant-level metrics, Restaurant Nutrient Density (*RND*), an aggregated nutrient density of all menu items offered by the restaurant. Extending this evaluation to the broader context of the food environment, we create metrics at the food environment level, Food Environment Nutrient Density (*FEND*), to evaluate the healthy food accessibility in the area.


## Mapping Nutritional Health through U.S. Restaurant Menus

Increasing evidences show that the food environment strongly dictates both diet and diet-related diseases. Recent efforts to modify food environments for dietary improvement have demonstrated limited success, partly due to a limited understanding of how environments affect eating behavior from inadequate characterization of their nutritional features. We leverage deep learning, language models, and vast restaurant menu data to analyze the nutritional quality of the conterminous food environment and map the nutritional health in the United States. We introduce the food nutrient density prediction model, MINT: Menu Item to NutrienT, which is applied to approximately 70 million menu items from 600,000 U.S. restaurants, assessing their nutritional quality and then creating aggregate measures to assess the nutritional quality of the food environment. These metrics reveal that environments offering healthier restaurant menus do not coincide with traditional labels of ‘food deserts’ or ‘food swamps’ and are a stronger predictor of nutritional health outcomes, including obesity, diabetes, and coronary heart disease, than the commonly-used existing measures of food environment nutritional availability. By evaluating restaurant nutritional quality at a menu-item level across restaurants throughout the U.S., this study provides insights and tools for policymakers, researchers, the food and restaurant industry, and individual consumers to support a shift towards healthier restaurant food choices.

Details of the MINT model are described below:

![MINT pipeline](https://github.com/alexdseo/mint/blob/main/figures/model_diagram.png)

**Data** We use $\approx 80,000$ generic food items as our primary training dataset. Both menu item names and menu item names concatenated with their ingredients are utilized to generate menu descriptions and for embedding purposes.

**Embeddings** We extract food names, ingredients, and recipes from the Recipe1M+ to train a FastText word embedding model, referred to as RecipeFT. RecipeFT is used to generate menu embeddings from menu item names. Additionally, we embed training data containing menu items concatenated with their ingredients or their generated descriptions using a pre-trained MPNet model. Soft clustering of ingredient embeddings is performed using HDBSCAN, with the resulting latent food clusters treated as food category training labels.

**Nutrient Density (ND) Prediction Training.** We independently train a feed-forward neural network to predict the nutrient density score with uncertainty estimates via Monte Carlo dropout using menu embeddings and description embeddings. The model is 1) pre-trained on the entire dataset and then 2) fine-tuned separately based on food category training labels, enabling it to serve as an expert model for each specific food category in predicting nutrient density scores.

**Inference.** We take $\approx 600,000$ menu items from the US restaurants and encode their name using the RecipeFT, to predict their most likely food category via neural network with softmax for classification. Based on the availability of descriptions, we determine the appropriate model (name-based or description-based) and input type to allocate each menu item to the corresponding fine-tuned expert model according to the predicted food category.


## Quickstart

1. Set up your environment. You can either use your base environment or create a conda virtual environment. Assuming anaconda is installed and using Python 3.9+:

```
conda create -n mint_env
conda activate mint_env
```

2. Set working directory:
```
git clone https://github.com/alexdseo/mint.git
cd mint
```

3. Install requirements:
```setup
conda install pip
pip install -r requirements.txt
```

## Usage

MINT uses data from multiple sources, including high-quality generic food items, large-scale recipe datasets, LLM-generated synthetic data labels, and real-world restaurant menu items in the United States. For more details on producing embeddings, generating labels, post-processing, and creating metrics using these datasets, please see [`data`](https://github.com/alexdseo/mint/tree/main/data).

MINT consists of two modeling components, clustering and food category-specific predictions. After initial clustering, we provide two options to train MINT using different deep learning frameworks, TensorFlow and PyTorch. For more details, please see [`modeling`](https://github.com/alexdseo/mint/tree/main/modeling).

Following the nutrient density prediction using MINT, we can estimate restaurant-level and food environment-level nutrient density. These metrics can be used to analyze and map the nutritional health of the United States. For more details, please see [`analysis`](https://github.com/alexdseo/mint/tree/main/analysis). 

## Community Support

We encourage users to contribute by:

Reporting issues via the GitHub Issues tab.
Submitting feature requests or suggestions for improvement.
For questions, please contact the authors at [alexdseo@isi.edu].
