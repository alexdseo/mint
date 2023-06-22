import pandas as pd
import json
import fasttext
from tqdm import tqdm
import numpy as np
import string
from sentence_transformers import SentenceTransformer


def get_text(recipe_text):
    """
    Get text of interest from Recipe 1M+ dataset

    Args:
        recipe_text: Raw Recipe1M+ dataset

    Returns:
        lst_n: A list of food names from the Recipe1M+ dataset
        lst_ni: A list of food names and its ingredients from the Recipe1M+ dataset
        lst_nir: A list of food names, its ingredients, and its recipes from the Recipe1M+ dataset
    """
    # Define list to store the text
    lst_n, lst_ni, lst_nir = list(), list(), list()
    for i in tqdm(range(len(recipe_text)), desc='Getting text...'):
        # Initialize ingredients and recipe place holder
        ingr, recipe = '', ''
        # Save just name
        lst_n.append(recipe_text[i]['title'])
        # Get ingredients of each item
        for j in range(len(recipe_text[i]['ingredients'])):
            ingr += recipe_text[i]['ingredients'][j]['text']
        # Save name and ingredients
        lst_ni.append(recipe_text[i]['title'] + ' ' + ingr)
        # Get recipe of each item
        for k in range(len(recipe_text[i]['instructions'])):
            recipe += recipe_text[i]['instructions'][k]['text']
        # Save name, ingredients, and recipe
        lst_nir.append(recipe_text[i]['title'] + ' ' + ingr + recipe)

    return lst_n, lst_ni, lst_nir


def train_fasttext(txt_file, edamam_df):
    """
    Get text of interest from Recipe 1M+ dataset

    Args:
        txt_file: Retrieved texts from Recipe1M+, default as names, ingredients, and recipe. Can change to
                  different dataset
        edamam_df: Main dataset that we want to train fasttext with

    Returns:
        ft_we: A numpy array of average word embeddings of each menu items
    """
    model = fasttext.train_unsupervised(txt_file, dim=300, epoch=20)
    # Save model
    model.save_model("model_nir_nopp.bin")
    # Load model
    # model = fasttext.load_model("model_nir_nopp.bin")
    # Get names from Edamam dataset
    food_name = list(edamam_df['Name'])
    # Tokenize with fasttext
    word_tokens = [fasttext.tokenize(sent) for sent in tqdm(food_name)]
    # Get average word embeddings of the menu items
    ft_we = list()
    for menu in tqdm(word_tokens):
        tmp = list()
        for w in menu:
            try:
                # Get each word embeddings
                tmp.append(model.get_word_vector(w))
            except KeyError:
                continue
        # Get average of the word embeddings
        tmp = np.mean(tmp, axis=0)
        ft_we.append(tmp)
    # Make it numpy array
    ft_we = np.array(ft_we)

    return ft_we


def menu_with_ingr(df):
    """
       Append food names and ingredients from Edamam dataset

       Args:
           df: Edamam ingredients dataset

       Returns:
           lst_txt: Appended texts from Edamam
       """
    # Save text in human-readable form
    lst_txt = list()
    for i in tqdm(range(len(df)), desc='Getting text...'):
        txt = df['Name'][i]
        # Append ingredients to food name
        if df['Ingredients_only'][i] is not np.nan:
            txt += ' made with ' + df['Ingredients_only'][i]
        # Preprocess to delete punctuation
        txt = ''.join(txt).translate(str.maketrans('', '', string.punctuation))
        lst_txt += [txt]

    return lst_txt


def get_setence_embeddings(name_with_ingredients):
    """
       Get sentence embeddings of names and ingredients from Edamam using pretrained MPNET

       Args:
           name_with_ingredients: Appended texts of names and ingredients from Edamam
       Returns:
           mpnet_embeddings: A numpy array of sentence embeddings from pretrained MPNET
       """
    # Download pretrained model
    mpnet = SentenceTransformer('all-mpnet-base-v2')
    # SBERT embeddings #Run with gpu
    mpnet_embeddings = mpnet.encode(name_with_ingredients)

    return mpnet_embeddings


if __name__ == "__main__":
    # Set seed
    np.random.seed(2023)
    # Read Edamam dataset
    menu_nutri = pd.read_csv('edamam_nutrition_sample.csv')
    menu_ingr = pd.red_csv('edamam_ingredients_sample.csv')

    # Read Recipe 1M+ dataset # You can get access this data through their project website
    recipe1m = json.load(open('layer1.json'))
    names, names_ingr, names_ingr_recipe = get_text(recipe1m)
    # Write in text # Choose text with names, ingredients, and recipe # You can change to different text as well
    with open('train_nir_nopp.txt', 'w') as f:
        for line in names_ingr_recipe:
            f.write(f"{line}\n")

    # Train fasttext model
    edamam_ft_we = train_fasttext('train_nir_nopp.txt', menu_nutri)
    # Save the embeddings
    np.save('edamam_menu_embedding.npy', edamam_ft_we)

    # Get Edamam menu with ingredients
    edamam_name_ingr_txt = menu_with_ingr(menu_ingr)
    # Get sentence embeddings from pretrained mpnet
    edamam_setence_embeddings = get_setence_embeddings(edamam_name_ingr_txt)
    # Save the embeddings
    np.save('edamam_sentence_embeddings_mpnet.npy', edamam_setence_embeddings)
