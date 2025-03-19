import pandas as pd
import json
import torch
import fasttext
from tqdm import tqdm
import numpy as np
import string
from transformers import pipeline
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


def train_recipeft(txt_file):
    """
    Train FastText model with Recipe 1M+ dataset to create RecipeFT

    Args:
        txt_file: Retrieved texts from Recipe1M+, default as names, ingredients, and recipe. Can change to
                  different dataset
    """
    model = fasttext.train_unsupervised(txt_file, dim=300, epoch=20)
    # Save model
    model.save_model("model_nir_nopp.bin")

def recipeft(df):
    """
    Retreive word embeddings from RecipeFT

    Args:
        df: Inference dataset
    Returns:
        ft_we: A numpy array of average word embeddings of each menu items name words
    """

    # Load model
    model = fasttext.load_model("model_nir_nopp.bin")
    # Get names from Edamam dataset
    food_name = list(df['Name'])
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


def recipebert(df):
    # Set seed
    torch.manual_seed(2025)
    food_name = list(df['Name'])
    # pipeline
    embedding = pipeline(
        'feature-extraction', model='alexdseo/bert-base-uncased-finetuned-recipe1m-ALL', framework='pt'
    )
    # Get embeddings
    ebd = list()
    for i in tqdm(food_name):
        ebd.append(embedding(i, return_tensors='pt')[0].numpy().mean(axis=0))
    # Make it numpy arry
    bert_we = np.array(ebd)

    return bert_we

def name_ingr(df):
    """
       Append food names and ingredients from the dataset

       Args:
           df: Edamam ingredients dataset

       Returns:
           lst_txt: Appended texts
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

def name_des(df):
    """
       Append food names and description from the dataset

       Args:
           df: Dataset with menu description

       Returns:
           lst_txt: Appended texts
    """
    # Save only text that is human readable
    lst_text = list()
    for i in tqdm(range(len(df), desc='Getting text...')):
        text = df['name'][i]
        if df['description'][i] is not np.nan:
            text += '. ' + df['description'][i]  # add . after the name, then description
        # Append to list
        lst_text += [text]

    return lst_text


def get_setence_embeddings(txt):
    """
       Get sentence embeddings using pretrained MPNET

       Args:
           txt: Appended texts of either names and ingredients or names and descriptions
       Returns:
           mpnet_embeddings: A numpy array of sentence embeddings from pretrained MPNET
       """
    # Download pretrained model
    mpnet = SentenceTransformer('all-mpnet-base-v2', device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    # SBERT embeddings #Run with gpu
    mpnet_embeddings = mpnet.encode(txt)

    return mpnet_embeddings


if __name__ == "__main__":
    # Set seed
    np.random.seed(2025)
    # Read training dataset
    training_menu = pd.read_csv('./files/generic_food_training_nutrition_sample.csv')
    training_ingr = pd.red_csv('./files/generic_food_training_ingredients_sample.csv')

    # Read Recipe 1M+ dataset # Access this data through their project website
    recipe1m = json.load(open('layer1.json'))
    names, names_ingr, names_ingr_recipe = get_text(recipe1m)
    # Write in text # Choose text with names, ingredients, and recipe
    with open('train_nir_nopp.txt', 'w') as f:
        for line in names_ingr_recipe:
            f.write(f"{line}\n")

    # Train fasttext model
    train_recipeft('train_nir_nopp.txt')
    # Retrieve RecipeFT embeddings
    recipeft_we = recipeft(training_menu)
    # Save the RecipeFT embeddings
    np.save('./files/training_menu_embedding_recipeft.npy', recipeft_we)

    # Retrieve RecipeFT embeddings
    recipebert_we = recipebert(training_menu)
    # Save the RecipeFT embeddings
    np.save('./files/training_menu_embedding_recipebert.npy', recipebert_we)

    # Get menu with ingredients # Embeddings for clustering
    training_name_ingr_txt = name_ingr(training_ingr)
    # Get sentence embeddings from pretrained mpnet
    ingr_setence_embeddings = get_setence_embeddings(training_name_ingr_txt)
    # Save the embeddings
    np.save('./files/training_ingr_sentence_embeddings_mpnet.npy', ingr_setence_embeddings)

    # Read Inference dataset
    inference_menu = pd.read_csv('./files/restaurant_inference_nandes_sample.csv', low_memory=False, lineterminator='\n')
    inference_desc = pd.read_csv('./files/restaurant_inference_des_sample.csv', low_memory=False, lineterminator='\n')

    # Retrieve RecipeFT embeddings # Inference
    recipeft_we_inf = recipeft(inference_menu)
    # Save the RecipeFT embeddings # Inference
    np.save('./files/inference_menu_embedding.npy', recipeft_we_inf)

    # Get name with description
    inference_name_desc_txt = name_des(inference_desc)
    # Get sentence embeddings from pretrained mpnet
    desc_sentence_embeddings = get_setence_embeddings(inference_name_desc_txt)
    # Save the embeddings
    np.save('./files/inference_des_sentence_embedding.npy', desc_sentence_embeddings)
