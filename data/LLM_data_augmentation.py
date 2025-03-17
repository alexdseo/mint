import openai
import pandas as pd
import numpy as np
from tqdm import tqdm
import functools
import signal
import pickle
from tenacity import retry, wait_random_exponential, retry_if_exception_type


# API key
openai.api_key = '' # Insert your OpenAI API key


def timeout(seconds=5, default=None):
    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            def handle_timeout(signum, frame):
                raise TimeoutError()

            signal.signal(signal.SIGALRM, handle_timeout)
            signal.alarm(seconds)


            result = func(*args, **kwargs)

            signal.alarm(0)

            return result

        return wrapper

    return decorator

@retry(
    retry=retry_if_exception_type((openai.error.APIError, openai.error.APIConnectionError, openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.Timeout)),
    wait=wait_random_exponential(multiplier=1, min=4, max=20)
)
@timeout(seconds=5, default=None)
def AMDD_cls(name):
    """
    Classify menu type using GPT-3.5 turbo

    Args:
        name (str): menu item name

    Returns:
        Classified label
    """
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            # system role
            {"role": "system",
             "content": "Is this food item appetizer, main dish, dessert, or a drink? Provide an answer with only one of these 4 answers"},
            # Ask
            {"role": "user", "content": name}
        ]
    )

    return completion.choices[0].message.content

@retry(
    retry=retry_if_exception_type((openai.error.APIError, openai.error.APIConnectionError, openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.Timeout)),
    wait=wait_random_exponential(multiplier=1, min=4, max=20)
)
@timeout(seconds=5, default=None)
def sauce_cls(name):
    """
    Classify menu type using GPT-3.5 turbo (Additional)

    Args:
        name (str): menu item name

    Returns:
        Classified label
    """
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            # system role
            {"role": "system",
             "content": "Is this food item either sauce, dressing, seasoning, or a spice? Answer yes or no."},
            # Ask
            {"role": "user", "content": name}
        ]
    )

    return completion.choices[0].message.content


@retry(
    retry=retry_if_exception_type((openai.error.APIError, openai.error.APIConnectionError, openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.Timeout)),
    wait=wait_random_exponential(multiplier=1, min=4, max=20)
)
@timeout(seconds=5, default=None)
def generate_description(name, ingredients):
    """
    Generate menu description using GPT-3.5 turbo

    Args:
        name (str): menu item name
        ingredients (str): menu item ingredients

    Returns:
        Generated descriptions
    """
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                # system role
                {"role": "system",
                 "content": "You make simple restaurant menu descriptions based on the menu item's ingredients."},
                # example 1
                {"role": "user",
                 "content": "Fajita Burrito, the ingredients are 'sirloin steak', 'green bell pepper', 'onion', 'rice', 'refried beans', 'flour tortilla', 'guacamole', 'cheese', 'salsa'"},
                {"role": "assistant",
                 "content": "Grilled steak with onions and peppers stuffed with rice and beans in a flour tortilla topped with guacamole."},
                # example 2
                {"role": "user",
                 "content": "Yummy Chicken Teriyaki Combo , the ingredients are 'terriyaki sauce', 'garlic', 'broccoli', 'zucchini', 'chicken', 'carrots', 'fried rice'"},
                {"role": "assistant", "content": "Served over Japanese fried rice and sweet carrots."},
                # example 3
                {"role": "user",
                 "content": "Café White Chocolate, the ingredients are 'coffee beans', 'white chocolate powder', 'non-fat milk'"},
                {"role": "assistant",
                 "content": "Lightly roasted coffee with white chocolate powder, topped with steamed non-fat milk."},
                # example 4
                {"role": "user",
                 "content": "Thai Glazed Japanese Eggplant, the ingredients are 'eggplant', 'rice', 'chilies', 'miso paste', 'hoisin sauce', 'rice flour', 'soy sauce', 'sesame oil'"},
                {"role": "assistant",
                 "content": "Deep fried, rice flour, miso paste, hoisin, chilies and rice. Vegetarian."},
                # example 5
                {"role": "user",
                 "content": "Buffalo Chicken Salad, the ingredients are 'romaine lettuce hearts, 'shredded carrot', 'celery', 'blue cheese crumbles', 'hot sauce', 'chicken breast'"},
                {"role": "assistant", "content": "Garden salad with Buffalo chicken."},
                # target
                {"role": "user",
                 "content": name + ", the ingredients are " + ingredients}
            ],
         max_tokens=30
    )

    return completion.choices[0].message.content#, completion.usage.completion_tokens # Number of output tokens per descriptions



def generating_loop(ind_s=0, ind_e=0, names=[], ingr = [], additional_check=False, gen_desc=False):
    """
    Generates labels based on the provided `names` list using either `sauce_cls` or `AMDD_cls`.

    Args:
    - ind_s (int, optional): Starting index for iteration. Defaults to 0.
    - ind_e (int): Ending index for iteration. To process all, use length of the menu item list.
    - names: List of menu names.
    - ingr: Series or list of ingredients for each menu 
    - additional_check (bool): Determines which function to use for label generation.
    - gend_desc (bool): Determines if it is desecription generation task of classification task
    
    

    Returns:
    - lst_labels: Generated labels.
    """
    gen_lst = []
    
    while True:
        try:
            if gen_desc:
                for i in tqdm(range(ind_s, ind_e), desc="Generating Descriptions..."):
                    generated_labels = generate_description(names[i], ingr[i][1:-1])
                    gen_lst.append(generated_labels)
            
                # Successfully processed all elements
                return gen_lst
            else:
                for i in tqdm(range(ind_s, ind_e), desc="Classifying Menus..."):
                    generated_labels = sauce_cls(names[i]) if additional_check else AMDD_cls(names[i])
                    gen_lst.append(generated_labels)
                
                # Successfully processed all elements
                return gen_lst
        
        except TimeoutError:
            print(f"TimeoutError occurred at index {i}. Restarting from index {i+1}...")
            ind_s = i + 1  # Move to the next index
            
        except IndexError:
            print(f"IndexError occurred at index {i}. Check if `names` has sufficient elements.")
            break  # Exit loop

    return gen_lst



if __name__ == "__main__":
    # Set seed
    np.random.seed(1996)
    # Read dataset
    df = pd.read_csv('./files/generic_food_training_nutrition_sample.csv')
    names = df['Name']
    generated_labels=generating_loop(ind_e=len(names), names=names, additional_check=False)

    # Export via pickle
    with open('AMDD_cls_result', 'wb') as fp:
        pickle.dump(generated_labels, fp)


    # Read dataset # Use dataset that was human-validated for hallucinations for additional classification
    df_hlc = pd.read_csv('AMDD_labels_hlc_result.csv') # Hallucinated results
    names = df_hlc['Name']
    generated_labels = generating_loop(ind_e=len(names), names=names, additional_check=True)

    # Export via pickle
    with open('AMDDS_cls_result', 'wb') as fp:
        pickle.dump(generated_labels, fp)


    # Read dataset # Use ingredients dataset for description generation
    df_ingr = pd.read_csv('./files/generic_food_training_ingredients_sample.csv')
    names = df_ingr['Name']
    generated_labels = generating_loop(ind_e=len(names), names=names, ingr=df_ingr['Ingredients_only'], gen_desc=True)

    # Export via pickle
    with open('gen_desc_result', 'wb') as fp:
        pickle.dump(generated_labels, fp)