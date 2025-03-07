import openai
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import functools
import signal
import pickle
from tenacity import retry, wait_random_exponential, retry_if_exception_type

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
def generate_description(name, ingredients):
    """
    Generate menu description using GPT-3.5 turbo

    Args:
        df: dataset with food name and ingredients list

    Returns:
        lst_gd: list of generated descriptions
        lst_ot: list of number of output tokens per descriptions
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

    return completion.choices[0].message.content, completion.usage.completion_tokens

if __name__ == "__main__":
    # Set seed
    np.random.seed(1996)
    start = sys.argv[1]
    end = sys.argv[2]
    # API key
    openai.api_key = ''
    # Read dataset
    ingr = pd.read_csv('edamam_ingredients_sample.csv')
    # Sample
    ingr_sample = ingr.iloc[int(start):int(end)]
    ingr_sample = ingr_sample.reset_index(drop=True)
    # Define list to store the text
    lst_gd, lst_ot = list(), list()
    # Get generated description and its number of tokens
    ind_s = 0
    ind_e = len(ingr_sample)
    while True:
        try:
            for i in tqdm(range(ind_s, ind_e)):
                generated_description, output_tokens = generate_description(ingr_sample['Name'][i], ingr_sample['Ingredients_only'][i][1:-1])
                lst_gd.append(generated_description)
                lst_ot.append(output_tokens)
            end_loop = i + 1
            if end_loop == ind_e:
                break
            else:
                print('Something is missing, run again')
        except TimeoutError:
            print('TimeoutError, ')
            print('Stopped at:', i)
            ind_s = i
        except IndexError:
            print('IndexError, ')
            print('Stopped at:', i)
            ind_s = i

    # Export via pickle
    with open('gen_des_' + start + '_' + end, 'wb') as fp:
        pickle.dump(lst_gd, fp)
    with open('op_tok_' + start + '_' + end, 'wb') as fp:
        pickle.dump(lst_ot, fp)