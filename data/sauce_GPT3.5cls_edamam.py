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
def AMDD_cls(name):
    """
    Generate menu description using GPT-3.5 turbo

    Args:
        df: dataset with food name list

    Returns:
        lst_gd: list of labels
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

if __name__ == "__main__":
    # Set seed
    np.random.seed(1996)
    # API key
    openai.api_key = ''
    # Read dataset
    recipe = pd.read_csv('edamam_sorted_sample.csv')
    # Sample
    names = recipe['Name']
    # Define list to store the text
    lst_labels= list()
    # Get generated description and its number of tokens
    ind_s = 0
    ind_e = len(names)
    while True:
        try:
            for i in tqdm(range(ind_s, ind_e)):
                generated_labels = AMDD_cls(names[i])
                lst_labels.append(generated_labels)
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
    with open('edamam_sauce_cls', 'wb') as fp:
        pickle.dump(lst_labels, fp)