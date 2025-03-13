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


def generating_loop(names: list, additional_check: bool, ind_e: int, ind_s=0):
    """
    Generates labels based on the provided `names` list using either `sauce_cls` or `AMDD_cls`.

    Args:
    - names (list): List of menu names.
    - additional_check (bool): Determines which function to use for label generation.
    - ind_e (int): Ending index for iteration. To process all, use length of the menu item list.
    - ind_s (int, optional): Starting index for iteration. Defaults to 0.

    Returns:
    - lst_labels: Generated labels.
    """
    lst_labels = []
    
    while True:
        try:
            for i in tqdm(range(ind_s, ind_e), desc="Classifying Menus..."):
                generated_labels = sauce_cls(names[i]) if additional_check else AMDD_cls(names[i])
                lst_labels.append(generated_labels)
            
            # Successfully processed all elements
            return lst_labels
        
        except TimeoutError:
            print(f"TimeoutError occurred at index {i}. Restarting from index {i+1}...")
            ind_s = i + 1  # Move to the next index
            
        except IndexError:
            print(f"IndexError occurred at index {i}. Check if `names` has sufficient elements.")
            break  # Exit loop

    return lst_labels



if __name__ == "__main__":
    # Set seed
    np.random.seed(1996)
    # Read dataset
    df = pd.read_csv('generic_food_training_nutrition_sample.csv')
    names = df['Name']
    generated_labels=generating_loop(names=names, additional_check=False, ind_e=len(names))

    # Export via pickle
    with open('AMDD_cls_result', 'wb') as fp:
        pickle.dump(generated_labels, fp)

    # Read dataset # Use dataset that was human-validated for hallucinations for additional classification
    df = pd.read_csv('AMDD_labels_initial_result.csv')
    names = df['Name']
    generated_labels = generating_loop(names=names, additional_check=True, ind_e=len(names))

    # Export via pickle
    with open('AMDDS_cls_result', 'wb') as fp:
        pickle.dump(generated_labels, fp)