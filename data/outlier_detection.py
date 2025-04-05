import pandas as pd
from tqdm import tqdm

def outlier_detection(csv):
    """
    Perform outlier detection on large-scale real-world restaurant menu dataset

    Returns:
        `ol_1p.csv`: csv files with outlier restaurant ID with menu less than 6 itmes or more than 692 items.
    """
    # Save number of menu per restaurant
    inference_complete = pd.read_csv(csv, low_memory=False, lineterminator='\n')
    num_menu = pd.DataFrame(inference_complete.groupby(['restaurant_ID']).count()['menu_ID'])
    num_menu['restaurant_ID'] = num_menu.index
    num_menu = num_menu[['restaurant_ID', 'menu_ID']].reset_index(drop=True)
    num_menu = num_menu.rename(columns={'menu_ID': 'count'})
    # <6 and >692 # 1% from each tails
    ol_1p = num_menu[(num_menu['count'] < 6) | (num_menu['count'] > 692)]['restaurant_ID']
    # Export
    ol_1p.to_csv('ol_1p.csv',index=False)

if __name__ == "__main__":
    outlier_detection(csv='./files/restaurant_inference_sample.csv')