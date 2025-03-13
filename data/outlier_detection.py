import pandas as pd
from tqdm import tqdm

def outlier_detection():
    # Save number of menu per restaurant
    num_menu = pd.DataFrame()
    # Large-scale dataset divided in 10 batches
    for i in tqdm(range(1, 11)):
        inference_complete = pd.read_csv('inference_complete_' + str(i) + '.csv', low_memory=False, lineterminator='\n')
        num_menu_tmp = pd.DataFrame(inference_complete.groupby(['restaurant_ID']).count()['menu_ID'])
        # print(len(x))
        num_menu_tmp['restaurant_ID'] = num_menu_tmp.index
        num_menu_tmp = num_menu_tmp.reset_index(drop=True)
        num_menu = pd.concat([num_menu, num_menu_tmp])
    num_menu = num_menu.reset_index(drop=True)
    num_menu = num_menu[['restaurant_ID', 'menu_ID']]
    num_menu = num_menu.rename(columns={'menu_ID': 'count'})
    # <6 and >692 # 1% from each tails
    ol_1p = num_menu[(num_menu['count'] < 6) | (num_menu['count'] > 692)]['restaurant_ID']
    # Export
    ol_1p.to_csv('ol_1p.csv',index=False)

if __name__ == "__main__":
    outlier_detection()