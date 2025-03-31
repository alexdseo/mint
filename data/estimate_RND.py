import pandas as pd
from tqdm import tqdm
import sys

def estimate_RND(ol_1p, nds):
    """
    Estimate RND using predicted nutrient density score on each menu item from ~600k restaurants in the US

    Args:
        ol_1p: Outlier restaurant IDs to exclude
        nds: nutrient density score

    Returns:
        `RND_{nds}.csv`: csv file with estimated Restaurant Nutrient Density (RND) score based on selected nutrient density score of menu item
        `RND_{nds}_{meal_type}.csv`: csv files with estimated RND considering the predicted meal type and selected nutrient density score

    """
    # AMDD
    RND, RND_APP, RND_MAIN, RND_DSRT, RND_DRNK = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    # Large-scale dataset divided in 10 batches
    for i in tqdm(range(1, 11), desc='Estimating RND...'):
        inference_complete = pd.read_csv('inference_complete_' + str(i) + '.csv', low_memory=False, lineterminator='\n')
        # Discard detected outliers # 1pct from each tails
        ol_1p_index = inference_complete[inference_complete['restaurant_ID'].isin(ol_1p['restaurant_ID'])].index
        inference_complete.drop(ol_1p_index, inplace=True)
        # Divide it by AMDD
        appetizer = inference_complete[inference_complete['AMDD'] == 0]
        main = inference_complete[inference_complete['AMDD'] == 1]
        dessert = inference_complete[inference_complete['AMDD'] == 2]
        drink = inference_complete[inference_complete['AMDD'] == 3]
        # Median of nutreint density score of all menu for each restaurant
        RND_batch = inference_complete[[f"Predicted_{nds}", 'lower_ci', 'upper_ci', 'restaurant_ID',
                                        'location_latitude', 'location_longitude']]\
            .groupby(['restaurant_ID']).median(numeric_only=True)
        RND_batch['restaurant_ID'] = RND_batch.index
        RND_batch = RND_batch.rename(columns={f"Predicted_{nds}": f"RND_{nds}"}).reset_index(drop=True)
        RND = pd.concat([RND, RND_batch])
        # Median of nutreint density score of appetizer for each restaurant
        RND_APP_batch = appetizer[[f"Predicted_{nds}", 'lower_ci', 'upper_ci', 'restaurant_ID',
                                   'location_latitude', 'location_longitude']]\
            .groupby(['restaurant_ID']).median(numeric_only=True)
        RND_APP_batch['restaurant_ID'] = RND_APP_batch.index
        RND_APP_batch = RND_APP_batch.rename(columns={f"Predicted_{nds}": f"RND_{nds}_APP"}).reset_index(drop=True)
        RND_APP = pd.concat([RND_APP, RND_APP_batch])
        # Median of nutreint density score of main dish for each restaurant
        RND_MAIN_batch = main[[f"Predicted_{nds}", 'lower_ci', 'upper_ci', 'restaurant_ID',
                               'location_latitude', 'location_longitude']]\
            .groupby(['restaurant_ID']).median(numeric_only=True)
        RND_MAIN_batch['restaurant_ID'] = RND_MAIN_batch.index
        RND_MAIN_batch = RND_MAIN_batch.rename(columns={f"Predicted_{nds}": f"RND_{nds}_MAIN"}).reset_index(drop=True)
        RND_MAIN = pd.concat([RND_MAIN, RND_MAIN_batch])
        # Median of nutreint density score of dessert dish for each restaurant
        RND_DSRT_batch = dessert[[f"Predicted_{nds}", 'lower_ci', 'upper_ci', 'restaurant_ID',
                                  'location_latitude', 'location_longitude']]\
            .groupby(['restaurant_ID']).median(numeric_only=True)
        RND_DSRT_batch['restaurant_ID'] = RND_DSRT_batch.index
        RND_DSRT_batch = RND_DSRT_batch.rename(columns={f"Predicted_{nds}": f"RND_{nds}_DSRT"}).reset_index(drop=True)
        RND_DSRT = pd.concat([RND_DSRT, RND_DSRT_batch])
        # Median of nutreint density score of drink dish for each restaurant
        RND_DRNK_batch = drink[[f"Predicted_{nds}", 'lower_ci', 'upper_ci', 'restaurant_ID',
                                'location_latitude', 'location_longitude']]\
            .groupby(['restaurant_ID']).median(numeric_only=True)
        RND_DRNK_batch['restaurant_ID'] = RND_DRNK_batch.index
        RND_DRNK_batch = RND_DRNK_batch.rename(columns={f"Predicted_{nds}": f"RND_{nds}_DRNK"}).reset_index(drop=True)
        RND_DRNK = pd.concat([RND_DRNK, RND_DRNK_batch])
    # Export
    RND.to_csv(f"./files/RND_{nds}.csv", index=False)
    RND_APP.to_csv(f"./files/RND_{nds}_APP.csv", index=False)
    RND_MAIN.to_csv(f"./files/RND_{nds}_MAIN.csv", index=False)
    RND_DSRT.to_csv(f"./files/RND_{nds}_DSRT.csv", index=False)
    RND_DRNK.to_csv(f"./files/RND_{nds}_DRNK.csv", index=False)

if __name__ == "__main__":
    # Get arguments for score
    nds = sys.argv[1]  # nutrition desnsity score # ex) RRR, NRF9.3, NRF6.3, LIM, WHO, FSA
    outliers = pd.read_csv('ol_1p.csv')
    estimate_RND(outliers, nds=nds)