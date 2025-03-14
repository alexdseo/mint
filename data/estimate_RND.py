import pandas as pd
from tqdm import tqdm

def estimate_RND(ol_1p):
    # AMDD
    RND, RND_APP, RND_MAIN, RND_DSRT, RND_DRNK = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    # Large-scale dataset divided in 10 batches
    for i in tqdm(range(1, 11)):
        inference_complete = pd.read_csv('inference_complete_' + str(i) + '.csv', low_memory=False, lineterminator='\n')
        # Discard detected outliers # 1pct from each tails
        ol_1p_index = inference_complete[inference_complete['restaurant_ID'].isin(ol_1p['restaurant_ID'])].index
        inference_complete.drop(ol_1p_index, inplace=True)
        # Divide it by AMDD
        appetizer = inference_complete[inference_complete['AMDD'] == 0]
        main = inference_complete[inference_complete['AMDD'] == 1]
        dessert = inference_complete[inference_complete['AMDD'] == 2]
        drink = inference_complete[inference_complete['AMDD'] == 3]
        # RRR Median of all menu for each restaurant
        RND_batch = inference_complete[['Predicted_RRR', 'lower_ci', 'upper_ci', 'restaurant_ID',
                                        'location_latitude', 'location_longitude']]\
            .groupby(['restaurant_ID']).median(numeric_only=True)
        RND_batch['restaurant_ID'] = RND_batch.index
        RND_batch = RND_batch.rename(columns={"Predicted_RRR": "RND_RRR"}).reset_index(drop=True)
        RND = pd.concat([RND, RND_batch])
        # RRR Median of appetizer for each restaurant
        RND_APP_batch = appetizer[['Predicted_RRR', 'lower_ci', 'upper_ci', 'restaurant_ID',
                                   'location_latitude', 'location_longitude']]\
            .groupby(['restaurant_ID']).median(numeric_only=True)
        RND_APP_batch['restaurant_ID'] = RND_APP_batch.index
        RND_APP_batch = RND_APP_batch.rename(columns={"Predicted_RRR": "RND_RRR_APP"}).reset_index(drop=True)
        RND_APP = pd.concat([RND_APP, RND_APP_batch])
        # RRR Median of main dish for each restaurant
        RND_MAIN_batch = main[['Predicted_RRR', 'lower_ci', 'upper_ci', 'restaurant_ID',
                               'location_latitude', 'location_longitude']]\
            .groupby(['restaurant_ID']).median(numeric_only=True)
        RND_MAIN_batch['restaurant_ID'] = RND_MAIN_batch.index
        RND_MAIN_batch = RND_MAIN_batch.rename(columns={"Predicted_RRR": "RND_RRR_MAIN"}).reset_index(drop=True)
        RND_MAIN = pd.concat([RND_MAIN, RND_MAIN_batch])
        # RRR Median of dessert dish for each restaurant
        RND_DSRT_batch = dessert[['Predicted_RRR', 'lower_ci', 'upper_ci', 'restaurant_ID',
                                  'location_latitude', 'location_longitude']]\
            .groupby(['restaurant_ID']).median(numeric_only=True)
        RND_DSRT_batch['restaurant_ID'] = RND_DSRT_batch.index
        RND_DSRT_batch = RND_DSRT_batch.rename(columns={"Predicted_RRR": "RND_RRR_DSRT"}).reset_index(drop=True)
        RND_DSRT = pd.concat([RND_DSRT, RND_DSRT_batch])
        # RRR Median of drink dish for each restaurant
        RND_DRNK_batch = drink[['Predicted_RRR', 'lower_ci', 'upper_ci', 'restaurant_ID',
                                'location_latitude', 'location_longitude']]\
            .groupby(['restaurant_ID']).median(numeric_only=True)
        RND_DRNK_batch['restaurant_ID'] = RND_DRNK_batch.index
        RND_DRNK_batch = RND_DRNK_batch.rename(columns={"Predicted_RRR": "RND_RRR_DRNK"}).reset_index(drop=True)
        RND_DRNK = pd.concat([RND_DRNK, RND_DRNK_batch])
    # Export
    RND.to_csv('RND_RRR.csv', index=False)
    RND_APP.to_csv('RND_RRR_APP.csv', index=False)
    RND_MAIN.to_csv('RND_RRR_MAIN.csv', index=False)
    RND_DSRT.to_csv('RND_RRR_DSRT.csv', index=False)
    RND_DRNK.to_csv('RND_RRR_DRNK.csv', index=False)

if __name__ == "__main__":
    outliers = pd.read_csv('ol_1p.csv')
    estimate_RND(outliers)