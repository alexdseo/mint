import pandas as pd
import argparse

def estimate_RND(ol_1p, nds, csv):
    """
    Estimate RND using predicted nutrient density score on each menu item from ~600k restaurants in the US

    Args:
        ol_1p: Outlier restaurant IDs to exclude
        nds: nutrient density score

    Returns:
        `RND_{nds}.csv`: csv file with estimated Restaurant Nutrient Density (RND) score based on selected nutrient density score of menu item
        `RND_{nds}_{meal_type}.csv`: csv files with estimated RND considering the predicted meal type and selected nutrient density score

    """
    # Inference dataset w/ nutrient density prediction
    inference_complete = pd.read_csv(csv, low_memory=False, lineterminator='\n')
    # Discard detected outliers # 1pct from each tails of entire df
    ol_1p_index = inference_complete[inference_complete['restaurant_ID'].isin(ol_1p['restaurant_ID'])].index
    inference_complete.drop(ol_1p_index, inplace=True)
    # Divide it by AMDD
    appetizer = inference_complete[inference_complete['AMDD'] == 0]
    main = inference_complete[inference_complete['AMDD'] == 1]
    dessert = inference_complete[inference_complete['AMDD'] == 2]
    drink = inference_complete[inference_complete['AMDD'] == 3]
    # Median of nutreint density score of all menu for each restaurant
    RND = inference_complete[[f"Predicted_{nds}", 'lower_ci', 'upper_ci', 'restaurant_ID',
                              'location_latitude', 'location_longitude']]\
        .groupby(['restaurant_ID']).median(numeric_only=True)
    RND['restaurant_ID'] = RND.index
    RND = RND.rename(columns={f"Predicted_{nds}": f"RND_{nds}"}).reset_index(drop=True)
    # Median of nutreint density score of appetizer for each restaurant
    RND_APP = appetizer[[f"Predicted_{nds}", 'lower_ci', 'upper_ci', 'restaurant_ID',
                         'location_latitude', 'location_longitude']]\
        .groupby(['restaurant_ID']).median(numeric_only=True)
    RND_APP['restaurant_ID'] = RND_APP.index
    RND_APP = RND_APP.rename(columns={f"Predicted_{nds}": f"RND_{nds}_APP"}).reset_index(drop=True)
    # Median of nutreint density score of main dish for each restaurant
    RND_MAIN = main[[f"Predicted_{nds}", 'lower_ci', 'upper_ci', 'restaurant_ID',
                     'location_latitude', 'location_longitude']]\
        .groupby(['restaurant_ID']).median(numeric_only=True)
    RND_MAIN['restaurant_ID'] = RND_MAIN.index
    RND_MAIN = RND_MAIN.rename(columns={f"Predicted_{nds}": f"RND_{nds}_MAIN"}).reset_index(drop=True)
    # Median of nutreint density score of dessert dish for each restaurant
    RND_DSRT = dessert[[f"Predicted_{nds}", 'lower_ci', 'upper_ci', 'restaurant_ID',
                        'location_latitude', 'location_longitude']]\
        .groupby(['restaurant_ID']).median(numeric_only=True)
    RND_DSRT['restaurant_ID'] = RND_DSRT.index
    RND_DSRT = RND_DSRT.rename(columns={f"Predicted_{nds}": f"RND_{nds}_DSRT"}).reset_index(drop=True)
    # Median of nutreint density score of drink dish for each restaurant
    RND_DRNK = drink[[f"Predicted_{nds}", 'lower_ci', 'upper_ci', 'restaurant_ID',
                      'location_latitude', 'location_longitude']]\
        .groupby(['restaurant_ID']).median(numeric_only=True)
    RND_DRNK['restaurant_ID'] = RND_DRNK.index
    RND_DRNK = RND_DRNK.rename(columns={f"Predicted_{nds}": f"RND_{nds}_DRNK"}).reset_index(drop=True)
    # Export
    RND.to_csv(f"./files/RND_{nds}.csv", index=False)
    RND_APP.to_csv(f"./files/RND_{nds}_APP.csv", index=False)
    RND_MAIN.to_csv(f"./files/RND_{nds}_MAIN.csv", index=False)
    RND_DSRT.to_csv(f"./files/RND_{nds}_DSRT.csv", index=False)
    RND_DRNK.to_csv(f"./files/RND_{nds}_DRNK.csv", index=False)

if __name__ == "__main__":
    # Get arguments for score
    parser = argparse.ArgumentParser(description="Run the script with nutrient density score of interest.")
    parser.add_argument('nds', type=str, help="Nutrition desnsity score. ex) RRR, NRF9.3, NRF6.3, LIM, WHO, FSA")
    args = parser.parse_args()
    # Read outliers
    outliers = pd.read_csv('ol_1p.csv')
    estimate_RND(ol_1p=outliers, nds=args.nds, csv='./files/restaurant_inference_sample.csv')