import numpy as np
from sklearn.model_selection import KFold


def set_fold(df):
    """
        Divide into 5 different folds for cross-validation

        Args:
            df: Any embedding dataset

        Returns:
            Index numbers for each fold
    """
    # Set 5-fold
    kf = KFold(n_splits=5, random_state=2025, shuffle=True)
    for i, (train_index, test_index) in enumerate(kf.split(df)):
        if i == 0:
            kf_1_tr, kf_1_ts = train_index, test_index
        elif i == 1:
            kf_2_tr, kf_2_ts = train_index, test_index
        elif i == 2:
            kf_3_tr, kf_3_ts = train_index, test_index
        elif i == 3:
            kf_4_tr, kf_4_ts = train_index, test_index
        elif i == 4:
            kf_5_tr, kf_5_ts = train_index, test_index

    # Save all
    np.save('kf1_tr.npy', kf_1_tr)
    np.save('kf1_ts.npy', kf_1_ts)
    np.save('kf2_tr.npy', kf_2_tr)
    np.save('kf2_ts.npy', kf_2_ts)
    np.save('kf3_tr.npy', kf_3_tr)
    np.save('kf3_ts.npy', kf_3_ts)
    np.save('kf4_tr.npy', kf_4_tr)
    np.save('kf4_ts.npy', kf_4_ts)
    np.save('kf5_tr.npy', kf_5_tr)
    np.save('kf5_ts.npy', kf_5_ts)

    return kf_1_tr, kf_1_ts, kf_2_ts, kf_2_ts, kf_3_tr, kf_3_ts, kf_4_tr, kf_4_ts, kf_5_tr, kf_5_ts


def bound_score_who(data, lower_bound, upper_bound, minimize):
    """
        Threshold scoring system based on upper bound and lower bound for each nutrients to construct WHO score

        Args:
            data: Nutreint of the food item
            lower_bound: Defined lower bound per 100g for nutrient of interest
            upper_bound: Defined upper bound per 100g for nutrient of interest
            minimize (boolean): Whether the nutrient is a minimize score or not

        Returns: Individual nutrient score that consists to construct WHO score
            
    """
    if minimize:
        return np.where(data < lower_bound, 2, np.where(data < upper_bound, 1, 0))
    else:
        return np.where(data < lower_bound, 0, np.where(data < upper_bound, 1, 2))


def calc_who_score(df):
    """
        Calculate WHO score

        Args:
            df: Dataset with nutrients

        Returns:
            WHO score
    """
    # Use serving size for sodium
    sodium_g = df['Sodium(mg)'] / 1000
    sodium_ss = sodium_g * df['Serving Weight(g)'] / 100
    # calculate bounding score for each nutrient
    protein_who = bound_score_who(df['Protein(g)'], lower_bound=10, upper_bound=15, minimize=False)
    carb_who = bound_score_who(df['Carbohydrate(g)'], lower_bound=55, upper_bound=75, minimize=False)
    sugar_who = bound_score_who(df['Total Sugar(g)'], lower_bound=0, upper_bound=10, minimize=True)
    fat_who = bound_score_who(df['Total Fat(g)'], lower_bound=15, upper_bound=30, minimize=True)
    satfat_who = bound_score_who(df['Saturated Fat(g)'], lower_bound=0, upper_bound=10, minimize=True)
    fiber_who = bound_score_who(df['Fiber(g)'], lower_bound=0, upper_bound=3, minimize=False)
    # Special case for sodium calculating based per serving size
    sodium_who = bound_score_who(sodium_ss, lower_bound=0, upper_bound=2, minimize=True)
    # sum the bounding score for final score
    who_score = sum([protein_who, carb_who, sugar_who, fat_who, satfat_who, fiber_who, sodium_who])
    
    return who_score


def bound_score_fsa(data, serving_size, lower_bound, upper_bound, upper_bound_pp):
    """
        Threshold scoring system based on upper bound and lower bound for each nutrients to construct FSA score

        Args:
            data: Nutrient of the food item
            serving_size: Serving size of the food item
            lower_bound: Defined lower bound per 100g for nutrient of interest
            upper_bound: Defined upper bound per 100g for nutrient of interest
            upper_bound_pp: Defined upper bound per portion for nutrient of interest

        Returns:
            Individual nutrient score that consists to construct FSA score
    """
    data_ss = data * serving_size / 100
    score = np.where(
        data_ss > upper_bound_pp, 0,
        np.where(
            data < lower_bound, 2,
            np.where(data < upper_bound, 1, 0)
        )
    )
    
    return score

    
def calc_fsa_score(df):
    """
        Calculate FSA score

        Args:
            df: Dataset with nutrients

        Returns:
            FSA score
    """
    # get serving size
    ss = df['Serving Weight(g)']
    # Use g instead of mg
    sodium_g = df['Sodium(mg)'] / 1000
    # calculate bounding score for each nutrient
    fat_fsa = bound_score_fsa(df['Total Fat(g)'], serving_size=ss, lower_bound=3, upper_bound=17.5, upper_bound_pp=21)
    satfat_fsa = bound_score_fsa(df['Saturated Fat(g)'], serving_size=ss, lower_bound=1.5, upper_bound=5, upper_bound_pp=6)
    sugar_fsa = bound_score_fsa(df['Total Sugar(g)'], serving_size=ss, lower_bound=5, upper_bound=22.5, upper_bound_pp=27)
    sodium_fsa = bound_score_fsa(sodium_g, serving_size=ss, lower_bound=0.3, upper_bound=1.5, upper_bound_pp=1.8)
    # sum the bounding score for final score
    fsa_score = sum([sugar_fsa, fat_fsa, satfat_fsa, sodium_fsa])
    
    return fsa_score


def nutrient_density_score(nutri):
    """
        Calculate nutrition density scores

        Args:
            nutri: Any nutrition dataset

        Returns:
            nutri: New dataset with nutrition density scores added
     """
    # RRR 
    nutri['RRR'] = ((nutri['Protein(g)'] / 50 + nutri['Fiber(g)'] / 28 + nutri['Calcium(mg)'] / 1300 +
                     nutri['Iron(mg)'] / 18 + nutri['Vitamin A(µg)'] / 900 + nutri['Vitamin C(mg)'] / 90) / 6) \
                   / ((nutri['Calories(kcal)'] / 2000 + nutri['Total Sugar(g)'] / 50 +
                       nutri['Cholesterol(mg)'] / 300 + nutri['Saturated Fat(g)'] / 20 +
                       nutri['Sodium(mg)'] / 2300) / 5)

    # NRF9.3 # Did not multiply 100
    nutri['NRF9.3'] = (nutri['Protein(g)']/50 + nutri['Fiber(g)']/28 + nutri['Vitamin A(µg)']/900 + nutri['Vitamin C(mg)']/90 + nutri['Vitamin E(mg)']/15 + nutri['Calcium(mg)']/1300 \
    + nutri['Iron(mg)']/18 + nutri['Magnesium(mg)']/420 + nutri['Potassium(mg)']/4700 - nutri['Saturated Fat(g)']/20 - nutri['Total Sugar(g)']/50 - nutri['Sodium(mg)']/2300)

    # NRF6.3 
    nutri['NRF6.3'] = (nutri['Protein(g)']/50 + nutri['Fiber(g)']/28 + nutri['Vitamin A(µg)']/900 + nutri['Vitamin C(mg)']/90 + nutri['Calcium(mg)']/1300 \
    + nutri['Iron(mg)']/18 - nutri['Saturated Fat(g)']/20 - nutri['Total Sugar(g)']/50 - nutri['Sodium(mg)']/2300)
    
    # LIM 
    nutri['LIM'] = (nutri['Saturated Fat(g)']/20 + nutri['Total Sugar(g)']/50 + nutri['Sodium(mg)']/2300)
    
    # WHO score
    nutri['WHO'] = calc_who_score(nutri)

    # FSA score
    nutri['FSA'] = calc_fsa_score(nutri)
    
    return nutri
