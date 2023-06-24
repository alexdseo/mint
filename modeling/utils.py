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
    kf = KFold(n_splits=5, random_state=2023, shuffle=True)
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


def nutrient_density_score(nutri):
    """
        Calculate nutrition density scores

        Args:
            nutri: Any nutrition dataset

        Returns:
            nutri: New dataset with nutrition density scores added
     """
    # RRR unmodified
    nutri['RRR'] = ((nutri['Protein(g)'] / 50 + nutri['Fiber(g)'] / 28 + nutri['Calcium(mg)'] / 1300 +
                     nutri['Iron(mg)'] / 18 + nutri['Vitamin A(µg)'] / 900 + nutri['Vitamin C(mg)'] / 90) / 6) \
                   / ((nutri['Calories(kcal)'] / 2000 + nutri['Total Sugar(g)'] / 50 +
                       nutri['Cholesterol(mg)'] / 300 + nutri['Saturated Fat(g)'] / 20 +
                       nutri['Sodium(mg)'] / 2300) / 5)

    # RRR macro
    nutri['RRR_m1'] = ((nutri['Protein(g)'] / 50 + nutri['Fiber(g)'] / 28) / 2) \
                      / ((nutri['Calories(kcal)'] / 2000 + nutri['Total Sugar(g)'] / 50 +
                          nutri['Cholesterol(mg)'] / 300 + nutri['Saturated Fat(g)'] / 20 +
                          nutri['Sodium(mg)'] / 2300) / 5)

    return nutri
