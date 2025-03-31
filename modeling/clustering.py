import numpy as np
import pandas as pd
import hdbscan
import umap
from modeling.utils import *
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import sys


class FoodCategorySoftClustering:
    """
        Perform soft clustering on training dataset to create food category pseudo-label that will be used to train 
        the food category prediction model. The food category pseudo-labels are subsequently used to partition
        the data and predicted food category psuedo-label to be used to assign to each food category specific
        nutrient density predicting expert model

        Returns:
            Training dataset in csv file with cluster labels acting as food category pseudo-label
    """
    def __init__(self,
                random_seed = 2025,
                data_path = '../data/files/',
                cluster_size = 1000,
                hdbscan_min_sam = [10, 20, 50],
                hdbscan_epsilon = [0, 0.05, 0.1],
                umap_dim =[2, 10, 50],
                umap_neigh = [10, 20, 50],
        ):
        self.random_seed = random_seed
        # fixed hyperparameter
        self.cluster_size = cluster_size
        # hdbscan hyperparameter
        self.min_sam_check = hdbscan_min_sam
        self.ce_check = hdbscan_epsilon
        # umap hyperparameter
        self.dim_check = umap_dim
        self.n_neigh_check = umap_neigh
        # import the sentence embeddings to cluster
        self.se_df = np.load(data_path + 'training_ingr_sentence_embeddings_mpnet.npy')
        # import nutrition dataset
        self.nt_df = pd.read_csv(data_path + 'generic_food_training_nutrition_sample.csv')
        # Use only nutritions
        self.nt_only_df = self.nt_df.iloc[:, 2:]

    def tuning(self):
        # Hyperparmeter tuning
        # Set seed
        np.random.seed(self.random_seed)
        for d in self.dim_check:
            for nb in self.n_neigh_check:
                # Map embeddings using cosine distance
                tune_umap = umap.UMAP(n_neighbors=nb, n_components=d, min_dist=0.0, metric='cosine',
                                      random_state=self.random_seed).fit_transform(self.se_df)
                # UMAP
                df = pd.DataFrame(tune_umap)
                for eps in self.ce_check:
                    for sam in self.min_sam_check:
                        # HDBSCAN # Clustering based on Euclidean distance
                        tune_umap_hdbscan = hdbscan.HDBSCAN(min_cluster_size=self.cluster_size, metric='l2',
                                                            cluster_selection_epsilon=eps, min_samples=sam,
                                                            prediction_data=True).fit(tune_umap)
                        # Membership vector
                        membership_vec = hdbscan.all_points_membership_vectors(tune_umap_hdbscan)
                        # Assign soft cluster labels with the highest probability
                        tune_sc = [np.argmax(x) for x in membership_vec]
                        df['sc'] = tune_sc
                        # Check the tested hyperparameters
                        print(d, nb, eps, sam)
                        # higher the better
                        # Score on embeddings
                        ch_emb = calinski_harabasz_score(self.se_df, df['sc'])
                        # Score on nutrition
                        ch_nut = calinski_harabasz_score(self.nt_only_df, df['sc'])
                        print('CH cluster score by embeddings:', ch_emb, '\n',
                              'CH cluster score by nutrition:', ch_nut, '\n',
                              'CH cluster mean score:', np.mean([ch_emb, ch_nut]))
                        # lower the better
                        # Score on embeddings
                        db_emb = davies_bouldin_score(self.se_df, df['sc'])
                        # Score on nutrition
                        db_nut = davies_bouldin_score(self.nt_only_df, df['sc'])
                        print('DB cluster score by embeddings:', db_emb, '\n',
                              'DB cluster score by nutrition:', db_nut, '\n',
                              'DB cluster mean score:', np.mean([db_emb, db_nut]))
                        # higher the better
                        # Score on embeddings
                        si_emb = silhouette_score(self.se_df, df['sc'], sample_size=10000, random_state=self.random_seed)
                        # Score on nutrition
                        si_nut = silhouette_score(self.nt_only_df, df['sc'], sample_size=10000, random_state=self.random_seed)
                        print('Silhouette cluster score by embeddings:', si_emb, '\n',
                              'Silhouette  cluster score by nutrition:', si_nut, '\n',
                              'Silhouette  cluster mean score:', np.mean([si_emb, si_nut]))

    def clustering(self):
        # Divide into 5-fold of training and test dataset
        kf_1_tr, _, kf_2_tr, _, kf_3_tr, _, kf_4_tr, _, kf_5_tr, _ = set_fold(self.se_df)
        folds_tr_ind = [kf_1_tr, kf_2_tr, kf_3_tr, kf_4_tr, kf_5_tr]
        # Set seed
        np.random.seed(self.random_seed)
        for i, fold_ind in enumerate(folds_tr_ind):
            # UMAP # Chosen hyperparmeter
            embedding_umap = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine',
                                       random_state=self.random_seed).fit_transform(self.se_df[fold_ind])
            # Get original dataset and add nutrient density score
            df = self.nt_df[fold_ind]
            df = nutrient_density_score(df)
            # HDBSCAN # Chosen hyperparmeter
            embedding_umap_hdbscan = hdbscan.HDBSCAN(min_cluster_size=1000, metric='l2', cluster_selection_epsilon=0.1,
                                                     min_samples=20, prediction_data=True).fit(embedding_umap)
            # Membership vector
            soft_clusters_base = hdbscan.all_points_membership_vectors(embedding_umap_hdbscan)
            # Soft cluster labels
            sc = [np.argmax(x) for x in soft_clusters_base]
            # Assign soft clusters
            df['sc'] = sc
            # Export it to csv
            df.to_csv(f"training_kf{i + 1}.csv", encoding='utf-8', index=False)


if __name__ == "__main__":
    # Get arguments
    tuning = sys.argv[1]  # Perform hyperparmeter for tuning or not # boolean (True or False)
    # Clustering
    create_food_category = FoodCategorySoftClustering()
    # Tuning hyperparameter for clustering
    if tuning:
        create_food_category.tuning()
    # Get food category by clustering the sentence embedding # Export the csv file for all folds
    create_food_category.clustering()
