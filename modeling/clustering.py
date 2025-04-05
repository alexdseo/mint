import numpy as np
import pandas as pd
import hdbscan
import umap
from modeling.utils import *
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import argparse


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
                si_sample = 10000,
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
        # Sample size for silhouette_score
        self.si_sample = si_sample

    @staticmethod
    def z_score(series):
        """
            Compute Z-score normalization to a Pandas series.
        """
        mean, std = series.mean(), series.std()
        return (series - mean) / (std if std > 0 else 1)  # Avoid division by zero

    def tuning(self):
        # Hyperparmeter tuning
        # Set seed
        np.random.seed(self.random_seed)
        # Lists for scores
        scores_lst = list()
        for d in self.dim_check:
            for nb in self.n_neigh_check:
                # Map embeddings using cosine distance
                tune_umap = umap.UMAP(n_neighbors=nb, n_components=d, min_dist=0.0, metric='cosine',
                                    random_state=self.random_seed).fit_transform(self.se_df)

                for eps in self.ce_check:
                    for sam in self.min_sam_check:
                        # HDBSCAN # Clustering based on Euclidean distance
                        tune_umap_hdbscan = hdbscan.HDBSCAN(
                            min_cluster_size=self.cluster_size, metric='l2',
                            cluster_selection_epsilon=eps, min_samples=sam,
                            prediction_data=True
                        ).fit(tune_umap)
                        # Membership vector
                        membership_vec = hdbscan.all_points_membership_vectors(tune_umap_hdbscan)
                        # Assign soft cluster labels with the highest probability
                        tune_sc = [np.argmax(x) for x in membership_vec]

                        # Compute clustering scores
                        ch_score = np.mean([
                            calinski_harabasz_score(self.se_df, tune_sc),
                            calinski_harabasz_score(self.nt_only_df, tune_sc) # Clustering evaluation based on nutrient # Can be excluded
                        ])
                        db_score = np.mean([
                            davies_bouldin_score(self.se_df, tune_sc),
                            davies_bouldin_score(self.nt_only_df, tune_sc)
                        ])
                        si_score = np.mean([
                            silhouette_score(self.se_df, tune_sc, sample_size=self.si_sample, random_state=self.random_seed),
                            silhouette_score(self.nt_only_df, tune_sc, sample_size=self.si_sample, random_state=self.random_seed)
                        ])
                        # Store raw scores
                        scores_lst.append({
                            'd': d, 'nb': nb, 'eps': eps, 'sam': sam,
                            'ch_score': ch_score, 'db_score': db_score, 'si_score': si_score
                        })

        # To df
        scores_df = pd.DataFrame(scores_lst)
        # Apply Z-score normalization
        scores_df['ch_norm'] = self.z_score(scores_df['ch_score'])
        scores_df['db_norm'] = self.z_score(scores_df['db_score'])  
        scores_df['si_norm'] = self.z_score(scores_df['si_score'])
        # Compute overall score # Invert DB score since lower is better
        scores_df['overall_score'] = (scores_df['ch_norm'] - scores_df['db_norm'] + scores_df['si_norm']) / 3

        # Get best hyperparameters
        best_row = scores_df.loc[scores_df['overall_score'].idxmax()]
        best_params = {
            'n_components': best_row['d'],
            'n_neighbors': best_row['nb'],
            'epsilon': best_row['eps'],
            'min_samples': best_row['sam'],
            'best_score': best_row['overall_score']
        }
        # Print best hyperparameters
        print("\nBest Hyperparameters:")
        for key, value in best_params.items():
            print(f"{key}: {value}")
        
        return best_params

    def clustering(self, n_comp=2, n_neigh=15, eps=0.1, min_sam=20):
        # Divide into 5-fold of training and test dataset
        kf_1_tr, _, kf_2_tr, _, kf_3_tr, _, kf_4_tr, _, kf_5_tr, _ = set_fold(self.se_df)
        folds_tr_ind = [kf_1_tr, kf_2_tr, kf_3_tr, kf_4_tr, kf_5_tr]
        # Set seed
        np.random.seed(self.random_seed)
        for i, fold_ind in enumerate(folds_tr_ind):
            # UMAP # Chosen hyperparmeter
            embedding_umap = umap.UMAP(n_neighbors=n_neigh, n_components=n_comp, min_dist=0.0, metric='cosine',
                                       random_state=self.random_seed).fit_transform(self.se_df[fold_ind])
            # Get original dataset and add nutrient density score
            df = self.nt_df[fold_ind]
            df = nutrient_density_score(df)
            # HDBSCAN # Chosen hyperparmeter
            embedding_umap_hdbscan = hdbscan.HDBSCAN(min_cluster_size=self.cluster_size, metric='l2', cluster_selection_epsilon=eps,
                                                     min_samples=min_sam, prediction_data=True).fit(embedding_umap)
            # Membership vector
            soft_clusters_base = hdbscan.all_points_membership_vectors(embedding_umap_hdbscan)
            # Assign soft cluster labels
            df['sc'] = [np.argmax(x) for x in soft_clusters_base]
            # Export it to csv
            df.to_csv(f"training_kf{i + 1}.csv", encoding='utf-8', index=False)


if __name__ == "__main__":
    # Get arguments
    parser = argparse.ArgumentParser(description="Run the script with an option for hyperparmeter tuning")
    parser.add_argument('tuning', type=bool, help="Perform hyperparmeter for tuning or not. ex) True, False")
    args = parser.parse_args()
    # Clustering
    create_food_category = FoodCategorySoftClustering()
    # Tuning hyperparameter for clustering
    if args.tuning:
        best_params = create_food_category.tuning()
        create_food_category.clustering(n_comp=best_params['n_components'], n_neigh=best_params['n_neighbors'],
                                        eps=best_params['epsilon'], min_sam=best_params['min_samples'])
    else:
        # Get food category by clustering the sentence embedding # Export the csv file for all folds
        create_food_category.clustering()
