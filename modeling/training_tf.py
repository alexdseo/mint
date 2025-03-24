import numpy as np
import pandas as pd
from modeling.utils import *
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score
import scipy.stats as st
import sys


class NNArchitecture(keras.Model):
    def __init__(
            self,
            random_seed = 2025, 
            dim = 300, # For RecipeFT: 300
            # Tuned Hyperparmeter for MINT's training data # Food Category Model
            fcm_l2 = 0.0001, 
            fcm_initializer = 'he_uniform', 
            fcm_activation = 'relu',
            fcm_neurons = 256,
            fcm_do = 0.2,
            # Tuned Hyperparmeter for MINT's training data # Nutrient Density Prediction Model
            ndpm_neurons = 256,
            ndpm_leaky = 0.1,
            ndpm_do = 0.2
        ):
        super().__init__()
        self.random_seed = random_seed
        self.dim = dim
        self.fcm_l2 = fcm_l2
        self.fcm_init = fcm_initializer
        self.fcm_acti = fcm_activation
        self.fcm_neurons = fcm_neurons
        self.fcm_do = fcm_do
        self.ndpm_neurons = ndpm_neurons
        self.ndpm_leaky = ndpm_leaky
        self.ndpm_do = ndpm_do


    def food_category_classification_model(self, output_dim:int):
        """
            Build neural network architecture for food category classification model.

            Args:
                n: Number of food category pseudo labels

            Returns:
                classification_model: model architecture
        """
        # Set seed
        np.random.seed(self.random_seed)
        # # Feature Normalization
        # embedding_normalizer = Normalization(input_shape=[dim, ], axis=None)
        # embedding_normalizer.adapt(X_train)
        # Classification model architecture # Best parameter from gridsearch
        classification_model = Sequential([
            #embedding_normalizer,
            tf.keras.Input(shape=(self.dim,)),
            Dense(self.fcm_neurons, activation=self.fcm_acti, kernel_initializer=self.fcm_init, 
                  kernel_regularizer=regularizers.l2(self.fcm_l2)),
            # BatchNormalization(),
            Dropout(self.fcm_do),
            Dense(self.fcm_neurons, activation=self.fcm_acti, kernel_initializer=self.fcm_init, 
                  kernel_regularizer=regularizers.l2(self.fcm_l2)),
            # BatchNormalization(),
            Dropout(self.fcm_do),
            Dense(self.fcm_neurons, activation=self.fcm_acti, kernel_initializer=self.fcm_init, 
                  kernel_regularizer=regularizers.l2(self.fcm_l2)),
            # BatchNormalization(),
            Dropout(self.fcm_do),
            Dense(output_dim, activation='softmax')
        ])

        return classification_model

    def nutrition_density_prediction_model(self, output_dim=1):
        """
            Build neural network architecture for nutrition prediction model.

            Returns:
                prediction_model: model architecture
        """
        # Set seed
        np.random.seed(self.random_seed)
        # Prediction model architecture
        prediction_model = Sequential([
            tf.keras.Input(shape=(self.dim,)),
            Dense(self.ndpm_neurons),
            LeakyReLU(self.ndpm_leaky),
            Dropout(self.ndpm_do),
            Dense(self.ndpm_neurons),
            LeakyReLU(self.ndpm_leak),
            Dropout(self.ndpm_do),
            Dense(self.ndpm_neurons),
            LeakyReLU(self.ndpm_leak),
            Dropout(self.ndpm_do),
            Dense(output_dim, activation='linear')
        ])

        return prediction_model


class DataProcessor:
    def __init__(self, random_seed=2025):
        self.random_seed = random_seed

    def one_hot_encoding(self, train_df):
        """
            One hot encoding for the categorical label (food category pseudo-label).

            Args:
                train_df: training dataset that needs to be one hot encoded

            Returns:
                ohe_train: one hot encoded labels
        """
        # set seed
        np.random.seed(self.random_seed)
        # Change category to categorical label: One hot encoding
        sc_tr = np.array(train_df['sc']).reshape(-1, 1)
        # sc_ts=np.array(y_test['sc']).reshape(len(y_test),1)
        # Integer label to one hot encoding dummy variable
        ohe = OneHotEncoder(sparse=False)
        ohe_train = ohe.fit_transform(sc_tr)
        # ohe_test = ohe.transform(sc_ts)

        return ohe_train
    
    @staticmethod
    def mint_preprocessing(num_lb, train_y, test_y):
        """
            Partition data to make food category expert model.

            Args:
                num_lb: number of food category pseudo labels
                train_y: training data
                test_y: test data

            Returns:
                training_result_dict: partitioned training data based on the food category psuedo label
                test_result_dict: partitioned test data based on the predicted food category psuedo label
        """
        # food categories- training
        fc_name_lst, tr_fc_df_lst = list(), list()
        for i in range(num_lb):
            fc_name_lst.append(f"sc_{i}") # category numbers
            tr_fc_df_lst.append(train_y.loc[train_y['sc'] == i])
        fc_name_tuple = tuple(fc_name_lst)
        tr_fc_df_tuple = tuple(tr_fc_df_lst)

        # food categories - test
        ts_fc_df_lst = list()
        for i in range(num_lb):
            ts_fc_df_lst.append(test_y.loc[test_y['sc'] == i])
        ts_fc_df_tuple = tuple(ts_fc_df_lst)

        # Save tuples into dictionary
        training_result_dict = dict(zip(fc_name_tuple, tr_fc_df_tuple))
        test_result_dict = dict(zip(fc_name_tuple, ts_fc_df_tuple))

        return training_result_dict, test_result_dict
    

class PredictionMethod:
    @staticmethod
    def get_weighted_mean(num_lb, nutri, y_true_train_base, y_true_test_base, y_pred_test_base):
        """
            Predict nutrient density score using weighted mean method. MINT without the deep learning portion.
            Partition the data based on the food category pseudo-labels and use their mean to predict the nutrient density score.
            Predicted probability to each food category label act as weight to calculate weighted mean.

            Args:
                num_lb: number of food category pseudo labels
                nutri: nutrient density score to predict
                y_true_train_base: training data after one hot encoding
                y_true_test_base: test data
                y_pred_test_base: predicted probabilty on food category label using test data

            Returns:
                Prints r2 score in predicting target nutrient density score using weighted means method and exports results in csv file
        """
        # Get true mean nutrition score for each food category
        true_mean = list()
        for i in range(num_lb):
            true_mean.append(y_true_train_base.loc[y_true_train_base['sc'] == i][nutri].mean())
        # Make prediction by calculating weighted mean from food category prediction
        pred = list()
        for i in range(len(y_pred_test_base)):
            pred += [sum(true_mean * y_pred_test_base[i])]
        pred = pd.Series(pred, name=f"Predicted_{nutri}")
        print('Baseline Test r-squared: %f' % r2_score(y_true_test_base[nutri], pred))
        # Export as csv
        pred = pd.concat([y_true_test_base, pred], axis=1).reset_index(drop=True)
        pred.to_csv(f"FCWM_pred_{nutri}.csv", encoding='utf-8', index=False)
    
    @staticmethod
    def predict_distribution(X, model, num_samples):
        """
            Bayesian approximation with MC dropout

            Args:
                model: fitted model
                num_samples: samples for Bayesian approximation

            Returns:
                prediction list to construct distribution
        """
        # Bayesian approximation using MC dropout, get distribution
        preds = [model(X, training=True) for _ in range(num_samples)]
        return np.hstack(preds)

    def predict_uncertainty(self, X, model, num_samples):
        """
            Bayesian approximation with MC dropout point estimates

            Args: 
                X: test embedding
                model: fitted model
                num_samples: samples for Bayesian approximation

            Returns:
                prediction list and its average
        """
        # Bayesian approximation using MC dropout, get mean of distribution
        pred_dist = self.predict_distribution(X, model, num_samples)
        return pred_dist, pred_dist.mean(axis=1)


class Pipeline(PredictionMethod):
    def __init__(self, 
                 random_seed = 2025, 
                 fcm_lr = 0.0001, 
                 ndpm_lr = 0.0001,
                 em_wd = 0.001,
                 batch_size = 64
        ):
        super().__init__()
        self.random_seed = random_seed
        self.data_processor = DataProcessor()
        self.nn_arch = NNArchitecture()
        self.fcm_lr = fcm_lr
        self.ndpm_lr = ndpm_lr
        self.em_wd = em_wd # Expert model Weight Decay
        self.batch_size = batch_size # Change batch size based on the processor

    
    def baseline(self, output_dim, train_X, test_X, train_y, test_y, score):
        """
            Baseline approach of weighted mean method for predicting nutrient density score.
            Used to compare with other approach for ablation study.

            Args:
                output_dim: number of food category pseudo labels
                train_X: training embeddings
                test_X: test embeddings
                train_y: training data
                test_y: test data
                score: nutrient density score to predict

            Returns:
                test_y: test data with predicted food category psuedo labels
        """
        # Set seed
        np.random.seed(self.random_seed)
        # One hot encoding for the classification label
        ohe_train = self.data_processor.one_hot_encoding(train_y)
        # Call classification model
        model = self.nn_arch.food_category_classification_model(output_dim)
        # Set learning rate
        opt = keras.optimizers.Adam(learning_rate=self.fcm_lr)
        # Compile, categorical_accuracy if one hot label
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])
        # print(model.summary())
        # Early stopping
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        # Fit model
        history = model.fit(train_X, ohe_train, validation_split=0.2, epochs=100, 
                            batch_size=self.batch_size, verbose=2, callbacks=[es])
        # Predict food category for test dataset
        y_pred_target = model.predict(test_X) # Can be used for MoE # Use different activation and loss function
        # Assign one label from the prediction, this will be used later for mint's fine-tuning process
        ts_label = np.argmax(y_pred_target, axis=1)
        # Get baseline result # User can change to different nutrition density score
        self.get_weighted_mean(output_dim, score, train_y, test_y, y_pred_target)
        # Save predicted food category for test dataset
        test_y['sc'] = pd.Series(ts_label)

        return test_y

    def mlp(self, train_X, test_X, train_y, test_y, score, kf):
        """
            Vanila Multi Layer Perceptron (MLP) to predict the nutrient density score using the defined neural network architecture.

            Args:
                train_X: training embeddings
                test_X: test embeddings
                train_y: training data
                test_y: test data with predicted food category psuedo label
                score: nutrient density score to predict
                kf: one of the folds to test on

            Returns:
                Prints r2 score in predicting target nutrient density score using vanila MLP and exports results in csv file.
                Saves the weights after training the model with entire training dataset.
        """
        # Set seed
        np.random.seed(self.random_seed)
        # Normalize the nutrition density score
        y_sc = StandardScaler()
        Ny_train = y_sc.fit_transform(np.array(train_y[score]).reshape(-1, 1))
        # Call nutrient density prediction model
        model = self.nn_arch.nutrition_density_prediction_model()
        # Set tuned learning rate
        opt = keras.optimizers.Adam(learning_rate=self.ndpm_lr)
        # Compile, regression task
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
        # print(model.summary())
        # Early stopping
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        # Fit model
        history = model.fit(train_X, Ny_train, validation_split=0.2, epochs=100, 
                            batch_size=self.batch_size, verbose=2, callbacks=[es])
        # Save model weights # User need to change the name of the weights for different fold
        model.save_weights(f"{score}_weights_{kf}")
        # Bayesian Approximation using MC dropout
        _, y_pred = self.predict_uncertainty(np.array(test_X), model, 100)
        y_pred = y_sc.inverse_transform(y_pred.reshape(-1, 1))
        print('MLP Test r-squared: %f' % r2_score(test_y[score], y_pred))
        # Export the result as csv
        pred = pd.DataFrame(y_pred, columns=[f"Predicted_{score}"]).reset_index(drop=True)
        pred.to_csv(f"MLP_pred_{score}_{kf}.csv", encoding='utf-8', index=False)


    def expert_model(self, train_X_fc, test_X_fc, train_y_fc, weights):
        """
            Individual expert model in a multi-expert system that MINT employs. Each experts represents food category expert.
            Each expert loads the pre-trained weights before further training with the specific subset of the data.
            Derives uncertainty estimates through MC dropout and gets confidence interval.            

            Args:
                train_X_fc: partitioned training embeddings for each food category
                test_X_fc: partitioned test embeddings for each food category
                train_y_fc: partitioned training data
                weights: pre-trained weights trained with entire dataset

            Returns:
                y_pred: Predicted nutrient density score
                ci: Confidence interval on prediction for uncertainty
        """
        # Set seed
        np.random.seed(self.random_seed)
        # Normalize the nutrition density score
        y_sc = StandardScaler()
        Ny_train_fc = y_sc.fit_transform(np.array(train_y_fc).reshape(-1, 1))
        expert_model = self.nn_arch.nutrition_density_prediction_model()
        # Load weights
        expert_model.load_weights(weights)
        # Set tuned learning rate
        opt = keras.optimizers.Adam(learning_rate=self.ndpm_lr, weight_decay = self.em_wd)
        # Compile, regression task
        expert_model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
        # print(model.summary())
        # Early stopping
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        # Fit model
        history = expert_model.fit(train_X_fc, Ny_train_fc, validation_split=0.2, epochs=30, 
                                   batch_size=self.batch_size, verbose=2, callbacks=[es])
        # Bayesian Approximation using MC dropout
        y_pred_dist, y_pred = self.predict_uncertainty(np.array(test_X_fc), expert_model, 100)
        # Inverse transform to real value
        y_pred = y_sc.inverse_transform(y_pred.reshape(-1, 1))
        # Get CI
        lower_ci, upper_ci = [], []
        # Create 95% confidence interval for population mean weight
        for i in range(len(y_pred_dist)):
            lci = st.norm.interval(confidence=0.95, loc=np.mean(y_pred_dist[i]), scale=st.sem(y_pred_dist[i]))[0]
            uci = st.norm.interval(confidence=0.95, loc=np.mean(y_pred_dist[i]), scale=st.sem(y_pred_dist[i]))[1]
            lower_ci.append(lci)
            upper_ci.append(uci)

        ci = np.concatenate([y_sc.inverse_transform(np.array(lower_ci).reshape(-1, 1)), y_sc.inverse_transform(np.array(upper_ci).reshape(-1, 1))], axis=1)


        return y_pred, ci

    def finetune(self, train_X, test_X, train_dict, test_dict, nutri, ptw):
        """
            Fine tuning the expert model. Gets nutrient density prediction by allocating the data based on the food category pseudo-labels.

            Args:
                train_X: training embeddings
                test_X: test embeddings
                traind_dict: partitioned training data based on the food category psuedo label
                test_dict: partitioned test data based on the predicted food category psuedo label
                nutri: nutrient density score to predict
                ptw: pre-trained weights

            Returns:
                tests: ground truth test data with predicted food category pseudo-labels
                preds: predicted nutrient density score from each food category experts
                ci: confidence interval of predicted nutrient density score
        """
        # Individual experts
        for i, fc in enumerate(train_dict.keys()):
            # Get X data for each food category
            X_sc = train_X[train_dict[fc].index]
            X_sc_t = test_X[test_dict[fc].index]
            # Get prediction for individual food category expert # Multi-expert system
            y_sc, y_sc_ci = self.expert_model(X_sc, X_sc_t, train_dict[fc][nutri], ptw)
            # Concatenate the prediction from different food categories
            if i == 0:
                preds = y_sc
                ci = y_sc_ci
            else:
                preds = np.concatenate([preds, y_sc], axis=0)
                ci = np.concatenate([ci, y_sc_ci], axis=0)

        # Save each menu item's true nutrient density score and their food category
        # Define empty placeholder
        tests = pd.DataFrame()
        for fc in test_dict.keys():
            tests = pd.concat([tests, test_dict[fc][['Name', nutri, 'sc']]], axis=0)

        return tests, preds, ci

    def mint(self, n, train_X, test_X, train_y, test_y, score , kf):
        """
            MINT. Partition the data through food category pseudo label (mint_processing) and finetune the pretrained
            model to create food category expert models.

            Args:
                n: Number of food categories
                train_X: training embeddings
                test_X: test embeddings
                train_y: training data
                test_y: test data with predicted food category psuedo label
                score: nutrient density score to predict
                kf: one of the folds to test on

            Returns:
                Prints r2 score in predicting target nutrient density score using MINT and exports results in csv file
        """
        # Divide dataset into categories
        train_dict, test_dict = self.data_processor.mint_preprocessing(n, train_y, test_y)
        # Finetune by the food category
        true_test, pred_test, ci_test = self.finetune(train_X, test_X, train_dict, test_dict, score, f"{score}_weights_{kf}")
        # Save results in a dataframes
        pred = pd.DataFrame(pred_test, columns=[f"Predicted_{score}"]).reset_index(drop=True)
        test = pd.DataFrame(true_test).reset_index(drop=True)
        ci = pd.DataFrame(ci_test, columns=['lower_ci', 'upper_ci']).reset_index(drop=True)
        final_results = pd.concat([test, pred, ci], axis=1)
        # Print mint r2 score
        print('MINT Test r-squared: %f' % r2_score(final_results[score], final_results[f"Predicted_{score}"]))
        # print('MINT Test RMSE: %f' % mean_squared_error(final_results[score], final_results[f"Predicted_{score}"], squared=False))
        # Export the result
        final_results.to_csv(f"MINT_pred_{score}_{kf}.csv", encoding='utf-8', index=False)


if __name__ == "__main__":
    # Set seed
    np.random.seed(2025)
    # Get arguments for score and fold that wanted to test 
    nds = sys.argv[1]  # nutrition desnsity score # ex) RRR, NRF9.3, NRF6.3, LIM, WHO, FSA
    fold = sys.argv[2]  # one of the folds to test on # ex) kf1 ~ kf5
    # Read one of the fold's training and test index # Created using utils.set_fold
    # User can change this part and try different folds
    train_ind = np.load(f"{fold}_tr.npy")
    test_ind = np.load(f"{fold}_ts.npy")
    # Read FastText embeddings
    X = np.load('../data/files/training_menu_embedding_recipeft.npy')
    # Define training and test dataset
    X_train, X_test = X[train_ind], X[test_ind]
    # Read one of the fold that was created from the clustering.py
    y_train = pd.read_csv(f"training_{fold}.csv")
    num_sc = len(set(y_train['sc']))
    # Full dataset without fodd category
    y = pd.read_csv('../data/files/generic_food_training_nutrition_sample.csv')
    y_test = y[test_ind]
    # Assign true nutrition density score
    y_test = nutrient_density_score(y_test)

    mint = Pipeline()
    # Run baseline method and get the predicted food category for the test dataset
    y_test = mint.baseline(num_sc, X_train, X_test, y_train, y_test, nds)
    # Run vanila mlp method and save weight for finetuning process
    mint.mlp(X_train, X_test, y_train, y_test, nds, fold)
    # Run MINT
    mint.mint(num_sc, X_train, X_test, y_train, y_test, nds, fold)