import numpy as np
import pandas as pd
from modeling.utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import scipy.stats as st
import argparse


# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'
torch.cuda.empty_cache()


class FoodCateogryCLSModel(nn.Module):
    """
        Build neural network architecture for food category classification model.
    """
    def __init__(
            self, 
            input_dim: int, 
            output_dim: int,
            neurons = 256,
            do_rate = 0.2
        ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, neurons)
        self.fc2 = nn.Linear(neurons, neurons)
        self.fc3 = nn.Linear(neurons, neurons)
        self.fc4 = nn.Linear(neurons, output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(do_rate)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = self.activation(self.fc3(x))
        x = self.dropout(x)
        x = F.softmax(self.fc4(x), dim=1)
        
        return x
    

class NutrientDensityPredModel(nn.Module):
    """
        Build neural network architecture for nutrition prediction model.
    """
    def __init__(
            self, 
            input_dim: int,
            output_dim =1,
            neurons = 256,
            leaky_slope = 0.1,
            do_rate = 0.2

        ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, neurons)
        self.fc2 = nn.Linear(neurons, neurons)
        self.fc3 = nn.Linear(neurons, neurons)
        self.fc4 = nn.Linear(neurons, output_dim)
        self.leaky_slope = leaky_slope
        self.do_rate = do_rate

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=self.leaky_slope)
        x = F.dropout(x, p=self.do_rate , training=True)
        x = F.leaky_relu(self.fc2(x), negative_slope=self.leaky_slope)
        x = F.dropout(x, p=self.do_rate , training=True)
        x = F.leaky_relu(self.fc3(x), negative_slope=self.leaky_slope)
        x = F.dropout(x, p=self.do_rate , training=True)
        x = self.fc4(x)

        return x


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
    def __init__(self, random_seed=2025):
        self.random_seed = random_seed

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

    def training(self, model, device, train_X, train_y, optimizer, criterion, epochs):
        """
            Model training with early stopping and validation loop

            Args:
                model: model to train
                device: cpu or gpu
                train_X: training embeddings
                train_y: training data
                optimizer: optimizer function for the training
                criterion: loss function for the training
                epochs: epochs for the training

            Returns:
                model: trained model
        """
        # Split the data into training and validation sets
        train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.2, random_state=self.random_seed)
        # Convert data to PyTorch tensors
        train_X = torch.tensor(train_X, dtype=torch.float32)
        train_y = torch.tensor(train_y, dtype=torch.float32)
        val_X = torch.tensor(val_X, dtype=torch.float32)
        val_y = torch.tensor(val_y, dtype=torch.float32)

        # Create DataLoaders for training and validation
        train_dataset = TensorDataset(train_X, train_y)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_dataset = TensorDataset(val_X, val_y)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        # Early stopping
        early_stopping = False
        patience = 10
        best_val_loss = float('inf')
        counter = 0

        # Training loop
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X.to(device))
                loss = criterion(outputs, batch_y.to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            average_loss = total_loss / len(train_loader)

            # Validation
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X.to(device))
                    val_loss += criterion(outputs, batch_y.to(device)).item()
        
            average_val_loss = val_loss / len(val_loader)
            print(
                f'Epoch [{epoch + 1}/{epochs}], Training Loss: {average_loss:.4f}, Validation Loss: {average_val_loss:.4f}')
        
            # Early stopping
            if average_val_loss < best_val_loss:
                best_val_loss = average_val_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print('Early stopping...')
                    early_stopping = True
                    break
        
        if not early_stopping:
            print('Training completed without early stopping.')

        return model
    
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
        # Enable dropout layers during inference
        model.train() 
        # Bayesian approximation using MC dropout, get distribution
        preds = torch.cat([model(X).unsqueeze(0) for _ in range(num_samples)], dim=0)
        return preds

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
        return pred_dist, pred_dist.mean(dim=0)


class Pipeline(PredictionMethod):
    def __init__(self,
                 device, 
                 random_seed = 2025, 
                 fcm_lr = 0.0001, 
                 ndpm_lr = 0.0001,
                 fcm_wd = 0.0001,
                 em_wd = 0.001,
                 model_epochs = 100,
                 em_epochs = 30,
                 batch_size = 64
        ):
        super().__init__()
        self.device = device
        self.random_seed = random_seed
        self.data_processor = DataProcessor()
        self.fcm_lr = fcm_lr
        self.ndpm_lr = ndpm_lr
        self.fcm_wd = fcm_wd # Food Category CLS model Weight Decay
        self.em_wd = em_wd # Expert model Weight Decay
        self.model_epochs = model_epochs
        self.em_epochs = em_epochs
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
        train_y = self.data_processor.one_hot_encoding(train_y)
        input_dim = train_X.shape[1]
        # Instantiate the model
        model = FoodCateogryCLSModel(input_dim=input_dim, output_dim=output_dim).to(self.device) 
        # Set optimizer and learning rate
        optimizer = optim.Adam(model.parameters(), lr=self.fcm_lr, weight_decay=self.fcm_wd)
        # Define loss function
        criterion = nn.CrossEntropyLoss()
        # Training
        epochs = self.model_epochs
        model = self.training(model, self.device, train_X, train_y, optimizer, criterion, epochs)
        # Use fully trained model for prediction
        model.eval()
        # Predict food category for test dataset # Can be used for MoE # Use different activation and loss function
        with torch.no_grad():
            y_pred_target = model(torch.tensor(test_X, dtype=torch.float32).to(self.device)) 
        # Assign one label from the prediction
        ts_label = torch.argmax(y_pred_target, dim=1).cpu().numpy()
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
        input_dim = train_X.shape[1]
        # Instantiate the model
        model = NutrientDensityPredModel(input_dim=input_dim).to(self.device)  
        # Normalize nutrient density score
        y_sc = StandardScaler()
        Ny_train = torch.tensor(y_sc.fit_transform(np.array(train_y[score]).reshape(-1, 1)).astype(np.float32)).view(-1, 1)
        # Set optimizer and learning rate
        optimizer = optim.Adam(model.parameters(), lr=self.ndpm_lr)
        # Define loss function
        criterion = nn.MSELoss()
        # Training
        epochs = self.model_epochs
        model = self.training(model, self.device, train_X, Ny_train, optimizer, criterion, epochs)
        # Save model weights # User need to change the name of the weights for different fold
        torch.save(model.state_dict(), f"{score}_weights_{kf}")
        # Use fully trained model for prediction
        model.eval()
        # Predict food category for test dataset # Can be used for MoE # Use different activation and loss function
        with torch.no_grad():
            # Direct prediction without MC dropout
            # y_pred = model(torch.tensor(test_X, dtype=torch.float32).to(self.device)) 
            # Bayesian Approximation using MC dropout
            _, y_pred = self.predict_uncertainty(torch.tensor(test_X, dtype=torch.float32).to(self.device), model, 100)
        y_pred = y_sc.inverse_transform(y_pred.detach().cpu().numpy())
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
        input_dim = train_X_fc.shape[1]
        # Instantiate the model
        model = NutrientDensityPredModel(input_dim=input_dim).to(self.device)  
        # Load weights
        model.load_state_dict(torch.load(weights))
        # Normalize nutrient density score
        y_sc = StandardScaler()
        Ny_train_fc = torch.tensor(y_sc.fit_transform(np.array(train_y_fc).reshape(-1, 1)).astype(np.float32)).view(-1, 1)
        # Set optimizer and learning rate
        optimizer = optim.Adam(model.parameters(), lr=self.ndpm_lr, weight_decay=self.em_wd)
        # Define loss function
        criterion = nn.MSELoss()
        # Training
        epochs = self.em_epochs
        model = self.training(model, self.device, train_X_fc, Ny_train_fc, optimizer, criterion, epochs)
        # Use fully trained model for prediction
        model.eval()
        # Predict food category for test dataset # Can be used for MoE # Use different activation and loss function
        with torch.no_grad():
            # Bayesian Approximation using MC dropout
            y_pred_dist, y_pred = self.predict_uncertainty(torch.tensor(test_X_fc, dtype=torch.float32).to(self.device), model, 100)
        y_pred = y_sc.inverse_transform(y_pred.detach().cpu().numpy())
        y_pred_dist = y_pred_dist.permute(1, 0, 2).reshape(-1, y_pred_dist.shape[-1])
        y_pred_dist = y_sc.inverse_transform(y_pred_dist.detach().cpu().numpy())
        # Get CI
        lower_ci, upper_ci = [], []
        # Create 95% confidence interval for population mean weight
        for i in range(len(y_pred_dist)):
            lci = st.norm.interval(confidence=0.95, loc=np.mean(y_pred_dist[i]), scale=st.sem(y_pred_dist[i]))[0]
            uci = st.norm.interval(confidence=0.95, loc=np.mean(y_pred_dist[i]), scale=st.sem(y_pred_dist[i]))[1]
            lower_ci.append(lci)
            upper_ci.append(uci)

        ci = np.concatenate([np.array(lower_ci).reshape(-1, 1), np.array(upper_ci).reshape(-1, 1)], axis=1)

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
    parser = argparse.ArgumentParser(description="Run the script with nutrient density score of interest and fold to test on.")
    parser.add_argument('nds', type=str, help="Nutrition desnsity score. ex) RRR, NRF9.3, NRF6.3, LIM, WHO, FSA")
    parser.add_argument('fold', type=str, help=" ex) Folds created from clustering.py, test for cross-validation. ex) kf1, kf2, kf3, kf4, kf5")
    args = parser.parse_args()
    # Set gpu # If not using cuda, change
    gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Read one of the fold's training and test index # Created by running clutering.py from utils.set_fold
    train_ind = np.load(f"{args.fold}_tr.npy")
    test_ind = np.load(f"{args.fold}_ts.npy")
    # Read FastText embeddings
    X = np.load('../data/files/training_menu_embedding_recipeft.npy')
    # Define training and test dataset
    X_train, X_test = X[train_ind], X[test_ind]
    # Read one of the fold that was created from the clustering.py
    y_train = pd.read_csv(f"training_{args.fold}.csv")
    num_sc = len(set(y_train['sc']))
    # Full dataset without fodd category
    y = pd.read_csv('../data/files/generic_food_training_nutrition_sample.csv')
    y_test = y[test_ind]
    # Assign true nutrition density score
    y_test = nutrient_density_score(y_test)

    mint = Pipeline(device=gpu)
    # Run baseline method and get the predicted food category for the test dataset
    y_test = mint.baseline(num_sc, X_train, X_test, y_train, y_test, args.nds)
    # Run vanila mlp method and save weight for finetuning process
    mint.mlp(X_train, X_test, y_train, y_test, args.nds, args.fold)
    # Run MINT
    mint.mint(num_sc, X_train, X_test, y_train, y_test, args.nds, args.fold)



