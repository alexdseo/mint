import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import scipy.stats as st

# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'
torch.cuda.empty_cache()

def preprocessing(y_train):
    # set seed
    np.random.seed(1996)
    # Change category to categorical label: One hot encoding
    sc_tr=np.array(y_train['sc']).reshape(-1,1)
    #sc_ts=np.array(y_test).reshape(-1,1)
    # integer label to one hot encoding dummys
    ohe = OneHotEncoder(sparse=False)
    ohe_train = ohe.fit_transform(sc_tr)
    #ohe_test = ohe.transform(sc_ts)

    return ohe_train

def predict_dist(X, model, num_samples):
    # Bayesian approximation using MC dropout, get distribution
    preds_lst = torch.cat([model(X).unsqueeze(0) for _ in range(num_samples)], dim=0)
    return preds_lst

def predict_point(X, model, num_samples):
    # Bayesian approximation using MC dropout, get mean of distribution
    pred_dist = predict_dist(X, model, num_samples)
    return pred_dist, pred_dist.mean(dim=0)

# Define a PyTorch model
class CLSModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CLSModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(256, 256)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(256, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        x = self.softmax(x)
        return x


def baseline(n, train_X, train_y, device):
    train_y = preprocessing(train_y)
    dim = train_X.shape[1]
    output_dim = n
    weight_decay = 0.0001

    model = CLSModel(dim, output_dim).to(device)  # Instantiate the model
    # # Split the data into training and validation sets
    # train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.2, random_state=1996)
    # Convert data to PyTorch tensors
    train_X = torch.tensor(train_X, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float32)
    # val_X = torch.tensor(val_X, dtype=torch.float32)
    # val_y = torch.tensor(val_y, dtype=torch.float32)

    # Create DataLoaders for training and validation
    batch_size = 64
    train_dataset = TensorDataset(train_X, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_dataset = TensorDataset(val_X, val_y)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Set learning rate
    learning_rate = 0.0001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Define loss function
    criterion = nn.BCELoss()

    # # Early stopping
    # early_stopping = False
    # patience = 10
    # best_val_loss = float('inf')
    # counter = 0

    # Training loop
    epochs = 100
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
        print(f'Epoch [{epoch + 1}/{epochs}], Training Loss: {average_loss:.4f}')

    #     # Validation
    #     model.eval()
    #     with torch.no_grad():
    #         val_loss = 0.0
    #         for batch_X, batch_y in val_loader:
    #             outputs = model(batch_X.to(device))
    #             val_loss += criterion(outputs, batch_y.to(device)).item()
    #
    #     average_val_loss = val_loss / len(val_loader)
    #     print(
    #         f'Epoch [{epoch + 1}/{epochs}], Training Loss: {average_loss:.4f}, Validation Loss: {average_val_loss:.4f}')
    #
    #     # Early stopping
    #     if average_val_loss < best_val_loss:
    #         best_val_loss = average_val_loss
    #         counter = 0
    #     else:
    #         counter += 1
    #         if counter >= patience:
    #             print('Early stopping...')
    #             early_stopping = True
    #             break
    #
    # if not early_stopping:
    #     print('Training completed without early stopping.')

    return model


class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 1)

    def forward(self, x):
        x1 = F.leaky_relu(self.fc1(x), negative_slope=0.1)
        x1 = F.dropout(x1, p=0.2, training=True)
        x2 = F.leaky_relu(self.fc2(x1), negative_slope=0.1)
        x2 = F.dropout(x2, p=0.2, training=True)
        x3 = F.leaky_relu(self.fc3(x2), negative_slope=0.1)
        x3 = F.dropout(x3, p=0.2, training=True)
        output = self.fc4(x3)
        return output


def base_mlp(train_X, train_y, device, ndf='RRR'):
    dim = train_X.shape[1]
    # weight_decay = 0.0001

    model = MLP(dim).to(device)  # Instantiate the model

    y_sc = StandardScaler()
    Ny_train = torch.tensor(y_sc.fit_transform(np.array(train_y[ndf]).reshape(-1, 1)).astype(np.float32))
    Ny_train = Ny_train.view(-1, 1)

    # # Split the data into training and validation sets
    # train_X, val_X, train_y, val_y = train_test_split(train_X, Ny_train, test_size=0.2, random_state=1996)

    # Convert data to PyTorch tensors
    train_X = torch.tensor(train_X, dtype=torch.float32)
    train_y = torch.tensor(Ny_train, dtype=torch.float32)
    # val_X = torch.tensor(val_X, dtype=torch.float32)
    # val_y = torch.tensor(val_y, dtype=torch.float32)

    # Create DataLoaders for training and validation
    batch_size = 64
    train_dataset = TensorDataset(train_X, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_dataset = TensorDataset(val_X, val_y)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Set learning rate
    learning_rate = 0.0001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Define loss function
    criterion = nn.MSELoss()

    # # Early stopping
    # early_stopping = False
    # patience = 10
    # best_val_loss = float('inf')
    # counter = 0

    # Training loop
    epochs = 35
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
        print(f'Epoch [{epoch + 1}/{epochs}], Training Loss: {average_loss:.4f}')

    #     # Validation
    #     model.eval()
    #     with torch.no_grad():
    #         val_loss = 0.0
    #         for batch_X, batch_y in val_loader:
    #             outputs = model(batch_X)
    #             val_loss += criterion(outputs, batch_y).item()
    #
    #     average_val_loss = val_loss / len(val_loader)
    #     print(
    #         f'Epoch [{epoch + 1}/{epochs}], Training Loss: {average_loss:.4f}, Validation Loss: {average_val_loss:.4f}')
    #
    #     # Early stopping
    #     if average_val_loss < best_val_loss:
    #         best_val_loss = average_val_loss
    #         counter = 0
    #     else:
    #         counter += 1
    #         if counter >= patience:
    #             print('Early stopping...')
    #             early_stopping = True
    #             break
    #
    # if not early_stopping:
    #     print('Training completed without early stopping.')

    # Save model weights for transfer learning/fine-tuning process
    torch.save(model.state_dict(), ndf + '_weights_FT_ED.pth')

def mint_preprocessing(num_lb, train_y):
    # food categories- training
    fc_name_lst, tr_fc_df_lst = list(), list()
    for cn in range(num_lb):
        fc_name_lst.append('sc' + str(cn))
        tr_fc_df_lst.append(train_y.loc[train_y['sc'] == cn])
    fc_name_tuple = tuple(fc_name_lst)
    tr_fc_df_tuple = tuple(tr_fc_df_lst)

    # # food categories - test
    # ts_fc_df_lst = list()
    # for i in range(num_lb):
    #     ts_fc_df_lst.append(test_y.loc[test_y['sc'] == i])
    # ts_fc_df_tuple = tuple(ts_fc_df_lst)

    # Save tuples into dictionary
    training_result_dict = dict(zip(fc_name_tuple, tr_fc_df_tuple))
    #test_result_dict = dict(zip(fc_name_tuple, ts_fc_df_tuple))

    return training_result_dict


def task_model(train_X, test_X, train_y, weights, device):
    dim = train_X.shape[1]
    weight_decay = 0.001

    model = MLP(dim).to(device)  # Instantiate the model

    # Load weights
    model.load_state_dict(torch.load(weights))

    y_sc = StandardScaler()
    Ny_train = torch.tensor(y_sc.fit_transform(np.array(train_y).reshape(-1, 1)).astype(np.float32))
    Ny_train = Ny_train.view(-1, 1)

    # # Split the data into training and validation sets
    # train_X, val_X, train_y, val_y = train_test_split(train_X, Ny_train, test_size=0.2, random_state=1996)

    # Convert data to PyTorch tensors
    train_X = torch.tensor(train_X, dtype=torch.float32)
    train_y = torch.tensor(Ny_train, dtype=torch.float32)
    # val_X = torch.tensor(val_X, dtype=torch.float32).to(device)
    # val_y = torch.tensor(val_y, dtype=torch.float32).to(device)

    # Create DataLoaders for training and validation
    batch_size = 64
    train_dataset = TensorDataset(train_X, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_dataset = TensorDataset(val_X, val_y)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Set learning rate
    learning_rate = 0.0001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Define loss function
    criterion = nn.MSELoss()

    # # Early stopping
    # early_stopping = False
    # patience = 10
    # best_val_loss = float('inf')
    # counter = 0

    # Training loop
    epochs = 20 # Fine-tuned hyperparameter
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
        print(f'Epoch [{epoch + 1}/{epochs}], Training Loss: {average_loss:.4f}')

    #     # Validation
    #     model.eval()
    #     with torch.no_grad():
    #         val_loss = 0.0
    #         for batch_X, batch_y in val_loader:
    #             outputs = model(batch_X)
    #             val_loss += criterion(outputs, batch_y).item()
    #
    #     average_val_loss = val_loss / len(val_loader)
    #     print(
    #         f'Epoch [{epoch + 1}/{epochs}], Training Loss: {average_loss:.4f}, Validation Loss: {average_val_loss:.4f}')
    #
    #     # Early stopping
    #     if average_val_loss < best_val_loss:
    #         best_val_loss = average_val_loss
    #         counter = 0
    #     else:
    #         counter += 1
    #         if counter >= patience:
    #             print('Early stopping...')
    #             early_stopping = True
    #             break
    #
    # if not early_stopping:
    #     print('Training completed without early stopping.')

    # Predict food category for the test dataset
    model.eval()
    with torch.no_grad():
        # Get CI using MC dropout
        lower_ci, upper_ci = [], []
        y_pred_all = np.array([])
        for itg in range(0, 10):
            if itg == 9:
                X_test_batch = test_X[itg * round(len(test_X) / 10):]
            else:
                X_test_batch = test_X[itg * round(len(test_X) / 10): (itg + 1) * round(len(test_X) / 10)]
            X_test_batch = torch.tensor(X_test_batch, dtype=torch.float32).to(gpu)
            # test_X = torch.tensor(test_X, dtype=torch.float32).to(device)
            y_pred_dist, y_pred = predict_point(X_test_batch, model, num_samples=100)
            y_pred = y_sc.inverse_transform(y_pred.detach().cpu().numpy())
            y_pred_all = np.concatenate([y_pred_all, y_pred.reshape(-1)])
            y_pred_dist = y_sc.inverse_transform(y_pred_dist.permute(1, 0, 2).reshape(
                torch.Size([y_pred_dist.shape[1], y_pred_dist.shape[0]])).detach().cpu().numpy())
            # Create 95% confidence interval for population mean weight
            for ind in range(len(y_pred_dist)):
                lci = st.norm.interval(confidence=0.95, loc=np.mean(y_pred_dist[ind]), scale=st.sem(y_pred_dist[ind]))[0]
                uci = st.norm.interval(confidence=0.95, loc=np.mean(y_pred_dist[ind]), scale=st.sem(y_pred_dist[ind]))[1]
                lower_ci.append(lci)
                upper_ci.append(uci)

        ci = np.concatenate([np.array(lower_ci).reshape(-1, 1), np.array(upper_ci).reshape(-1, 1)], axis=1)

    return y_pred_all.reshape(-1,1), ci


def finetune(num_lb, train_X, test_X, train_dict, test_y, nutri, ft, device):
    # Individual task
    # Define empty placeholder
    # preds, ci = np.array([]), np.array([])
    for nc, fc in enumerate(train_dict.keys()):
        # torch.cuda.empty_cache()
        # Get X data for each task
        X_sc = train_X[train_dict[fc].index]
        X_sc_t = test_X[test_y[test_y['sc']==nc].index]
        # Get prediction for individual task
        y_msc, y_sc_ci = task_model(X_sc, X_sc_t, train_dict[fc][nutri], ft, device)
        # Concatenate the prediction from different food categories
        if nc == 0:
            preds = y_msc
            ci = y_sc_ci
        else:
            preds = np.concatenate([preds, y_msc], axis=0)
            ci = np.concatenate([ci, y_sc_ci], axis=0)

    # Save each menu item's true RRR and their food category
    # Define empty placeholder
    tests = pd.DataFrame()
    for nsc in range(num_lb):
        tests = pd.concat([tests, test_y[test_y['sc'] == nsc]], axis=0)
    # print('Overall MINT Test r-squared: %f' % r2_score(tests[nutri], preds))
    # print('Overall MINT Test RMSE: %f' % mean_squared_error(tests[nutri], preds, squared=False))

    return tests, preds, ci

def mint(n, train_X, test_X, train_y, test_y, device,  ndf='RRR'):
    # Divide dataset into categories
    train_dict = mint_preprocessing(n, train_y)
    # Finetune by the food category
    true_test, pred_test, ci_test = finetune(n, train_X, test_X, train_dict, test_y, ndf,
                                             ndf + '_weights_FT_ED.pth', device)
    # Save results in a dataframes
    pred = pd.DataFrame(pred_test, columns=['Predicted_'+ ndf]).reset_index(drop=True)
    test = pd.DataFrame(true_test).reset_index(drop=True)
    ci = pd.DataFrame(ci_test, columns=['lower_ci', 'upper_ci']).reset_index(drop=True)
    final_results = pd.concat([test, pred, ci], axis=1)
    # Export the result
    final_results.to_csv('MINT_pred_' + ndf + '_torch.csv', encoding='utf-8', index=False)

if __name__ == "__main__":
    # Set seed
    torch.manual_seed(1996)
    # Set gpu
    gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Read all edamam dataset as training and test it on users dataset
    X_train = np.load('../data/edamam_menu_embedding.npy')
    y_train = pd.read_csv('../data/edamam_nutrition_sample.csv')
    # number of clusters
    num_sc = len(set(y_train['sc']))
    # Run baseline for classifying the soft cluster pseudo-labels and get the predicted food category for the user dataset
    sc_CLS_model = baseline(num_sc, X_train, y_train, gpu)
    # Run simple mlp method and save weight for mint, save weights
    base_mlp(X_train, y_train, gpu)
    # eval
    sc_CLS_model.eval()
    # Read dataset for inference
    X_test = np.load('your_embeddings.npy')
    y_test = pd.read_csv('your_df.csv',low_memory=False, lineterminator='\n')
    # get pseudolabels
    y_pred_target = sc_CLS_model(torch.tensor(X_test, dtype=torch.float32).to(gpu))
    # Assign one label from the prediction
    ts_label = torch.argmax(y_pred_target, dim=1).cpu().numpy()
    # Save predicted food category for test dataset
    y_test['sc'] = pd.Series(ts_label)
    # Run MINT
    mint(num_sc, X_train, X_test, y_train, y_test, gpu)



