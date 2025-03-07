import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

torch.cuda.empty_cache()

def preprocessing(train_df):
    # set seed
    np.random.seed(1996)
    # Change category to categorical label: One hot encoding
    sc_tr = np.array(train_df).reshape(-1, 1)
    # sc_ts=np.array(y_test['sc']).reshape(len(y_test),1)
    # Integer label to one hot encoding dummy variable
    ohe = OneHotEncoder(sparse=False)
    ohe_train = ohe.fit_transform(sc_tr)
    # ohe_test = ohe.transform(sc_ts)

    return ohe_train

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


def train_AMDD(train_X, train_y, device):
    train_y = preprocessing(train_y)
    dim = train_X.shape[1]
    output_dim = 4
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
    epochs = 31
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
    #         #torch.save(model.state_dict(), 'AMDD_cls_desA_ES_weights.pth')
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


if __name__ == "__main__":
    # Set seed
    torch.manual_seed(1996)
    # Set gpu
    gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Use edamam's pseudo-labels from ChatGPT to predict the AMDD label for the inference datset
    # Read all edamam dataset as training and test it on inference df
    X_train = np.load('edamam_ft_embed_AMDD.npy') # preprocessed - sauces and others removed
    y_train = pd.read_csv('edamam_amdd_labels.csv') # preprocessed - sauces and others removed
    # Run baseline method and get the predicted food category for the test dataset
    AMDD_model = train_AMDD(X_train, y_train, gpu)
    # # Load best weights from ES
    # AMDD_model.load_state_dict(torch.load('AMDD_cls_desA_ES_weights.pth'))
    # Predict food category for the test dataset
    AMDD_model.eval()
    # Read inference dataset
    X_test = np.load('inference_FT_embedding.npy')
    # to torch and device
    X_test = torch.tensor(X_test, dtype=torch.float32).to(gpu)
    y_pred_target = AMDD_model(X_test)
    # Class probability
    #ts_prob = y_pred_target.detach().cpu().numpy()
    # Assign one label from the prediction
    ts_label = torch.argmax(y_pred_target, dim=1).cpu().numpy()

    # Export labels
    with open('inference_FT_AMDD_labels','wb') as fp:
        pickle.dump(ts_label, fp)