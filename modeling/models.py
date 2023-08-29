import numpy as np
import pandas as pd
from modeling.utils import *
from modeling.nn_architectures import *
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.metrics import r2_score
import scipy.stats as st
import sys


def preprocessing(train_df):
    # set seed
    np.random.seed(2022)
    # Change category to categorical label: One hot encoding
    sc_tr = np.array(train_df['sc']).reshape(len(train_df), 1)
    # sc_ts=np.array(y_test['sc']).reshape(len(y_test),1)
    # Integer label to one hot encoding dummy variable
    ohe = OneHotEncoder(sparse=False)
    ohe_train = ohe.fit_transform(sc_tr)
    # ohe_test = ohe.transform(sc_ts)

    return ohe_train


def get_weighted_mean(num_lb, nutri, y_true_train_base, y_true_test_base, y_pred_test_base):
    # Get true mean nutrition score for each food category
    true_mean = list()
    for i in range(num_lb):
        true_mean.append(y_true_train_base.loc[y_true_train_base['sc'] == i][nutri].mean())
    # Make prediction by calculating weighted mean from food category prediction
    pred = list()
    for i in range(len(y_pred_test_base)):
        pred += [sum(true_mean * y_pred_test_base[i])]
    pred = pd.Series(pred)
    print('Baseline Test r-squared: %f' % r2_score(y_true_test_base[nutri], pred))


def baseline(n, train_X, test_X, train_y, test_y, score):
    # Set seed
    np.random.seed(2023)
    # One hot encoding for the classification label
    ohe_train = preprocessing(train_y)
    # Call classification model
    model = food_category_classification_model(train_X, n)
    # Set learning rate
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    # Compile, categorical_accuracy if one hot label
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])
    # print(model.summary())
    # Early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    # Fit model
    history = model.fit(X_train, ohe_train, validation_split=0.2, epochs=100, batch_size=64, verbose=2, callbacks=[es])
    # Predict food category for test dataset
    y_pred_target = model.predict(test_X)
    # Assign one label from the prediction, this will be used later for mint's fine-tuning process
    ts_label = list()
    for i in range(len(y_pred_target)):
        ts_label.append(np.argmax(y_pred_target[i]))
    # Get baseline result # User can change to different nutrition density score: 'RRR_m1'
    get_weighted_mean(n, score, train_y, test_y, y_pred_target)
    # Save predicted food category for test dataset
    test_y['sc'] = pd.Series(ts_label)

    return test_y


def predict_dist(X, model, num_samples):
    # Bayesian approximation using MC dropout, get distribution
    preds = [model(X, training=True) for _ in range(num_samples)]
    return np.hstack(preds)


def predict_point(X, model, num_samples):
    # Bayesian approximation using MC dropout, get mean of distribution
    pred_dist = predict_dist(X, model, num_samples)
    return pred_dist, pred_dist.mean(axis=1)


def mlp(train_X, test_X, train_y, test_y, score, kf):
    # Set seed
    np.random.seed(2023)
    # Normalize the nutrition density score # User can change and try with RRR-macro
    y_sc = StandardScaler()
    Ny_train = y_sc.fit_transform(np.array(train_y[score]).reshape(-1, 1))
    # Call nutrition classification model
    model = nutrition_prediction_model()
    # Set learning rate
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    # Compile, regression task
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
    # print(model.summary())
    # Early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    # Fit model
    history = model.fit(train_X, Ny_train, validation_split=0.2, epochs=100, batch_size=64, verbose=2, callbacks=[es])
    # Save model weights # User need to change the name of the weights for different fold
    model.save_weights(score + '_weights_' + kf)
    # Bayesian Approximation using MC dropout
    _, y_pred = predict_point(np.array(test_X), model, 100)
    y_pred = y_pred.reshape(-1, 1)
    print('MLP Test r-squared: %f' % r2_score(test_y[score], y_sc.inverse_transform(y_pred)))


def mint_preprocessing(num_lb, train_y, test_y):
    # food categories- training
    fc_name_lst, tr_fc_df_lst = list(), list()
    for i in range(num_lb):
        fc_name_lst.append('sc' + str(i))
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


def task_model(X_train_sm, X_test_sm, y_train_sm, weights):
    # Set seed
    np.random.seed(2023)
    # Normalize the nutrition density score # User can change and try with RRR-macro
    y_sc = StandardScaler()
    Ny_train_sm = y_sc.fit_transform(np.array(y_train_sm).reshape(-1, 1))
    target_model = nutrition_prediction_model()
    # Load weights
    target_model.load_weights(weights)
    # Set learning rate
    opt = keras.optimizers.Adam(learning_rate=0.001)
    # Compile, regression task
    target_model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
    # print(model.summary())
    # Early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    # Fit model
    history = target_model.fit(X_train_sm, Ny_train_sm, validation_split=0.2, epochs=100, batch_size=64, verbose=2,
                               callbacks=[es])
    # Bayesian Approximation using MC dropout
    y_pred_dist, y_pred = predict_point(np.array(X_test_sm), target_model, 100)
    y_pred = y_pred.reshape(-1, 1)
    # Inverse transform to real value
    y_pred = y_sc.inverse_transform(y_pred)
    # Get CI
    lower_ci, upper_ci = list(), list()
    # Create 95% confidence interval for population mean weight
    for i in range(len(y_pred_dist)):
        lci = st.t.interval(alpha=0.95, df=len(y_pred_dist[i]) - 1, loc=np.mean(y_pred_dist[i]),
                            scale=st.sem(y_pred_dist[i]))[0]
        uci = st.t.interval(alpha=0.95, df=len(y_pred_dist[i]) - 1, loc=np.mean(y_pred_dist[i]),
                            scale=st.sem(y_pred_dist[i]))[1]
        lower_ci.append(lci)
        upper_ci.append(uci)

    lower_ci = np.array(lower_ci).reshape(-1, 1)
    upper_ci = np.array(upper_ci).reshape(-1, 1)
    lower_ci = y_sc.inverse_transform(lower_ci)
    upper_ci = y_sc.inverse_transform(upper_ci)
    ci = np.concatenate([lower_ci, upper_ci], axis=1)

    return y_pred, ci


def finetune(train_X, test_X, train_dict, test_dict, nutri, ft):
    # Individual task
    # Define empty placeholder
    preds, ci = np.array([]), np.array([])
    for i, fc in enumerate(train_dict.keys()):
        # Get X data for each task
        X_sc = train_X.iloc[train_dict[fc].index]
        X_sc_t = test_X.iloc[test_dict[fc].index]
        # Get prediction for individual task
        y_sc, y_sc_ci = task_model(X_sc, X_sc_t, train_dict[fc][nutri], ft)
        # Concatenate the prediction from different food categories
        if i == 0:
            pred = y_sc
            ci = y_sc_ci
        else:
            preds = np.concatenate([preds, y_sc], axis=0)
            ci = np.concatenate([ci, y_sc_ci], axis=0)

    # Save each menu item's true RRR and their food category
    # Define empty placeholder
    tests = pd.DataFrame()
    for fc in test_dict.keys():
        tests = pd.concat([tests, test_dict[fc][['Name', nutri, 'sc']]], axis=0)
    # print('Overall MINT Test r-squared: %f' % r2_score(tests[nutri], preds))
    # print('Overall MINT Test RMSE: %f' % mean_squared_error(tests[nutri], preds, squared=False))

    return tests, preds, ci


def mint(n, train_X, test_X, train_y, test_y, score , kf):
    # Divide dataset into categories
    train_dict, test_dict = mint_preprocessing(n, train_y, test_y)
    # Finetune by the food category
    true_test, pred_test, ci_test = finetune(train_X, test_X, train_dict, test_dict, score, score + '_weights_' + kf)
    # Save results in a dataframes
    pred = pd.DataFrame(pred_test, columns=['Predicted ' + score]).reset_index(drop=True)
    test = pd.DataFrame(true_test).reset_index(drop=True)
    ci = pd.DataFrame(ci_test, columns=['lower_ci', 'upper_ci']).reset_index(drop=True)
    final_results = pd.concat([test, pred, ci], axis=1)
    # Print mint r2 score
    print('MINT Test r-squared: %f' % r2_score(final_results[score], final_results['Predicted ' + score]))
    # Export the result
    final_results.to_csv('MINT_pred_'+ score + '_' + kf + '.csv', encoding='utf-8', index=False)


if __name__ == "__main__":
    # Set seed
    np.random.seed(2023)
    # Read one of the fold's training and test index # Created using utils.set_fold
    nds = sys.argv[1]  # nutrition desnsity score # ex) RRR or RRR_m1
    fold = sys.argv[2]  # one of the folds to test on # ex) kf1 ~ kf5
    # User can change this part and try different folds
    train_ind = np.load(fold + '_tr.npy')
    test_ind = np.load(fold + '_ts.npy')
    # Read FastText embeddings
    X = np.load('../data/edamam_menu_embedding.npy')
    # Define training and test dataset
    X_train, X_test = X[train_ind], X[test_ind]
    # Read one of the fold that was created from the clustering.py
    y_train = pd.read_csv('target_' + fold + '.csv')
    num_sc = len(set(y_train['sc']))
    # Full dataset without fodd category
    y = pd.read_csv('../data/edamam_nutrition_sample.csv')
    y_test = y[test_ind]
    # Assign true nutrition density score
    y_test = nutrient_density_score(y_test)

    # Run baseline method and get the predicted food category for the test dataset
    y_test = baseline(num_sc, X_train, X_test, y_train, y_test)
    # Run simple mlp method and save weight for mint
    mlp(X_train, X_test, y_train, y_test, nds , fold)
    # Run MINT
    mint(num_sc, X_train, X_test, y_train, y_test)
