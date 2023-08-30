import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

def split_data(data, lookback):
    data_raw = data.to_numpy()

    X_list = []
    y_list = []
    target_list = []

    for index in range(len(data_raw) - lookback):
        X_list.append(data_raw[index: index + lookback, :-1])
        y_list.append(data_raw[index: index + lookback, -1].reshape(-1, 1))

        target_list.append(data_raw[index + lookback, -1])

    X = np.array(X_list)
    y = np.array(y_list)
    target = np.array(target_list)

    return [X, y, target]

def prepare_dataset(batch_size=128):
    data = pd.read_csv('nasdaq100_padding.csv')

    lookback = 16
    X, y, target = split_data(data, lookback)

    train_length = int(len(X) * 0.7)
    val_length = int(len(X) * 0.15)

    X_train = X[:train_length]
    y_train = y[:train_length]
    target_train = target[:train_length]

    X_val = X[train_length:train_length + val_length]
    y_val = y[train_length:train_length + val_length]
    target_val = target[train_length:train_length + val_length]

    X_test = X[train_length + val_length:]
    y_test = y[train_length + val_length:]
    target_test = target[train_length + val_length:]

    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    scaler_target = MinMaxScaler(feature_range=(0, 1))

    X_train_2D = X_train.reshape(-1, X_train.shape[-1])
    scaler_X.fit(X_train_2D)
    X_train = scaler_X.transform(X_train_2D).reshape(X_train.shape)

    y_train_2D = y_train.reshape(-1, y_train.shape[-1])
    scaler_y.fit(y_train_2D)
    y_train = scaler_y.transform(y_train_2D).reshape(y_train.shape)

    target_train_2D = target_train.reshape(-1, 1)
    scaler_target.fit(target_train_2D)
    target_train = scaler_target.transform(target_train_2D).flatten()

    X_val_2D = X_val.reshape(-1, X_val.shape[-1])
    X_val = scaler_X.transform(X_val_2D).reshape(X_val.shape)

    y_val_2D = y_val.reshape(-1, y_val.shape[-1])
    y_val = scaler_y.transform(y_val_2D).reshape(y_val.shape)

    target_val_2D = target_val.reshape(-1, 1)
    target_val = scaler_target.transform(target_val_2D).flatten()

    X_test_2D = X_test.reshape(-1, X_test.shape[-1])
    X_test = scaler_X.transform(X_test_2D).reshape(X_test.shape)

    y_test_2D = y_test.reshape(-1, y_test.shape[-1])
    y_test = scaler_y.transform(y_test_2D).reshape(y_test.shape)

    target_test_2D = target_test.reshape(-1, 1)
    target_test = scaler_target.transform(target_test_2D).flatten()

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    target_train_tensor = torch.tensor(target_train, dtype=torch.float32)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    target_val_tensor = torch.tensor(target_val, dtype=torch.float32)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    target_test_tensor = torch.tensor(target_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor, target_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor, target_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor, target_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, scaler_target