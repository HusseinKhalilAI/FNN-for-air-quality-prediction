import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd


def get_data_loaders(df: pd.DataFrame, batch_size: int = 32) -> tuple[DataLoader,DataLoader, DataLoader]:

    df = df.select_dtypes(include="number").dropna()

    X = df.drop(columns=["PM2.5", "No",'Unnamed: 0']).reset_index(drop=True)
    y_hat = df["PM2.5"] 

    X_90, X_test, y_90, y_test = train_test_split(X, y_hat, test_size=0.1, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_90, y_90, test_size=1/9, shuffle=False)  

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



   

    return train_loader, val_loader, test_loader





if __name__ == "__main__":
    path = r"C:\Users\hussa\air_quality_cleaned.csv"
    df = pd.read_csv(path)
    print(get_data_loaders(df,32))