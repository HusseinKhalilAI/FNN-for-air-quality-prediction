from data_split import get_data_loaders
from model import PM_Model
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np



def train_model(model: nn.Module, train_loader: DataLoader, val_loader:DataLoader, test_loader: DataLoader):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_func = nn.MSELoss()

    for epoch in range(150+1):
        model.train()

        total_train_loss = 0.0

        for x, y_hat in train_loader:
            x = x.to(device)
            y_hat = y_hat.to(device)
            optimizer.zero_grad()
            y = model(x)
            loss = loss_func(y, y_hat)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x, y_hat in val_loader:
                x = x.to(device)
                y_hat = y_hat.to(device)
                y = model(x)
                val_loss = loss_func(y, y_hat)
                total_val_loss += val_loss.item()

        val_loss = total_val_loss / len(val_loader)

        if epoch % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch}, Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), "model.pt")

    model.eval()
    with torch.no_grad():
        actual = []
        predicted = []
        total = len(test_loader)
        total_test_loss = 0.0
        for i,(x, y_hat) in enumerate(test_loader):
            print(f"Evaluating... {i+1}/{total} batches", end="\r") 
            x = x.to(device)
            y_hat = y_hat.to(device)
            y = model(x)
            predicted.extend(y.cpu().detach().numpy())
            actual.extend(y_hat.cpu().detach().numpy())
            loss = loss_func(y, y_hat)
            total_test_loss += loss.item()
        test_loss = total_test_loss / len(test_loader)
        print(f"Test Loss: {test_loss:.4f}")
        


    plt.figure(figsize=(14, 7)) 
    plt.plot(actual, color="orange", label="Actual")
    plt.plot(predicted, color="blue", label="Predicted")
    plt.title("PM2.5 Prediction")
    plt.legend()
    plt.savefig("model_prediction.pdf")
    plt.show()
        


if __name__ == "__main__":

    path = r"C:\Users\hussa\air_quality_cleaned.csv"
    data = pd.read_csv(path)
    train_loader, val_loader, test_loader = get_data_loaders(data)

    for i, _ in train_loader:
        input_size = i.shape[1]
        break

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PM_Model(hidden_layers=3, drop_out=True, drop_value=0.25, input_size=input_size, hidden_size=64).to(device)

    train_model(model,train_loader, val_loader, test_loader)







