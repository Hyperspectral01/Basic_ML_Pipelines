import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from itertools import product

df = pd.read_csv("Iris.csv")
df = df.drop(columns=['Id'], errors='ignore')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class TorchPipeline:
    def __init__(self, hidden_size=16):
        self.scaler = StandardScaler()
        self.model = nn.Sequential(
            nn.Linear(4, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3)
        )

    def custom_input_preprocessing(self, X):
        X_copy = X.copy()
        X_copy[:, 0] = np.log(X_copy[:, 0] + 1e-9)
        return X_copy

    def fit(self, X_train, y_train, lr=0.01, epochs=100):
        X_proc = self.custom_input_preprocessing(X_train)
        X_scaled = self.scaler.fit_transform(X_proc)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.long)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = loss_fn(outputs, y_tensor)
            loss.backward()
            optimizer.step()

    def predict(self, X):
        X_proc = self.custom_input_preprocessing(X)
        X_scaled = self.scaler.transform(X_proc)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            preds = torch.argmax(logits, dim=1)
        return preds.numpy()

    def save(self, path):
        torch.save(self, path)

    @staticmethod
    def load(path):
        import torch.serialization
        with torch.serialization.safe_globals([TorchPipeline]):
            pipeline = torch.load(path, weights_only=False)
        return pipeline


    @staticmethod   #The way to tell it is a static method is it doesnt depend on self
    def tune(X_train, y_train, X_val, y_val):
        hidden_sizes = [8, 16, 32]
        lrs = [0.01, 0.001]
        epochs_list = [50, 100]
        best_acc = 0.0
        best_params = None
        best_pipeline = None
        print("--- Starting Hyperparameter Tuning ---")
        for hidden_size, lr, n_epochs in product(hidden_sizes, lrs, epochs_list):
            pipeline = TorchPipeline(hidden_size=hidden_size)
            pipeline.fit(X_train, y_train, lr=lr, epochs=n_epochs)
            preds = pipeline.predict(X_val)
            acc = np.mean(preds == y_val)
            print(f"Hidden Size: {hidden_size}, LR: {lr}, Epochs: {n_epochs} -> Validation Accuracy: {acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                best_params = (hidden_size, lr, n_epochs)
                best_pipeline = pipeline
        print("\n--- Tuning Finished ---")
        print(f"Best Accuracy: {best_acc:.4f}")
        print(f"Best Params (Hidden Size, LR, Epochs): {best_params}")
        return best_pipeline, best_params, best_acc

X_subtrain, X_val, y_subtrain, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
best_pipeline, best_params, best_acc = TorchPipeline.tune(X_subtrain, y_subtrain, X_val, y_val)
print("\n--- Evaluating Best Model on Test Set ---")
test_preds = best_pipeline.predict(X_test)
test_acc = np.mean(test_preds == y_test)
print(f"Final Test Accuracy: {test_acc:.4f}")
pipeline_path = "best_pipeline.pth"
best_pipeline.save(pipeline_path)
print(f"\nBest pipeline saved to {pipeline_path}")
loaded_pipeline = TorchPipeline.load(pipeline_path)
sample = X_test[0].reshape(1, -1)
pred = loaded_pipeline.predict(sample)
print(f"\nPrediction for sample {X_test[0]}: {pred[0]}")
print(f"Actual value for sample: {y_test[0]}")