import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# Local imports
from config.config import MAX_NUM_EPOCHS
from data.data_loader import drop_outliers, load_data, preprocess_data
from model.housing_net import HousingNet, HousingNet2
from model.linear_reg import linear_reg_r2_score, rmse_linear_reg_model
from training.trainer import train_model
from visualisation.histogram import plot_house_values_histogram



def set_random_seeds(seed=42):
    """Set random seeds for reproducibility across all libraries."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def main():
    # Set random seeds for reproducibility
    set_random_seeds(42)
    
    print(f"Training model for a maximum of {MAX_NUM_EPOCHS} epochs")
    
    df = load_data()
    df_encoded = preprocess_data(df)

    # 2D Array
    X = df_encoded.drop('median_house_value', axis=1)  # input features only
    # 1D Array
    y = df_encoded['median_house_value']               # target variable

    # Standardize the features (mean = 0, std = 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Three-way split: Train (70%) / Validation (15%) / Test (15%)
    # First split: separate test set (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_scaled, y, test_size=0.15, random_state=42
    )
    
    # Second split: separate validation set from remaining data (15% of original = ~17.6% of remaining)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42  # 0.15 / 0.85 ≈ 0.176
    )


    print(f"Data split sizes:")
    print(f"  Training: {len(X_train)} samples ({len(X_train)/len(X_scaled)*100:.1f}%)")
    print(f"  Validation: {len(X_val)} samples ({len(X_val)/len(X_scaled)*100:.1f}%)")
    print(f"  Test: {len(X_test)} samples ({len(X_test)/len(X_scaled)*100:.1f}%)")

    # Convert to float32 tensors (required for neural networks)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    # Linear Regression Baseline (using train+validation for training, test for evaluation)
    X_train_val = np.vstack([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])
    lr_r2, lr_rmse = linear_reg_r2_score(X_train_val, y_train_val, X_test, y_test)
    
    print(f"Linear Regression R²: {lr_r2:.4f}")
    print(f"Linear Regression RMSE: {lr_rmse:.4f}")

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,    # Number of examples per batch
        shuffle=True      # Randomize the order each epoch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False     # No need to shuffle validation data
    )

    # X_train.shape[1] access the second element of the tuple
    # it's 13 because we have 13 features (aka columns <> 14-1=13, minus the one we predict)
    # setting it this way avoid hardcoding the number of features (we can change the data)
    model = HousingNet2(input_dim=X_train.shape[1])

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    train_model(model, train_loader, val_loader, optimizer, loss_fn, num_epochs=MAX_NUM_EPOCHS)

    # Final evaluation on test set (completely unseen data)
    print("\n" + "="*50)
    print("FINAL TEST SET EVALUATION")
    print("="*50)
    
    with torch.no_grad():  # Turn off gradients for evaluation
        test_predictions = model(X_test_tensor)
        test_loss = loss_fn(test_predictions, y_test_tensor)
        
        # Calculate R² score
        r2 = r2_score(y_test_tensor.numpy(), test_predictions.numpy())


    print(f"Test MSE: {test_loss.item():.2f}")
    print(f"RMSE: {torch.sqrt(test_loss).item():.2f}")
    print(f"🎯 R² Score: {r2:.4f}")
    
    # Compare with linear regression
    improvement = r2 - lr_r2
    print(f"Linear Regression R²: {lr_r2:.4f}")
    print(f"Improvement over Linear Regression: {improvement:+.4f}")


if __name__ == "__main__":
    main()
