
import torch
from model.housing_net import HousingNet


def create_model(input_dim):
    # X_train.shape[1] access the second element of the tuple
    # it's 13 because we have 13 features (aka columns <> 14-1=13, minus the one we predict)
    # setting it this way avoid hardcoding the number of features (we can change the data)
    return HousingNet(input_dim=input_dim)

def train_model(model, train_loader, val_loader, optimizer, loss_fn, num_epochs, patience=50):
    """
    Train the model with validation monitoring and early stopping.
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: The optimizer
        loss_fn: Loss function
        num_epochs: Maximum number of epochs
        patience: Number of epochs to wait before early stopping
    """
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        num_train_batches = 0
        
        for batch_X, batch_y in train_loader:
            # Forward pass: predict
            predictions = model(batch_X)

            # Compute loss (predictions vs actual house value)
            loss = loss_fn(predictions, batch_y)

            # Back propagation process
            # We clear the gradients (otherwise they would accumulate)
            optimizer.zero_grad()

            # Calculate how much each parameter contributed to the loss
            # It outputs the gradients for each parameter (or it is for each neuron of each layer? I don't know)
            loss.backward()

            optimizer.step()
            
            # Accumulate loss for this epoch
            train_loss += loss.item()
            num_train_batches += 1
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                predictions = model(batch_X)
                loss = loss_fn(predictions, batch_y)
                val_loss += loss.item()
                num_val_batches += 1
        
        # Calculate average losses
        avg_train_loss = train_loss / num_train_batches
        avg_val_loss = val_loss / num_val_batches
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        # Print progress every 15 epochs
        if (epoch + 1) % 15 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f}")
            print(f"  Best Val Loss: {best_val_loss:.4f}")
            if patience_counter > 0:
                print(f"  Patience: {patience_counter}/{patience}")
            print()
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            print(f"Best validation loss: {best_val_loss:.4f}")
            # Restore best model
            model.load_state_dict(best_model_state)
            break
    
    # If we didn't trigger early stopping, load the best model anyway
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Training completed. Best validation loss: {best_val_loss:.4f}")

