
from model.housing_net import HousingNet


def create_model(input_dim):
    # X_train.shape[1] access the second element of the tuple
    # it's 13 because we have 13 features (aka columns <> 14-1=13, minus the one we predict)
    # setting it this way avoid hardcoding the number of features (we can change the data)
    return HousingNet(input_dim=input_dim)


def train_model(model, train_loader, optimizer, loss_fn, num_epochs):
    for epoch in range(num_epochs):
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