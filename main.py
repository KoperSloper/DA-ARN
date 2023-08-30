from dataset import prepare_dataset
from model import initialize_model
from train import initialize_optimizer, validate, train_model, plot_predictions
from torch import nn

# Prepare data
train_loader, val_loader, test_loader, scaler_target = prepare_dataset()

# Initialize model
time_series = train_loader.dataset.tensors[0].shape[-1]  
hidden_dim1 = 64  
hidden_dim2 = 64  
timesteps = train_loader.dataset.tensors[0].shape[1] 
model = initialize_model(time_series, hidden_dim1, hidden_dim2, timesteps)

# Initialize optimizer and scheduler
optimizer, epoch_scheduler = initialize_optimizer(model)

# Train model
train_model(model=model,
            criterion=nn.MSELoss(),
            optimizer=optimizer,
            epoch_scheduler=epoch_scheduler,
            train_loader=train_loader,
            val_loader=val_loader)

# Validate model and plot predictions
val_loss, preds_val, targets_val = validate(model=model, criterion=nn.MSELoss(), val_loader=val_loader)
print(f"Validation loss: {val_loss:.6f}")
plot_predictions(preds_val, targets_val, scaler_target, "Validation: Actual vs predicted")

# Test model and plot predictions
test_loss, preds_test, targets_test = validate(model=model, criterion=nn.MSELoss(), val_loader=test_loader)
print(f"Test loss: {test_loss:.6f}")
plot_predictions(preds_test, targets_test, scaler_target, "Test: Actual vs predicted")