from training import dataset_builder, encoder, head
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# --- Load data ---
df = pd.read_csv('./training/data/s00.csv')

# --- Scaling (Min-Max) ---
data_min = df.min()
data_max = df.max()
df_scaled = (df - data_min) / (data_max - data_min + 1e-8)  # scaled to [0,1]

# --- Dataset ---
dataset = dataset_builder.RandomMaskedSequenceDataset(
    df_scaled, sample_interval=1, sequence_length=10, mask_prob=0.3, mask_value=0.0
)

# --- Model ---
encoder_model = encoder.Encoder(input_dim=8, hidden_dims=[584, 128], kernel_sizes=[2, 2])
reconstruction_head = head.ReconstructionHead(encoder=encoder_model, out_dim=80)
model = reconstruction_head.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# --- Hyperparameters ---
batch_size = 64
num_epochs = 200
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dataloader ---
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- Loss & Optimizer ---
criterion = nn.MSELoss()   # reconstruction loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# --- Training Loop ---
epsilon = 1e-3  # for percentage error

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    epoch_pct_error = 0.0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        # Forward pass
        outputs = model(x)
        outputs = outputs.view_as(y)

        # Compute MSE loss
        loss = criterion(outputs, y)

        # Backward & optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate MSE
        epoch_loss += loss.item() * x.size(0)

        # --- Compute percentage error in original units ---
        # rescale outputs and targets back
        outputs_rescaled = outputs.detach() * (data_max.values - data_min.values) + data_min.values
        y_rescaled = y * (data_max.values - data_min.values) + data_min.values

        pct_error = torch.abs(outputs_rescaled - y_rescaled) / (torch.abs(y_rescaled) + epsilon) * 100
        epoch_pct_error += pct_error.sum().item()

    avg_loss = epoch_loss / len(dataset)
    avg_pct_error = epoch_pct_error / (len(dataset) * dataset.sequence_sample_count * y.shape[2])

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}, Avg % Error: {avg_pct_error:.2f}%")
