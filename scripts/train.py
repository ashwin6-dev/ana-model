from training import dataset_builder, encoder, head
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# --- Load & concatenate CSVs ---
dfs = [pd.read_csv(f'./training/data/s0{i}.csv') for i in range(7)]
df = pd.concat(dfs, ignore_index=True)

# --- Scaling (Min-Max) ---
data_min = df.min()
data_max = df.max()
dfs_scaled = [(df - data_min) / (data_max - data_min + 1e-8) for df in dfs]

masked_dataset = dataset_builder.SequenceDataset(
    dfs=dfs_scaled,
    sample_interval=1,
    sequence_length=10,
    transform=dataset_builder.masked_transform(mask_prob=0.3, mask_value=0.0)
)

recon_dataset = dataset_builder.SequenceDataset(
    dfs=dfs_scaled,
    sample_interval=1,
    sequence_length=10,
    transform=dataset_builder.identity_transform()
)
# --- Dataloaders ---
batch_size = 64
masked_loader = DataLoader(masked_dataset, batch_size=batch_size, shuffle=True)
recon_loader = DataLoader(recon_dataset, batch_size=batch_size, shuffle=True)

# --- Model ---
encoder_model = encoder.Encoder(input_dim=8, hidden_dims=[584, 128], kernel_sizes=[4, 2])
masked_head = head.ReconstructionHead(encoder=encoder_model, out_dim=80)
recon_head = head.ReconstructionHead(encoder=encoder_model, out_dim=80)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
masked_head.to(device)
recon_head.to(device)

# --- Loss & Optimizer ---
criterion = nn.MSELoss()
optimizer = optim.Adam(list(encoder_model.parameters()) +
                       list(masked_head.parameters()) +
                       list(recon_head.parameters()), lr=1e-3)

# --- Training Loop ---
num_epochs = 200
epsilon = 1e-3

for epoch in range(num_epochs):
    masked_iter = iter(masked_loader)
    recon_iter = iter(recon_loader)

    total_masked_loss = 0.0
    total_recon_loss = 0.0
    total_masked_pct = 0.0
    total_recon_pct = 0.0

    steps = max(len(masked_loader), len(recon_loader))

    for _ in range(steps):
        # --- Get batches ---
        try:
            x_masked, y_masked = next(masked_iter)
            x_masked, y_masked = x_masked.to(device), y_masked.to(device)
        except StopIteration:
            x_masked, y_masked = None, None

        try:
            x_recon, y_recon = next(recon_iter)
            x_recon, y_recon = x_recon.to(device), y_recon.to(device)
        except StopIteration:
            x_recon, y_recon = None, None

        optimizer.zero_grad()

        # --- Masked head ---
        if x_masked is not None:
            out_masked = masked_head(x_masked).view_as(y_masked)
            loss_masked = criterion(out_masked, y_masked)
            loss_masked.backward(retain_graph=True)  # retain_graph so encoder grads can accumulate
            total_masked_loss += loss_masked.item() * x_masked.size(0)

            # Percentage error in original units
            out_rescaled = out_masked.detach() * (data_max.values - data_min.values) + data_min.values
            y_rescaled = y_masked * (data_max.values - data_min.values) + data_min.values
            pct_error = torch.abs(out_rescaled - y_rescaled) / (torch.abs(y_rescaled) + epsilon) * 100
            total_masked_pct += pct_error.sum().item()

        # --- Reconstruction head ---
        if x_recon is not None:
            out_recon = recon_head(x_recon).view_as(y_recon)
            loss_recon = criterion(out_recon, y_recon)
            loss_recon.backward()
            total_recon_loss += loss_recon.item() * x_recon.size(0)

            # Percentage error in original units
            out_rescaled = out_recon.detach() * (data_max.values - data_min.values) + data_min.values
            y_rescaled = y_recon * (data_max.values - data_min.values) + data_min.values
            pct_error = torch.abs(out_rescaled - y_rescaled) / (torch.abs(y_rescaled) + epsilon) * 100
            total_recon_pct += pct_error.sum().item()

        optimizer.step()

    # --- Average losses and percentage errors ---
    avg_masked_loss = total_masked_loss / len(masked_dataset)
    avg_masked_pct = total_masked_pct / (len(masked_dataset) * masked_dataset.sequence_sample_count * y_masked.shape[2])

    avg_recon_loss = total_recon_loss / len(recon_dataset)
    avg_recon_pct = total_recon_pct / (len(recon_dataset) * recon_dataset.sequence_sample_count * y_recon.shape[2])

    print(f"Epoch [{epoch+1}/{num_epochs}] | Masked Loss: {avg_masked_loss:.6f}, Avg % Error: {avg_masked_pct:.2f}% | "
          f"Reconstruction Loss: {avg_recon_loss:.6f}, Avg % Error: {avg_recon_pct:.2f}%")
