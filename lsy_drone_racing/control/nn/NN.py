import numpy as np
import torch
from torch.utils.data import DataLoader
from lsy_drone_racing.control.nn.nn_model import NeuralNetwork

device = "cuda:0"
lr = 1.0e-5
n_epochs = 600
batch_size = 256

dataset = np.load("output/race_data.npz")
inputs = torch.tensor(dataset['local_obs'], dtype=torch.float32)
outputs = torch.tensor(dataset['local_actions'], dtype=torch.float32)[:, :, :16]
dataset = torch.utils.data.TensorDataset(inputs, outputs)

train_set, val_set = torch.utils.data.random_split(dataset, (0.85, 0.15))
train_dataloader = DataLoader(train_set, batch_size=batch_size)
test_dataloader = DataLoader(val_set, batch_size=batch_size)

model = NeuralNetwork(23, hidden_size=300).to(device)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(n_epochs):
    model.train_episode(train_dataloader, loss_fn, optimizer)
    print(f"[Episode {epoch}] \t Avg loss: {model.test_episode(test_dataloader, loss_fn):>8f}")


torch.save(model.state_dict(), "output/modality.pth")





