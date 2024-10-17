import numpy as np
import torch
from torch.utils.data import DataLoader
from examples.nn_model import NeuralNetwork

device = 'cpu' # "cuda:0"
lr = 5.0e-4
n_epochs = 400
batch_size = 25

dataset = np.load("output/training.npz")
inputs = torch.tensor(dataset['inputs'], dtype=torch.float32)
outputs = torch.tensor(dataset['outputs'], dtype=torch.float32)
dataset = torch.utils.data.TensorDataset(inputs, outputs)

train_set, val_set = torch.utils.data.random_split(dataset, (0.8, 0.2))
train_dataloader = DataLoader(train_set, batch_size=batch_size)
test_dataloader = DataLoader(val_set, batch_size=batch_size)

model = NeuralNetwork().to(device)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(n_epochs):
    model.train_episode(train_dataloader, loss_fn, optimizer)
    print(f"[Episode {epoch}] \t Avg loss: {model.test_episode(test_dataloader, loss_fn):>8f}")

torch.save(model.state_dict(), "output/modality.pth")





