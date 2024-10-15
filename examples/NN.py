import numpy as np
import torch

device = "cuda:0"
lr = 1.0e-4
n_epochs = 100
batch_size = 25

# load dataset (state, batch, horizon)
history = np.load("output/multi_modality.npz")
state_history = np.swapaxes(history['mpc_states'], 0, 1)

# (batch, state, horizon)
X = torch.tensor(state_history[:, :, 0], device=device, dtype=torch.float32)
y = torch.tensor(state_history[:, 0:3, 1:], device=device, dtype=torch.float32)

model = torch.nn.Sequential(
    torch.nn.Linear(12, 30),
    torch.nn.ReLU(),
    torch.nn.Linear(30, 30),
    torch.nn.ReLU(),
    torch.nn.Linear(30, 3 * 33),
)
model.to(device)

if False:
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        for i in range(0, len(X), batch_size):
            Xbatch = X[i:i+batch_size]
            y_pred = model(Xbatch)
            ybatch = y[i:i+batch_size].reshape((batch_size, -1))
            loss = loss_fn(y_pred, ybatch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch: {epoch} Loss: {loss}")

    torch.save(model.state_dict(), "output/modality.pth")

model.load_state_dict(torch.load("output/modality.pth", weights_only=True))
model.eval()

nominal_state = torch.zeros(12)
nominal_state[0:3] = torch.tensor([-2.5, 1.0, 0.525])

noise = np.random.uniform(-2.5e-2, 2.5e-2, (100, nominal_state.size(0))).astype('f')
state = nominal_state + noise

with torch.no_grad():
    model.to('cpu')
    result = model(state).reshape((100, 3, 33))
    np.save("output/nn.npy", result.cpu().numpy())





