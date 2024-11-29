import torch

# Define model
class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_size, hidden_size = 90, device='cpu'):
        super().__init__()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 14 * 16),
        )
        self.device = device

    def forward(self, x):
        y = self.linear_relu_stack(x)
        return y

    def to(self, device):
        self.device = device
        return super().to(device=device)

    def train_episode(self, dataloader, loss_fn, optimizer):
        self.train()
        for X, y in dataloader:
            X, y = X.to(self.device), y.to(self.device)

            # Compute prediction error
            Y = self(X)
            loss = loss_fn(Y, y.reshape((len(X), -1)))

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    def test_episode(self, dataloader, loss_fn):
        self.eval()
        num_batches = len(dataloader)

        test_loss = 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self(X)
                test_loss += loss_fn(pred, y.reshape((len(X), -1))).item()

        test_loss /= num_batches
        return test_loss