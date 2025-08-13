import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(10)
# device = 'cuda:0'
device = 'cpu'

# Step 1: Create the Dataset
# Input data
X = torch.tensor([[1, 1], [1, -1], [0, 0]], dtype=torch.float32, device=device)
# Y = torch.tensor([1, 1, 0], dtype=torch.float32)
Y = torch.tensor([1, 1, -1], dtype=torch.float32, device=device)

hidden_dim = 2


# Step 2: Design the Neural Network for classification
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(2, hidden_dim, bias=False)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, 1, bias=False)
        # self.sigmoid = nn.Sigmoid()

        # Set weights of layer 2
        with torch.no_grad():
            # Assuming you want the weights to be [1, -1]
            self.layer2.weight[0][0] = 1.0
            self.layer2.weight[0][1] = -1.0

        # Freeze the weights of layer 2
        for param in self.layer2.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        # x = self.sigmoid(x)
        return x


def exponential_loss(output, target):
    loss = torch.exp(-target * output.squeeze())
    return torch.mean(loss)


model = SimpleNN().to(device)

# Step 3: Training with Cross-Entropy loss
criterion = nn.BCEWithLogitsLoss()

# optimizer = optim.Adam(model.parameters(), lr=0.01)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

from torch.utils.data import TensorDataset, DataLoader
dataset = TensorDataset(X, Y.unsqueeze(1))  # Y needs to have the same dimensions as outputs
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Step 4: Plot the Decision Boundary
def plot_decision_boundary(model, X, Y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32))
    Z = Z.detach().numpy()
    Z = (Z > 0.5).astype(int).reshape(xx.shape)

    plt.figure(figsize=(8, 7))
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Decision Boundary for Classification")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.savefig('minitoy.png')

# Training loop
for epoch in range(1000000):
    for inputs, targets in dataloader:
        # Forward pass
        outputs = model(inputs)
        loss = exponential_loss(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 10000 == 0:
        print(f'Epoch [{epoch}], Loss: {loss.item():.4f}')
        # Uncomment the line below to plot the decision boundary
        plot_decision_boundary(model, X.numpy(), Y.numpy())

print("Final weights of layer1:")
print(model.layer1.weight.data)