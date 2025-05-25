import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

for learning_rate in [10.0, 1.0, 0.1, 0.01, 0.001, 0.0001]:
    # Define the neural network architecture
    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(2, 5)  # Input size: 2, Output size: 5
            self.fc2 = nn.Linear(5, 1)  # Input size: 5, Output size: 1
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = torch.relu(self.fc1(x))  # ReLU activation for hidden layer
            x = self.fc2(x)
            x = self.sigmoid(x)  # Sigmoid activation for output layer
            return x


    # Prepare the data
    X_train = torch.tensor([[1, 0], [-1, 0], [0, 0]], dtype=torch.float32)
    y_train = torch.tensor([[1], [1], [0]], dtype=torch.float32)

    # Define the model, loss function, and optimizer
    model = SimpleNet()
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # Stochastic Gradient Descent

    # Training loop
    num_epochs = max(int(1000 / learning_rate), 10000)
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 1000 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Plot decision boundary
    x_min, x_max = -2, 2
    y_min, y_max = -2, 2
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_tensor = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    Z = model(grid_tensor)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z.detach().numpy(), cmap=plt.cm.RdYlBu, alpha=0.8)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train.numpy().reshape(-1), cmap=plt.cm.RdYlBu, edgecolors='k')
    plt.title('Decision Boundary')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.savefig(f'decision_boundary_{learning_rate}.png')
    plt.show()
    plt.close()
