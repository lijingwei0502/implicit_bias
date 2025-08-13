import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt



seeds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

class ExponentialLoss(nn.Module):
    def __init__(self):
        super(ExponentialLoss, self).__init__()

    def forward(self, input, target):
        """
        计算指数损失
        :param input: 模型的预测输出，维度为 (N, *)，其中 * 表示任意数量的额外维度
        :param target: 真实的标签，维度与输入相同，值为 -1 或 1
        :return: 损失值
        """
        # 确保target的值为-1或1
        assert torch.all(torch.eq(target, 1) | torch.eq(target, -1))
        # 计算指数损失
        loss = torch.exp(-target * input)
        return torch.mean(loss)
    
for seed in seeds:
    torch.manual_seed(seed)
    # device = 'cuda:0'
    device = 'cpu'

    # Step 1: Create the Dataset
    # Input data
    X = torch.tensor([[1, 0], [-1, 0]], dtype=torch.float32, device=device)
    # Y = torch.tensor([1, 1, 0], dtype=torch.float32)
    Y = torch.tensor([1, -1], dtype=torch.float32, device=device)

    hidden_dim = 5


    # Step 2: Design the Neural Network for classification
    class SimpleNN(nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.layer1 = nn.Linear(2, hidden_dim, bias=True)
            self.relu = nn.ReLU()
            self.layer2 = nn.Linear(hidden_dim, 1, bias=False)
            # self.sigmoid = nn.Sigmoid()

            # # Set weights of layer 1
            # with torch.no_grad():
            #     # Assuming you want the weights to be [1, -1]
            #     self.layer1.weight[0][0] = 1.0
            #     self.layer1.weight[0][1] = -1.0
            #     self.layer1.weight[1][0] = -1.0
            #     self.layer1.weight[1][1] = -1.0
            #     self.layer1.weight[2][0] = 2.0
            #     self.layer1.weight[2][1] = -1.0
            #     self.layer1.weight[3][0] = 1.0
            #     self.layer1.weight[3][1] = -1.0
            #     self.layer1.weight[4][0] = 1.0
            #     self.layer1.weight[4][1] = -1.0
           


            # # Set weights of layer 2
            with torch.no_grad():
                # Assuming you want the weights to be [1, -1]
                self.layer2.weight[0][0] = 1.0
                self.layer2.weight[0][1] = -1.0
                self.layer2.weight[0][2] = 1.0
                self.layer2.weight[0][3] = -1.0
                self.layer2.weight[0][4] = -1.0


            

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

    # Step 3: Training with exponential loss
    criterion = ExponentialLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.01)
    optimizer = torch.optim.SGD(model.parameters(), lr=10000)

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
        plt.savefig(str(seed) + '_toy.png')

    # Training loop
    for epoch in range(2000000):
        # Forward pass
        outputs = model(X)
        # loss = criterion(outputs.squeeze(), Y)
        loss = exponential_loss(outputs, Y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10000 == 0:
            print(f'Epoch [{epoch}/1000], Loss: {loss.item():.4f}')
            # Uncomment the line below to plot the decision boundary
            plot_decision_boundary(model, X.numpy(), Y.numpy())

            print("Final weights of layer1:")
            print(model.layer1.weight.data)
            print("Final weights of layer2:")
            print(model.layer2.weight.data)