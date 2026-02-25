import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 1. create training data
x = torch.linspace(-10, 10, 100).reshape(-1, 1)
y = 2 * x + 3

# 2. define neural network
model = nn.Sequential(
    nn.Linear(1, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)

# 3. define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. training loop
losses = []

for epoch in range(200):
    predictions = model(x)
    loss = criterion(predictions, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

print("Learned parameters:")
for name, param in model.named_parameters():
    print(name, param.data)

# 5. plot results
plt.scatter(x.detach().numpy(), y.detach().numpy(), label="True")
plt.scatter(x.detach().numpy(), model(x).detach().numpy(), label="Predicted")
plt.legend()
plt.show()