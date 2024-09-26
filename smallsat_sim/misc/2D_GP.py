import torch
import gpytorch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def print_params(model):
    for name, param in model.named_parameters():
                print(f"Parameter {name} has shape {param.shape} and values:")
                print(param)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define the function we want to model
def linear_function(X):
    x1 = X[:, 0]
    x2 = X[:, 1]  # This will not affect the output
    return 2.0 * x1  # No constant offset

# Generate synthetic data
def generate_data(n_samples=3):
    X = np.random.uniform(-5, 5, (n_samples, 2))
    y = linear_function(X) + np.random.normal(0, 0.1, n_samples)  # Add some noise
    return X, y

# Convert data to torch tensors
X_train, y_train = generate_data()
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()

# Define the GP model
class LinearGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(LinearGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean(input_size=2)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.LinearKernel()
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Initialize the likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = LinearGPModel(X_train, y_train, likelihood)
print_params(model)

# Set into training mode
model.train()
likelihood.train()

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

#Training loop
training_iterations = 50
# for i in range(training_iterations):
#     optimizer.zero_grad()
#     output = model(X_train)
#     loss = -mll(output, y_train)
#     loss.backward()
#     optimizer.step()

#     if i % 10 == 0:
#         print(f"Iteration {i+1}/{training_iterations} - Loss: {loss.item()}")

# Set into evaluation mode
model.eval()
likelihood.eval()

# Introduce N_test_points parameter
N_test_points = 2  # You can modify this value to increase or decrease the number of test points

# Generate random test data using uniform distribution
X_test = np.random.uniform(-5, 5, (N_test_points, 2))
X_test = torch.from_numpy(X_test).float()

# Get predictions
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(model(X_test))

# Reshape the prediction output
pred_mean = observed_pred.mean.numpy()
pred_lower, pred_upper = observed_pred.confidence_region()

# Compute true values and mean prediction error
y_true = linear_function(X_test.numpy())
mean_prediction_error = np.mean(np.abs(pred_mean - y_true))

print(f"Mean Prediction Error: {mean_prediction_error:.4f}")

# Plotting
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for the GP mean prediction
ax.scatter(X_test[:, 0].numpy(), X_test[:, 1].numpy(), pred_mean, cmap='viridis', label='GP Prediction')

# Scatter plot for the training data
ax.scatter(X_train[:, 0].numpy(), X_train[:, 1].numpy(), y_train.numpy(), color='r', marker='x', label='Training Data')

# Customize the plot
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.set_title('3D GP Regression with Linear Function')

plt.show()

