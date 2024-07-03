import torch
import gpytorch
from matplotlib import pyplot as plt

# Define the mean function
class Mean(gpytorch.means.Mean):
    def __init__(self, lambda_decay=0.2):
        super().__init__()
        self.lambda_decay = lambda_decay

    def forward(self, x):
        t = x[:, 0]  # Only time dimension
        mean = torch.exp(-self.lambda_decay * t) * torch.cos(t)
        mean = torch.sigmoid(mean)
        return mean

# Define the RBF kernel with only time dependency
class Kernel(gpytorch.kernels.Kernel):
    def __init__(self, lengthscale_t=1.0, **kwargs):
        super(Kernel, self).__init__(**kwargs)
        self.lengthscale_t = lengthscale_t

    def forward(self, x1, x2, diag=False, **params):
        t1 = x1[:, 0]
        t2 = x2[:, 0]
        
        t_diff = (t1.unsqueeze(1) - t2.unsqueeze(0)) / self.lengthscale_t
        
        dist_t = t_diff ** 2
        
        if diag:
            return torch.exp(-0.5 * dist_t.diagonal())
        else:
            return torch.exp(-0.5 * dist_t)

# Define the GP Model
class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = Mean()
        self.covar_module = gpytorch.kernels.ScaleKernel(Kernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Generate training data
t_train = torch.linspace(0, 60, 100)
train_x = t_train.unsqueeze(1)

# Generate mean vector based on the custom mean function
mean_func = Mean()
train_y = mean_func(train_x)

# Initialize the likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = GPModel(train_x, train_y, likelihood)

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# Training loop
training_iterations = 50
for i in range(training_iterations):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    print(f'Iteration {i + 1}/{training_iterations} - Loss: {loss.item()}')
    optimizer.step()

# Switch to evaluation mode
model.eval()
likelihood.eval()

# Generate test data
t_test = torch.linspace(0, 50, 100)
test_x = t_test.unsqueeze(1)

# Make predictions
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(model(test_x))

# Get the mean and variance of the predictions
pred_mean = observed_pred.mean
pred_variance = observed_pred.variance

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t_test.numpy(), pred_mean.numpy(), label='Mean Prediction')
plt.fill_between(t_test.numpy(), 
                 (pred_mean - 2 * torch.sqrt(pred_variance)).numpy(),
                 (pred_mean + 2 * torch.sqrt(pred_variance)).numpy(), alpha=0.5, label='Confidence Interval')
plt.xlabel('Time')
plt.ylabel('Disturbance')
plt.title('GP Disturbance Prediction over Time')
plt.legend()
plt.show()
