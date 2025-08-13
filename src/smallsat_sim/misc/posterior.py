import torch
import gpytorch
import matplotlib.pyplot as plt

# Define the GP model without the Scale Kernel
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.RBFKernel()
        self.covar_module.lengthscale = torch.tensor(0.5)  # Hardcoded lengthscale

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Training data: 20 random points between -2 and 2
train_x = 4 * torch.rand(20) - 2  # 20 random points between -2 and 2
train_y = torch.sin(train_x) + 0.1 * torch.randn(train_x.size())  # Add small noise to the data

# Initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-4))
likelihood.noise = torch.tensor(0.1)  # Hardcoded noise level

model = ExactGPModel(train_x, train_y, likelihood)

# Switch to evaluation mode directly (no training loop needed)
model.eval()
likelihood.eval()

# Print all model and likelihood parameters
print("\n--- GP Model Parameters ---")
for name, param in model.named_parameters():
    print(f"{name}: {param.data}")

print("\n--- Likelihood Parameters ---")
for name, param in likelihood.named_parameters():
    print(f"{name}: {param.data}")


# Print transformed model and likelihood parameters
print("\n--- Transformed GP Model Parameters ---")
print(f"RBF Kernel lengthscale: {model.covar_module.lengthscale.item()}")

print("\n--- Transformed Likelihood Parameters ---")
print(f"Likelihood noise: {likelihood.noise.item()}")

# Test points for prediction (for plotting)
test_x = torch.linspace(-4, 4, 100)

# Make predictions using the model
with torch.no_grad():
    observed_pred = model(test_x)

# Get the mean and confidence intervals
mean = observed_pred.mean
lower, upper = observed_pred.confidence_region()

# Plotting the results
plt.figure(figsize=(8, 6))

# Plot posterior mean as a red line
plt.plot(test_x.numpy(), mean.numpy(), 'r', label='Posterior Mean')

# Plot the shaded region (confidence interval) as a gray area
plt.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.3, label=r'$\mu_y \pm 2\sigma_y$')

# Plot training data as green '+'
plt.scatter(train_x.numpy(), train_y.numpy(), color='green', marker='+', s=100, label='Training Data')

# Plot 5 samples from the posterior as blue lines
for i in range(5):
    sampled_f = observed_pred.sample()  # Sample from the posterior
    plt.plot(test_x.numpy(), sampled_f.numpy(), 'b', alpha=0.5)

plt.title("GP Posterior with RBF Kernel (Fixed Noise and Lengthscale)")
plt.legend()
plt.ylim([-2, 2])
plt.show()
