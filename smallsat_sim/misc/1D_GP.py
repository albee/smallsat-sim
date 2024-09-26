import torch
import gpytorch
from matplotlib import pyplot as plt

# Define the model class
class LinearGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(LinearGPModel, self).__init__(train_x, train_y, likelihood)
        # Define the mean as zero
        self.mean_module = gpytorch.means.ZeroMean()
        # Define a linear kernel without an outputscale
        self.covar_module = gpytorch.kernels.LinearKernel() + gpytorch.kernels.RBFKernel()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def display_parameters(model, message):
    print(f"\n{message}")
    for name, param in model.named_parameters():
        print(f"{name}: {param.data}")

def train_and_test_gp(num_train_points, num_test_points, slope, training_iter=100):
    # Generate training data
    train_x = torch.linspace(0.5, 1, num_train_points)
    # Ground truth is y = slope * x (without noise)
    true_y = slope * train_x + 0.5
    # Add scaled noise to the training data
    noise_scale = 0.01 * slope  # Scale noise with the slope
    train_y = true_y + torch.randn(train_x.size()) * noise_scale  # y = slope * x + scaled noise

    # Define the likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = LinearGPModel(train_x, train_y, likelihood)

    # Display the initial parameters
    display_parameters(model, "Initial model parameters:")

    # Set the model into training mode
    model.train()
    likelihood.train()

    # Use the Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # Use the marginal log likelihood as the loss function
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Training loop
    if False:
        for i in range(training_iter):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Iter {i + 1}/{training_iter} - Loss: {loss.item()}")

    # Display the parameters after training
    display_parameters(model, "Model parameters after training:")

    # Set the model into evaluation mode
    model.eval()
    likelihood.eval()

    # Make predictions on test data
    test_x = torch.linspace(0, 1, num_test_points)
    true_test_y = slope * test_x  # Ground truth for test data
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = model(test_x)

    # Extract the posterior variance (diagonal of covariance matrix)
    posterior_cov = observed_pred.variance

    # Plot the results
    with torch.no_grad():
        f, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
        
        # Plot mean prediction and confidence intervals
        lower, upper = observed_pred.confidence_region()
        ax1.plot(test_x.numpy(), observed_pred.mean.numpy(), label='Mean prediction')
        ax1.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5, label='Confidence')
        ax1.plot(train_x.numpy(), train_y.numpy(), 'k*', label='Training data')
        ax1.plot(test_x.numpy(), true_test_y.numpy(), 'r--', label=f'Ground truth (y={slope}x)')
        ax1.legend()
        ax1.set_title(f'Linear GP Fit (Train: {num_train_points}, Test: {num_test_points})')

        # Plot the posterior variance
        ax2.plot(test_x.numpy(), posterior_cov.numpy(), 'b', label='Posterior Variance')
        ax2.set_title('Posterior Variance of Test Points')
        ax2.set_xlabel('Test Point')
        ax2.set_ylabel('Variance')
        ax2.legend()

        plt.tight_layout()
        plt.show()

# Example usage
num_train_points = 100  # Number of training points
num_test_points = 50    # Number of test points
slope = 0.0              # Slope of the linear function (y = slope * x)
train_and_test_gp(num_train_points, num_test_points, slope)
