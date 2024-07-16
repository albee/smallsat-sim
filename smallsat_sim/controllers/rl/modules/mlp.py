import torch.nn as nn

def mlp(sizes, activation, output_activation=nn.Identity):
    """
    Basic multilayer perceptron architecture.
    """
    modules = []

    for i in range(len(sizes) - 1):
        if i >= len(sizes) - 2:
            activation_function = output_activation
        else:
            activation_function = activation
        modules += [nn.Linear(sizes[i], sizes[i+1]), activation_function()]

    return nn.Sequential(*modules)
