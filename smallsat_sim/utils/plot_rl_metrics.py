import numpy as np
import matplotlib.pyplot as plt

from smallsat_sim.utils.logger import Logger


logger = Logger()

df = logger.load_log()
# print(df)

# Plot the mean returns
mean_return = df.dropna(subset=["mean_return"])["mean_return"]
plt.figure(figsize=(8, 6))
plt.plot(mean_return)
plt.xlabel("Epoch")
plt.ylabel("Mean Return")
plt.title("Average return over all environments")
plt.show()

# Plot the mean returns
mean_return = df.dropna(subset=["actor_loss"])["actor_loss"]
plt.figure(figsize=(8, 6))
plt.plot(mean_return)
plt.xlabel("Epoch")
plt.ylabel("Actor loss")
plt.title("Actor loss")
plt.show()
