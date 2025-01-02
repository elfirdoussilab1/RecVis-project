# In this file, we will plot the heatmap
import numpy as np
import matplotlib.pyplot as plt

# Latex template
plt.rcParams.update({"text.usetex": True,"font.family": "STIXGeneral"})

# Load the data
data = np.load("heatmap_data_feasible_3quantities.npz")
x_valid = data["x"]
y_valid = data["y"]
z1_valid = data["r1"]
z2_valid = data["r2"]
z3_valid = data["r3"]

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(30, 5), constrained_layout=True)

# Plot each quantity
heatmap1 = axes[0].tricontourf(x_valid, y_valid, z1_valid, levels=100, cmap='viridis')
axes[0].set_title('f1(x, y)')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
fig.colorbar(heatmap1, ax=axes[0], label='f1(x, y)')

heatmap2 = axes[1].tricontourf(x_valid, y_valid, z2_valid, levels=100, cmap='plasma')
axes[1].set_title('f2(x, y)')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
fig.colorbar(heatmap2, ax=axes[1], label='f2(x, y)')

heatmap3 = axes[2].tricontourf(x_valid, y_valid, z3_valid, levels=100, cmap='coolwarm')
axes[2].set_title('f3(x, y)')
axes[2].set_xlabel('x')
axes[2].set_ylabel('y')
fig.colorbar(heatmap3, ax=axes[2], label='f3(x, y)')

# Save figure ...
# TODO
print("Figure saved !")