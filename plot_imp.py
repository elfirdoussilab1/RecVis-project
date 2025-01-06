import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Customize font size and label size
fontsize = 35  # Title font size
labelsize = 30  # Axis labels and colorbar labels

# Latex template
plt.rcParams.update({"text.usetex": True, "font.family": "STIXGeneral"})

# Function to find max coordinate
def find_max_coord(x, y, z):
    max_idx = np.argmax(z)
    return x[max_idx], y[max_idx]

# Load the data
data = np.load("heatmap_max_pool.npz")
x_valid = data["x"]
y_valid = data["y"]
z1_valid = data["r1"]
z2_valid = data["r2"]
z3_valid = data["r3"]

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(30, 6), constrained_layout=True)

# Define the blank area (triangular mask)
triangle = np.array([[1, 0], [0, 1], [1, 1]])

# Function to add a mask
def add_mask(ax):
    polygon = patches.Polygon(triangle, color='gray', alpha=0.5, zorder=2)
    ax.add_patch(polygon)

# Plot each quantity
heatmap1 = axes[0].tricontourf(x_valid, y_valid, z1_valid, levels=100, cmap='viridis')
axes[0].set_title('R@1', fontsize=fontsize)
axes[0].set_xlabel('$w_0$', fontsize=labelsize)
axes[0].set_ylabel('$w_1$', fontsize=labelsize)
axes[0].tick_params(axis='x', labelsize=labelsize)
axes[0].tick_params(axis='y', labelsize=labelsize)
cbar1 = fig.colorbar(heatmap1, ax=axes[0])
cbar1.ax.tick_params(labelsize=labelsize)
max_x1, max_y1 = find_max_coord(x_valid, y_valid, z1_valid)
axes[0].scatter(max_x1, max_y1, color='black', marker='*', s=300)
add_mask(axes[0])

heatmap2 = axes[1].tricontourf(x_valid, y_valid, z2_valid, levels=100, cmap='plasma')
axes[1].set_title('R@5', fontsize=fontsize)
axes[1].set_xlabel('$w_0$', fontsize=labelsize)
axes[1].set_ylabel('$w_1$', fontsize=labelsize)
axes[1].tick_params(axis='x', labelsize=labelsize)
axes[1].tick_params(axis='y', labelsize=labelsize)
cbar2 = fig.colorbar(heatmap2, ax=axes[1])
cbar2.ax.tick_params(labelsize=labelsize)
max_x2, max_y2 = find_max_coord(x_valid, y_valid, z2_valid)
axes[1].scatter(max_x2, max_y2, color='black', marker='*', s=300)
add_mask(axes[1])

heatmap3 = axes[2].tricontourf(x_valid, y_valid, z3_valid, levels=100, cmap='coolwarm')
axes[2].set_title('R_mean', fontsize=fontsize)
axes[2].set_xlabel('$w_0$', fontsize=labelsize)
axes[2].set_ylabel('$w_1$', fontsize=labelsize)
axes[2].tick_params(axis='x', labelsize=labelsize)
axes[2].tick_params(axis='y', labelsize=labelsize)
cbar3 = fig.colorbar(heatmap3, ax=axes[2])
cbar3.ax.tick_params(labelsize=labelsize)
max_x3, max_y3 = find_max_coord(x_valid, y_valid, z3_valid)
axes[2].scatter(max_x3, max_y3, color='black', marker='*', s=300)
add_mask(axes[2])

# Save figure
path = 'heatmap_max_pool.pdf'
fig.savefig(path, bbox_inches='tight')
print("Figure saved!")
