import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams['savefig.dpi'] = 800

def monopole_field(x, y, mx, my, q):
    r_squared = (x - mx)**2 + (y - my)**2
    r_squared[r_squared == 0] = 1e-12
    Bx = q * (x - mx) / r_squared
    By = q * (y - my) / r_squared
    return Bx, By

def bar_magnet_field(x, y, x1, y1, x2, y2, strength):
    Bx1, By1 = monopole_field(x, y, x1, y1, strength)
    Bx2, By2 = monopole_field(x, y, x2, y2, -strength)
    Bx = Bx1 + Bx2
    By = By1 + By2
    return Bx, By

# Parameters for a single magnet
strength = 5
length = 2
height = 0.5
dipole_inset = 0.2

xbrim = 3
yrange = 3.5
field_density = 1

# Grid setup
x = np.linspace(-xbrim, length + xbrim, 200)
y = np.linspace(-yrange/2, yrange/2, 160)
X, Y = np.meshgrid(x, y)

# Dipole positions
start_x = 0
end_x = length
north_x = start_x + dipole_inset
south_x = end_x - dipole_inset

# Magnetic field
Bx, By = bar_magnet_field(X, Y, north_x, 0, south_x, 0, strength)

# Plot vector field
plt.figure(figsize=(12, 6))
plt.streamplot(X, Y, Bx, By, color=np.log(np.sqrt(Bx**2 + By**2)), cmap='viridis', density=field_density, zorder=1, linewidth=2, arrowsize=1.5)

# Plot the magnet rectangle
ax = plt.gca()
rect = patches.Rectangle((start_x, -height/2), length, height, linewidth=1, edgecolor='black', facecolor='grey', zorder=2)
ax.add_patch(rect)

# Plot dipoles
plt.scatter([north_x, south_x], [0, 0], c=['red', 'blue'], s=100, zorder=3)

plt.xlim(x.min(), x.max())
plt.ylim(-yrange/2, yrange/2)
plt.gca().set_aspect('equal', adjustable='box')
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.show()