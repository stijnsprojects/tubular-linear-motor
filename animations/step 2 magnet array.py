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

# Parameters
n_magnets = 8
strength = 5
gap = 0
length = 2
height = 0.5  # thickness of magnet rectangle
dipole_inset = 0.2  # distance from ends where monopoles sit
reverse_every_other = True  # Set to False for all aligned

# visuals
xbrim = 3
yrange = 10
field_density = 2  # density of vector field arrows

# Calculate total length including gaps
total_length = n_magnets * length + (n_magnets - 1) * gap
center_shift = total_length / 2

# Grid setup - shifted accordingly
x = np.linspace(-1*xbrim - center_shift, n_magnets * (length + gap) + xbrim - center_shift, int((xbrim + center_shift)*40))
y = np.linspace(-1*yrange/2, yrange/2, int(yrange*40))
X, Y = np.meshgrid(x, y)

Bx_total = np.zeros_like(X)
By_total = np.zeros_like(Y)

magnets = []

for i in range(n_magnets):
    # Position the magnet start, respecting the gap between magnets, then shift left to center at zero
    start_x = i * (length + gap) - center_shift
    end_x = start_x + length
    
    # Calculate dipole positions inset from edges
    if reverse_every_other and i % 2 == 1:
        north_x = end_x - dipole_inset
        south_x = start_x + dipole_inset
    else:
        north_x = start_x + dipole_inset
        south_x = end_x - dipole_inset
    
    magnets.append((start_x, 0, length, height, north_x, 0, south_x, 0))

    # Add the field of this magnet
    Bx, By = bar_magnet_field(X, Y, north_x, 0, south_x, 0, strength)
    Bx_total += Bx
    By_total += By

# Plot vector field
plt.figure(figsize=(12, 6))
plt.streamplot(X, Y, Bx_total, By_total, color=np.log(np.sqrt(Bx_total**2 + By_total**2)), cmap='viridis', density=field_density, linewidth=2, arrowsize=1.5, zorder=1)

# Plot magnets as rectangles and dipoles as red/blue dots
ax = plt.gca()
for (start_x, start_y, width, height, north_x, north_y, south_x, south_y) in magnets:
    rect = patches.Rectangle((start_x, -height/2), width, height, linewidth=1, edgecolor='black', facecolor='grey', zorder=2)
    ax.add_patch(rect)
    # Dipoles
    plt.scatter([north_x, south_x], [north_y, south_y], c=['red', 'blue'], s=75, zorder=3)


plt.xlim(x.min(), x.max())
plt.ylim(-1*yrange/2, yrange/2)  # same as vector field y limits
plt.gca().set_aspect('equal', adjustable='box')  # match aspect ratio
# remove x and y labels, ticks, and grid
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.show()