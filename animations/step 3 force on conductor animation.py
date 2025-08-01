import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio
from PIL import Image
import io

def monopole_field(x, y, mx, my, q):
    r_squared = (x - mx)**2 + (y - my)**2
    r_squared[r_squared == 0] = 1e-12
    Bx = q * (x - mx) / r_squared
    By = q * (y - my) / r_squared
    return Bx, By

def bar_magnet_field(x, y, x1, y1, x2, y2, strength):
    Bx1, By1 = monopole_field(x, y, x1, y1, strength)
    Bx2, By2 = monopole_field(x, y, x2, y2, -strength)
    return Bx1 + Bx2, By1 + By2

# almost no effect on the field from the wire, so we can use a simplified model for faster rendering
#def current_carrying_wire_field(x, y, x0, y0, I):
#    mu0 = 4 * np.pi * 1e-7
#    r_squared = (x - x0)**2 + (y - y0)**2
#    r_squared[r_squared == 0] = 1e-12
#    Bx = -mu0 * I * (y - y0) / (2 * np.pi * r_squared)
#    By = mu0 * I * (x - x0) / (2 * np.pi * r_squared)
#    return Bx, By

# Parameters
n_magnets = 8
strength = 5
gap = 0
length = 2
height = 0.5
dipole_inset = 0.2
reverse_every_other = True
xbrim = 3
yrange = 8
field_density = 2.5

total_length = n_magnets * length + (n_magnets - 1) * gap
center_shift = total_length / 2

x = np.linspace(-1*xbrim - center_shift, n_magnets * (length + gap) + xbrim - center_shift, int((xbrim + center_shift)*50))
y = np.linspace(-1*yrange/2, yrange/2, int(yrange*50))
X, Y = np.meshgrid(x, y)

# Precompute magnetic field from magnets
Bx_total = np.zeros_like(X)
By_total = np.zeros_like(Y)
magnets = []

for i in range(n_magnets):
    start_x = i * (length + gap) - center_shift
    end_x = start_x + length
    if reverse_every_other and i % 2 == 1:
        north_x = end_x - dipole_inset
        south_x = start_x + dipole_inset
    else:
        north_x = start_x + dipole_inset
        south_x = end_x - dipole_inset

    magnets.append((start_x, 0, length, height, north_x, 0, south_x, 0))

    Bx, By = bar_magnet_field(X, Y, north_x, 0, south_x, 0, strength)
    Bx_total += Bx
    By_total += By

# Pre-render the field background
fig, ax = plt.subplots(figsize=(12,6))
ax.streamplot(X, Y, Bx_total, By_total, color=np.log(np.sqrt(Bx_total**2 + By_total**2)), cmap='viridis', density=field_density, linewidth=2, arrowsize=1.5, zorder=1)

for (start_x, start_y, width, height, north_x, north_y, south_x, south_y) in magnets:
    rect = patches.Rectangle((start_x, -height/2), width, height, linewidth=1, edgecolor='black', facecolor='grey', zorder=2)
    ax.add_patch(rect)
    ax.scatter([north_x, south_x], [north_y, south_y], c=['red', 'blue'], s=75, zorder=3)

ax.set_xlim(x.min(), x.max())
ax.set_ylim(-yrange/2, yrange/2)
ax.set_xticks([])
ax.set_yticks([])
ax.grid(False)
fig.tight_layout()

buf = io.BytesIO()
plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=450)
plt.close(fig)
buf.seek(0)
field_image = Image.open(buf)

# --- Animation ---
wire_current = 10
wire_y = 1
x_positions = np.linspace(x.min(), x.max(), 120)

frames = []

for i, wire_x in enumerate(x_positions):
    print(f"Frame {i+1}/{len(x_positions)}")

    fig, ax = plt.subplots(figsize=(12,6))
    ax.imshow(field_image, extent=(x.min(), x.max(), -yrange/2, yrange/2), aspect='auto', zorder=0)

    # draw magnets
    for (start_x, start_y, width, height, north_x, north_y, south_x, south_y) in magnets:
        rect = patches.Rectangle((start_x, -height/2), width, height, linewidth=1, edgecolor='black', facecolor='grey', zorder=2)
        ax.add_patch(rect)
        ax.scatter([north_x, south_x], [north_y, south_y], c=['red', 'blue'], s=75, zorder=3)

    # Add dashed line showing conductor path
    ax.plot(x, np.ones_like(x), 'k--', linewidth=1, zorder=3)

    # Plot wire
    ax.scatter([wire_x], [wire_y], c='orange', s=150, marker='o', zorder=4)

    # Calculate B-field at wire location
    ix = np.abs(x - wire_x).argmin()
    iy = np.abs(y - wire_y).argmin()
    Bx_wire = Bx_total[iy, ix]
    By_wire = By_total[iy, ix]

    # Lorentz Force: F = I * (L x B), with current into the plane (-z)
    Fx = wire_current * By_wire
    Fy = -wire_current * Bx_wire

    # Plot force vector
    ax.quiver(wire_x, wire_y, Fx, Fy, color='black', scale=50, scale_units='xy', angles='xy', width=0.005, zorder=5)

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(-yrange/2, yrange/2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=450)
    plt.close(fig)
    buf.seek(0)
    frame = Image.open(buf)
    frames.append(frame)

frames = [frame.convert("RGBA") for frame in frames]
frames = [frame.convert("P", palette=Image.ADAPTIVE) for frame in frames]

frames[0].save('step 3.gif', save_all=True, append_images=frames[1:], duration=25, loop=0)