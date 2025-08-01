import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

def current_carrying_wire_field(x, y, x0, y0, I):
    mu0 = 4 * np.pi * 1e-7
    r_squared = (x - x0)**2 + (y - y0)**2
    r_squared[r_squared == 0] = 1e-12
    Bx = -mu0 * I * (y - y0) / (2 * np.pi * r_squared)
    By = mu0 * I * (x - x0) / (2 * np.pi * r_squared)
    return Bx, By

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
field_density = 2

total_length = n_magnets * length + (n_magnets - 1) * gap
center_shift = total_length / 2

x = np.linspace(-1*xbrim - center_shift, n_magnets * (length + gap) + xbrim - center_shift, int((xbrim + center_shift)*40))
y = np.linspace(-1*yrange/2, yrange/2, int(yrange*40))
X, Y = np.meshgrid(x, y)

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

# Precompute and save static PM field plot
Bx_pm_total = np.zeros_like(X)
By_pm_total = np.zeros_like(Y)

for (_, _, _, _, north_x, north_y, south_x, south_y) in magnets:
    Bx, By = bar_magnet_field(X, Y, north_x, north_y, south_x, south_y, strength)
    Bx_pm_total += Bx
    By_pm_total += By

fig, ax = plt.subplots()
ax.axis('off')  # Hide axes
ax.set_xlim(x.min(), x.max())
ax.set_ylim(-yrange/2, yrange/2)
ax.set_aspect('equal')

ax.streamplot(X, Y, Bx_pm_total, By_pm_total, color=np.log(np.sqrt(Bx_pm_total**2 + By_pm_total**2)), cmap='viridis', density=field_density, linewidth=1, arrowsize=0.75, zorder=1)

for (start_x, start_y, width, height, north_x, north_y, south_x, south_y) in magnets:
    rect = patches.Rectangle((start_x, -height/2), width, height, linewidth=1, edgecolor='black', facecolor='grey', zorder=2)
    ax.add_patch(rect)
    ax.scatter([north_x, south_x], [north_y, south_y], c=['red', 'blue'], s=15, zorder=3)

fig.savefig('static_pm_field.png', bbox_inches='tight', pad_inches=0, dpi=600)
plt.close(fig)

# Animation loop with PM field image as background
num_frames = 200
frames = []
static_pm_image = Image.open('static_pm_field.png')

for frame_idx in range(num_frames):
    print(f'Processing frame {frame_idx + 1}/{num_frames}')
    t = frame_idx / num_frames

    theta = 4 * np.pi * t
    I = 10
    IA = I * np.sin(theta - np.pi/3)
    IB = I * np.sin(theta)
    IC = I * np.sin(theta + np.pi/3)

    wires = [
        (-5.666667 + 8*t, 1, IA),
        (-5.000000 + 8*t, 1, IB),
        (-4.333333 + 8*t, 1, IC),
        (-3.666667 + 8*t, 1, -IA),
        (-3.000000 + 8*t, 1, -IB),
        (-2.333333 + 8*t, 1, -IC)
    ]

    fig, ax = plt.subplots()
    ax.imshow(static_pm_image, extent=(x.min(), x.max(), -yrange/2, yrange/2), aspect='auto', zorder=0)

    # --- Add rectangle around the conductors ---
    rect_x = -4 + 8*t - 2
    rect_y = -1.333333
    rect_width = 4
    rect_height = 2 * 1.333333
    carriage_rect = patches.Rectangle((rect_x, rect_y), rect_width, rect_height, linewidth=1, edgecolor='black', facecolor='none', zorder=3)
    ax.add_patch(carriage_rect)

    total_Fx = 0
    total_Fy = 0

    for wire_x, wire_y, current_I in wires:
        ix = np.abs(x - wire_x).argmin()
        iy = np.abs(y - wire_y).argmin()
        B_at_wire_x = Bx_pm_total[iy, ix]
        B_at_wire_y = By_pm_total[iy, ix]
        Fx = current_I * B_at_wire_y
        Fy = -current_I * B_at_wire_x

        ax.scatter([wire_x], [wire_y], c='orange', s=40, marker='o', zorder=4)
        ax.quiver(wire_x, wire_y, Fx, Fy, color='black', angles='xy', scale_units='xy', scale=40, width=0.005, zorder=5)

        return_wire_y = -wire_y
        ix_ret = np.abs(x - wire_x).argmin()
        iy_ret = np.abs(y - return_wire_y).argmin()
        B_at_return_x = Bx_pm_total[iy_ret, ix_ret]
        B_at_return_y = By_pm_total[iy_ret, ix_ret]
        Fx_ret = -current_I * B_at_return_y
        Fy_ret = current_I * B_at_return_x

        ax.scatter([wire_x], [return_wire_y], c='orange', s=40, marker='o', zorder=4)
        ax.quiver(wire_x, return_wire_y, Fx_ret, Fy_ret, color='black', angles='xy', scale_units='xy', scale=50, width=0.005, zorder=5)

        total_Fx += Fx + Fx_ret
        total_Fy += Fy + Fy_ret

    ax.quiver(-4 + 8*t, 0, total_Fx, total_Fy, color='black', angles='xy', scale_units='xy', scale=40, width=0.007, zorder=6)

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(-yrange/2, yrange/2)
    ax.set_aspect('equal')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=600)
    plt.close(fig)
    buf.seek(0)
    frames.append(Image.open(buf))

frames[0].save('step 5.gif', save_all=True, append_images=frames[1:], duration=60, loop=0)