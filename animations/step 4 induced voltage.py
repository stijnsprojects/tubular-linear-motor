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

def render_field_image(X, Y, Bx_total, By_total, x_limits, y_limits, dpi=450):
    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    aspect_ratio = x_range / y_range

    fig_width = 6
    fig_height = fig_width / aspect_ratio

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)

    magnitude = np.log(np.sqrt(Bx_total**2 + By_total**2) + 1e-9)
    ax.streamplot(X, Y, Bx_total, By_total, color=magnitude, cmap='viridis', density=2, linewidth=1, arrowsize=0.75)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

# --- Setup constants ---
n_magnets = 8
strength = 5
gap = 0
length = 2
height = 0.5
dipole_inset = 0.2
reverse_every_other = True
xbrim = 3
yrange = 6

total_length = n_magnets * length + (n_magnets - 1) * gap
center_shift = total_length / 2

x = np.linspace(-1 * xbrim - center_shift, n_magnets * (length + gap) + xbrim - center_shift, 400)
y = np.linspace(-yrange/2, yrange/2, 200)
X, Y = np.meshgrid(x, y)

magnets = []
Bx_total = np.zeros_like(X)
By_total = np.zeros_like(Y)

for i in range(n_magnets):
    start_x = i * (length + gap) - center_shift
    end_x = start_x + length
    if reverse_every_other and i % 2 == 1:
        north_x = end_x - dipole_inset
        south_x = start_x + dipole_inset
    else:
        north_x = start_x + dipole_inset
        south_x = end_x - dipole_inset
    magnets.append((north_x, 0, south_x, 0))

    Bx, By = bar_magnet_field(X, Y, north_x, 0, south_x, 0, strength)
    Bx_total += Bx
    By_total += By

# Pre-render the field image
x_limits = (x.min(), x.max())
y_limits = (-yrange/2, yrange/2)
field_image = render_field_image(X, Y, Bx_total, By_total, x_limits, y_limits)

# --- Simulation parameters ---
v_x = 2.0    # fixed scalar velocity
L = 1.0      # conductor length
x_positions = np.linspace(x_limits[0], x_limits[1], 240)
emf_values = []

frames = []

for step, x_pos in enumerate(x_positions):
    print(f"Step {step+1}/{len(x_positions)}")

    Bx, By = 0.0, 0.0
    for north_x, north_y, south_x, south_y in magnets:
        Bx_m, By_m = bar_magnet_field(np.array([x_pos]), np.array([1]), north_x, north_y, south_x, south_y, strength)
        Bx += Bx_m[0]
        By += By_m[0]

    emf = v_x * By * L
    emf_values.append(emf)

    # Render the field and conductor for every second step
    if step % 2 == 0:

        fig, axs = plt.subplots(2, 1, figsize=(12, 6))

        # --- Plot field with magnets, poles, dashed line, and conductor ---
        axs[0].imshow(field_image, extent=(*x_limits, *y_limits), aspect='auto', zorder=0)

        for i, (north_x, north_y, south_x, south_y) in enumerate(magnets):
            start_x = i * (length + gap) - center_shift
            rect = patches.Rectangle((start_x, -height/2), length, height, linewidth=1, edgecolor='black', facecolor='grey', zorder=1)
            axs[0].add_patch(rect)
            axs[0].scatter([north_x, south_x], [north_y, south_y], c=['red', 'blue'], s=50, zorder=2)

        axs[0].plot(x, np.ones_like(x), 'k--', label='Conductor Path', zorder=3)
        axs[0].scatter([x_pos], [1], color='orange', s=100, label='Conductor', zorder=4)
        axs[0].set_xlim(x_limits)
        axs[0].set_ylim(y_limits)
        axs[0].set_xticks([])
        axs[0].set_yticks([])

        # --- Plot EMF vs position ---
        axs[1].plot(x_positions[:step+1], emf_values, linewidth=2, color='grey')
        sine_wave = 12.6 * np.sin(2 * np.pi * x_positions / 4 + np.pi/2)
        axs[1].plot(x_positions, sine_wave, label='Reference Sine Wave', linewidth=1, linestyle='dashed', color='gray')
        axs[1].axhline(0, linewidth=0.5, linestyle='dashed', color='gray')
        axs[1].set_xlabel('')
        axs[1].set_ylabel('')
        axs[1].set_xlim(x_limits)
        axs[1].set_ylim(-15, 15)
        axs[1].grid()
        axs[1].set_xticks([])
        axs[1].set_yticks([0])

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=450)
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf))

frames = [frame.convert("RGBA") for frame in frames]
frames = [frame.convert("P", palette=Image.ADAPTIVE) for frame in frames]

frames[0].save('step 4.gif', save_all=True, append_images=frames[1:], duration=45, loop=0)
