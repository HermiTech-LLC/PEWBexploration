import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

def create_spacetime_grid(size, scale):
    """Create a grid representing spacetime."""
    x = np.linspace(-size, size, scale)
    y = np.linspace(-size, size, scale)
    z = np.linspace(-size, size, scale)
    x, y, z = np.meshgrid(x, y, z)
    return x, y, z

def refined_metric_tensor_with_wave(x, y, z, t, bubble_radius, density, speed, k, omega, sigma):
    """Refine the metric tensor for the warp bubble with additional wave terms."""
    r = np.sqrt(x**2 + y**2 + z**2) - speed * t
    wave_component = (1 + np.cos(k * x + omega * t))
    g_tt = -1
    g_xx = g_yy = g_zz = 1 + density * np.exp(-r**2 / sigma**2) * wave_component
    return g_tt, g_xx, g_yy, g_zz

def refined_energy_momentum_tensor_with_wave(g_tt, g_xx, g_yy, g_zz):
    """Refine the energy-momentum tensor from the refined metric tensor with wave terms."""
    T_tt = -g_tt
    T_xx = g_xx
    T_yy = g_yy
    T_zz = g_zz
    return T_tt, T_xx, T_yy, T_zz

def refined_warp_spacetime_dynamic_with_wave(x, y, z, bubble_radius, density, t, speed, k, omega, sigma):
    """Apply refined warp factor to spacetime grid with wave terms to create a dynamic warp bubble."""
    g_tt, g_xx, g_yy, g_zz = refined_metric_tensor_with_wave(x, y, z, t, bubble_radius, density, speed, k, omega, sigma)
    T_tt, T_xx, T_yy, T_zz = refined_energy_momentum_tensor_with_wave(g_tt, g_xx, g_yy, g_zz)
    r = np.sqrt(x**2 + y**2 + z**2) - speed * t
    warp_effect = density * np.exp(-r**2 / sigma**2) * (1 + np.cos(k * x + omega * t))
    return warp_effect, T_tt, T_xx, T_yy, T_zz

def smooth_warp_effect(warp_effect, sigma=1):
    """Smooth the warp effect using a Gaussian filter."""
    return gaussian_filter(warp_effect, sigma=sigma)

def plot_warped_spacetime_slices(x, y, z, warp_effect):
    """Plot slices of the warped spacetime grid in different planes."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    slice_idx = warp_effect.shape[2] // 2
    warp_slice = warp_effect[:, :, slice_idx]
    axes[0].contourf(x[:, :, slice_idx], y[:, :, slice_idx], warp_slice, cmap='viridis')
    axes[0].set_title('XY Plane')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    
    slice_idx = warp_effect.shape[0] // 2
    warp_slice = warp_effect[slice_idx, :, :]
    axes[1].contourf(y[slice_idx, :, :], z[slice_idx, :, :], warp_slice, cmap='viridis')
    axes[1].set_title('YZ Plane')
    axes[1].set_xlabel('Y')
    axes[1].set_ylabel('Z')
    
    slice_idx = warp_effect.shape[1] // 2
    warp_slice = warp_effect[:, slice_idx, :]
    axes[2].contourf(x[:, slice_idx, :], z[:, slice_idx, :], warp_slice, cmap='viridis')
    axes[2].set_title('XZ Plane')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Z')
    
    plt.colorbar(axes[0].contourf(x[:, :, slice_idx], y[:, :, slice_idx], warp_slice, cmap='viridis'), ax=axes, orientation='horizontal', fraction=0.05)
    plt.show()

def plot_energy_momentum(t, T_tt, T_xx, T_yy, T_zz):
    """Plot the energy-momentum tensor components over time."""
    plt.figure(figsize=(12, 6))
    plt.plot(t, T_tt, label='T_tt')
    plt.plot(t, T_xx, label='T_xx')
    plt.plot(t, T_yy, label='T_yy')
    plt.plot(t, T_zz, label='T_zz')
    plt.xlabel('Time Step')
    plt.ylabel('Energy-Momentum Tensor Component')
    plt.title('Energy-Momentum Tensor Components Over Time')
    plt.legend()
    plt.show()

def refined_comprehensive_analysis_with_wave(x, y, z, bubble_radius, density, speed, timesteps, time_interval, k, omega, sigma):
    """Perform comprehensive analysis of the refined warp bubble dynamics with wave terms."""
    t_values = np.arange(0, timesteps * time_interval, time_interval)
    T_tt_values = []
    T_xx_values = []
    T_yy_values = []
    T_zz_values = []

    for t in t_values:
        warp_effect, T_tt, T_xx, T_yy, T_zz = refined_warp_spacetime_dynamic_with_wave(x, y, z, bubble_radius, density, t, speed, k, omega, sigma)
        T_tt_values.append(np.mean(T_tt))
        T_xx_values.append(np.mean(T_xx))
        T_yy_values.append(np.mean(T_yy))
        T_zz_values.append(np.mean(T_zz))

        if t % 1 == 0:  # Plot every timestep
            warp_effect_smoothed = smooth_warp_effect(warp_effect, sigma=2.0)
            plot_warped_spacetime_slices(x, y, z, warp_effect_smoothed)

    plot_energy_momentum(t_values, T_tt_values, T_xx_values, T_yy_values, T_zz_values)

def run_comprehensive_analysis_with_wave(bubble_radius, density, speed, timesteps, time_interval, k, omega, sigma):
    """Run the comprehensive analysis for a given set of parameters with wave terms."""
    x, y, z = create_spacetime_grid(new_grid_size, new_grid_scale)
    refined_comprehensive_analysis_with_wave(x, y, z, bubble_radius, density, speed, timesteps, time_interval, k, omega, sigma)

# Parameters for the grid and simulation
new_grid_size = 10
new_grid_scale = 100
timesteps = 20
time_interval = 0.1

# Example parameters for the wave component
k = 1.0
omega = 2.0
sigma = 2.0

# Running a single configuration as an example
run_comprehensive_analysis_with_wave(bubble_radius=1.0, density=20.0, speed=1.0, timesteps=20, time_interval=0.1, k=k, omega=omega, sigma=sigma)
