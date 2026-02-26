"""
Script to read binary h and A trajectory files from C++ simulation.

The binary file contains interleaved float32 values: h, A, h, A, ...
The metadata CSV contains simulation parameters.
"""

from probability_flux_functions import *
import numpy as np
import pandas as pd
from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib as mpl
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],'size':20})
rc('text', usetex=True) # For nice LaTeX formatting in plots. True for fonts serif, False otherwise
rc('axes', titlesize=24, labelsize=24)
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def read_h_A_trajectory(bin_filename):
    """
    Read h and A trajectory from binary file.
    
    Parameters:
    -----------
    bin_filename : str
        Path to the .bin file containing interleaved h and A values
        
    Returns:
    --------
    h_values : np.ndarray
        Array of h values
    A_values : np.ndarray
        Array of A (expansion parameter) values
    """
    # Read all float32 values from binary file
    data = np.fromfile(bin_filename, dtype=np.float32)
    
    # Separate into h and A arrays (interleaved: h, A, h, A, ...)
    h_values = data[0::2]  # Every even index (0, 2, 4, ...)
    A_values = data[1::2]  # Every odd index (1, 3, 5, ...)
    
    return h_values, A_values


def read_metadata(bin_filename):
    """
    Read metadata CSV file associated with binary trajectory file.
    
    Parameters:
    -----------
    bin_filename : str
        Path to the .bin file (will look for corresponding _metadata.csv)
        
    Returns:
    --------
    metadata : pd.Series
        Series containing simulation parameters
    """
    # Construct metadata filename
    csv_filename = bin_filename.replace('.bin', '_metadata.csv')
    
    # Read metadata
    df = pd.read_csv(csv_filename)
    
    # Return as Series (assuming single row)
    return df.iloc[0]


def plot_trajectories(h_values, A_values, metadata=None, max_points=100000):
    """
    Plot h and A trajectories.
    
    Parameters:
    -----------
    h_values : np.ndarray
        Array of h values
    A_values : np.ndarray
        Array of A values
    metadata : pd.Series, optional
        Metadata containing simulation parameters
    max_points : int
        Maximum number of points to plot (for performance)
    """
    # Downsample if needed for plotting
    if len(h_values) > max_points:
        step = len(h_values) // max_points
        h_plot = h_values[::step]
        A_plot = A_values[::step]
        time_plot = np.arange(len(h_plot)) * step
    else:
        h_plot = h_values
        A_plot = A_values
        time_plot = np.arange(len(h_plot))
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot h trajectory
    axes[0].plot(time_plot, h_plot, linewidth=0.5, alpha=0.7)
    axes[0].set_ylabel(r'$h$', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    
    # Plot A trajectory
    axes[1].plot(time_plot, A_plot, linewidth=0.5, alpha=0.7, color='C1')
    axes[1].set_xlabel('Time step', fontsize=14)
    axes[1].set_ylabel(r'$A$', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    # Add title with metadata if available
    if metadata is not None:
        title = (f"$\\epsilon={metadata['epsilon']:.3f}$, $f={metadata['f']:.2f}$, "
                f"$\\tau_p={metadata['tau_p']:.3f}$, $T={metadata['T']:.2f}$")
        fig.suptitle(title, fontsize=12)
    
    plt.tight_layout()
    return fig, axes


def compute_circulation_bootstrap(h_abs, L, t_total,
                                  hcenters, Lcenters,
                                  h_start, h_end, L_start, L_end,
                                  n_bootstrap=100, block_size=2):
    """
    Compute circulation with block-bootstrap uncertainty estimation.

    Rather than resampling the raw trajectory on each iteration, this function
    extracts the transition event list once and resamples that. Each event is a
    pair of consecutive bin-index pairs that constitutes an axis-aligned
    nearest-neighbour transition. Resampling events with replacement and
    rebuilding the count matrices is equivalent to block-resampling the
    trajectory, but operates on a list that is at most O(N) and typically much
    shorter, making the bootstrap loop fast regardless of trajectory length.

    Block size controls how many consecutive events are drawn together, 
    preserving pairwise transition correlations following Battle et al. (2016).

    Parameters:
    -----------
    h_abs : np.ndarray
        Array of |h| values (absolute value already taken)
    L : np.ndarray
        Array of L values
    t_total : float
        Total simulated time (in reduced units)
    hcenters, Lcenters : np.ndarray
        Bin centres from the main histogram
    h_start, h_end, L_start, L_end : float
        Desired contour boundaries (snapped to nearest bin centres internally)
    n_bootstrap : int
        Number of bootstrap resamples
    block_size : int
        Number of consecutive events per block (default: 2, following Battle et al.)

    Returns:
    --------
    circ_mean : float
        Bootstrap mean circulation
    circ_std : float
        Bootstrap standard deviation (ddof=1)
    circ_samples : np.ndarray, shape (n_bootstrap,)
        Individual bootstrap circulation values
    """
    n_x1_bins = len(hcenters)
    n_x2_bins = len(Lcenters)

    # ------------------------------------------------------------------ #
    # Step 1: bin the full trajectory once, reusing the edges that
    # produced hcenters/Lcenters (reconstruct edges from centres).
    # ------------------------------------------------------------------ #
    dh = hcenters[1] - hcenters[0]
    dL = Lcenters[1] - Lcenters[0]
    h1edges = np.append(hcenters - dh/2, hcenters[-1] + dh/2)
    L1edges = np.append(Lcenters - dL/2, Lcenters[-1] + dL/2)

    x1_bins = np.clip(np.digitize(h_abs, h1edges) - 1, 0, n_x1_bins - 1)
    x2_bins = np.clip(np.digitize(L,     L1edges) - 1, 0, n_x2_bins - 1)

    dx1 = x1_bins[1:] - x1_bins[:-1]
    dx2 = x2_bins[1:] - x2_bins[:-1]

    x1_curr = x1_bins[1:]
    x2_curr = x2_bins[1:]
    x1_prev = x1_bins[:-1]
    x2_prev = x2_bins[:-1]

    same_x2 = (dx2 == 0)
    same_x1 = (dx1 == 0)

    # ------------------------------------------------------------------ #
    # Step 2: extract the four event lists (one per transition type).
    # Each event stores the bin index at which it should be accumulated.
    # ------------------------------------------------------------------ #
    # left → current  (accumulated at current bin)
    m = same_x2 & (dx1 == 1)
    ev_ltc = np.stack([x1_curr[m], x2_curr[m]], axis=1)  # shape (K, 2)

    # current → left  (accumulated at previous bin)
    m = same_x2 & (dx1 == -1)
    ev_ctl = np.stack([x1_prev[m], x2_prev[m]], axis=1)

    # below → current  (accumulated at current bin)
    m = same_x1 & (dx2 == 1)
    ev_btc = np.stack([x1_curr[m], x2_curr[m]], axis=1)

    # current → below  (accumulated at previous bin)
    m = same_x1 & (dx2 == -1)
    ev_ctb = np.stack([x1_prev[m], x2_prev[m]], axis=1)

    # Bin widths for flux formula (uniform bins assumed)
    dx1_width = dh
    dx2_width = dL

    # Helper: resample an event array with block bootstrap and rebuild count matrix
    def resample_counts(events, shape, block_size):
        n = len(events)
        if n == 0:
            return np.zeros(shape)
        n_blocks = n // block_size
        block_idx = np.random.choice(n_blocks, size=n_blocks, replace=True)
        # Expand block indices to individual event indices
        event_idx = (block_idx[:, None] * block_size +
                     np.arange(block_size)[None, :]).ravel()
        event_idx = event_idx[event_idx < n]   # trim any overhang
        sampled = events[event_idx]
        counts = np.zeros(shape)
        np.add.at(counts, (sampled[:, 0], sampled[:, 1]), 1)
        return counts

    shape = (n_x1_bins, n_x2_bins)

    total_events = sum(len(e) for e in [ev_ltc, ev_ctl, ev_btc, ev_ctb])
    print(f"\nTransition events extracted: {total_events:,} "
          f"({100*total_events/len(h_abs):.1f}% of trajectory steps)")
    print(f"Running bootstrap ({n_bootstrap} samples, block size={block_size})...")

    # ------------------------------------------------------------------ #
    # Step 3: bootstrap loop — resample events, rebuild flux, integrate.
    # ------------------------------------------------------------------ #
    circ_samples = np.empty(n_bootstrap)

    for i in tqdm(range(n_bootstrap)):
        N_ltc  = resample_counts(ev_ltc, shape, block_size)
        N_ctl  = resample_counts(ev_ctl, shape, block_size)
        N_btc  = resample_counts(ev_btc, shape, block_size)
        N_ctb  = resample_counts(ev_ctb, shape, block_size)

        # Symmetry relations to get the remaining four count matrices
        N_ctr = np.zeros(shape);  N_ctr[:-1, :] = N_ltc[1:, :]
        N_rtc = np.zeros(shape);  N_rtc[:-1, :] = N_ctl[1:, :]
        N_cta = np.zeros(shape);  N_cta[:, :-1] = N_btc[:, 1:]
        N_atc = np.zeros(shape);  N_atc[:, :-1] = N_ctb[:, 1:]

        # Flux (mirrors compute_probability_flux arithmetic exactly)
        w_lc = (N_ltc - N_ctl) / t_total
        w_cr = (N_ctr - N_rtc) / t_total
        w_bc = (N_btc - N_ctb) / t_total
        w_ca = (N_cta - N_atc) / t_total

        j_boot = np.stack(
            [(w_lc + w_cr) / (2 * dx1_width),
             (w_bc + w_ca) / (2 * dx2_width)],
            axis=-1
        )

        circ, _, _ = compute_circulation_on_rectangle_discrete(
            j_boot, hcenters, Lcenters,
            h_start, h_end, L_start, L_end
        )
        circ_samples[i] = circ

    circ_mean = np.mean(circ_samples)
    circ_std  = np.std(circ_samples, ddof=1)

    return circ_mean, circ_std, circ_samples


def plot_bootstrap_distribution(circ_samples, circ_mean, circ_std,
                                tau_p, epsilon, bin_num,
                                save=True):
    """
    Plot and optionally save the bootstrap distribution of the circulation.

    Parameters:
    -----------
    circ_samples : np.ndarray
        Bootstrap circulation values
    circ_mean, circ_std : float
        Summary statistics
    tau_p, epsilon : float
        Simulation parameters (used in title and filename)
    bin_num : int
        Number of bins used in flux calculation (used in filename)
    save : bool
        Whether to save the figure to disk
    """
    z_score = abs(circ_mean) / circ_std if circ_std > 0 else np.inf

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(circ_samples, bins=30, density=True, alpha=0.7, edgecolor='black')
    ax.axvline(circ_mean, color='red', linestyle='--', linewidth=2,
               label=r'Mean $= {:+.3e}$'.format(circ_mean))
    ax.axvline(0, color='black', linestyle='-', linewidth=1, label='Zero')
    ax.axvline(circ_mean - 1.96*circ_std, color='red', linestyle=':', linewidth=1.5,
               label=r'$95\%$ CI')
    ax.axvline(circ_mean + 1.96*circ_std, color='red', linestyle=':', linewidth=1.5)

    ax.set_xlabel(r'Circulation $\Omega$')
    ax.set_ylabel(r'Probability density')
    ax.set_title((r'Bootstrap distribution $(n={:d})$, '
                  r'$\tau_p={:.3f}$, $\epsilon={:.3f}$'
                  ).format(len(circ_samples), tau_p, epsilon))
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        fname = (f'bootstrap_circulation_bin_num={bin_num}'
                 f'_taup={tau_p}_epsilon={epsilon}.svg')
        plt.savefig(fname)
        print(f"Bootstrap plot saved to {fname}")

    return fig, ax


# =============================================================================
# Example usage
# =============================================================================
if __name__ == "__main__":
    ## Enter simulation parameters for import
    N = 3                    # number of disks
    r = 1/(2**(1/6))         # radius of disks
    reff = r * (2**(1/6))    # effective radius of disks, from WCA potential
    epsilon = 0.25
    T = 0                    # athermal simulations
    f = 1
    tau_p = 10            # persistence time [0.005, 0.890899, 10]
    R = N + epsilon          # enclosure radius R = N + epsilon
    Tmax = int(1e5)
    nsamp = int(1e9)
    dt = Tmax / nsamp
    num_import_steps = nsamp

    # Unit definitions for plotting data in reduced units
    friction_zeta = 1        # friction coefficient
    r_eff = 1                # length scale
    int_strength = 40        # energy scale from WCA potential
    length_unit = r_eff
    energy_unit = int_strength
    time_unit = friction_zeta * (r_eff**2) / int_strength

    # Filename for trajectories data from C++ code
    dpath = "Z:/Jared/Data/weeks_model_desktop/athermal_trajectories_fish_school_params_18_06_2025/"
    filename = dpath + "h_A_only_T={}_epsilon={}_f={}_taup={}_trajectories.bin".format(
        T, epsilon, f, tau_p)

    ## Number of bins -- must be consistent throughout
    bin_num = 30

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    print(f"Reading trajectory from {filename}...")
    h_values, A_values = read_h_A_trajectory(filename)
    print(f"Loaded {len(h_values):,} time steps")

    L_values = 2*np.array(A_values) / np.abs(np.array(h_values))
    print(f"L range: [{L_values.min():.4f}, {L_values.max():.4f}]")

    try:
        metadata = read_metadata(filename)
        print("\nSimulation parameters:")
        print(metadata.to_string())
    except FileNotFoundError:
        print("\nMetadata file not found")
        metadata = None

    # -------------------------------------------------------------------------
    # Probability flux
    # -------------------------------------------------------------------------
    t_total = dt * num_import_steps / time_unit
    h_abs = np.abs(h_values)

    prob_flux_vector_field = compute_probability_flux(h_abs, L_values, t_total, bins=bin_num)

    hist, hedges, Ledges = np.histogram2d(h_abs, L_values, bin_num)
    p_alpha = hist / np.sum(hist)

    hcenters = (hedges[:-1] + hedges[1:]) / 2
    Lcenters = (Ledges[:-1] + Ledges[1:]) / 2
    X, Y = np.meshgrid(hcenters, Lcenters)

    # -------------------------------------------------------------------------
    # Circulation on rectangular contour
    # -------------------------------------------------------------------------
    h_start = 1.9
    h_end   = hcenters[-1]
    L_start = Lcenters[0]
    L_end   = 2.7

    circulation, contributions, actual_bounds = compute_circulation_on_rectangle_discrete(
        prob_flux_vector_field, hcenters, Lcenters,
        h_start, h_end, L_start, L_end
    )
    h_start_actual, h_end_actual, L_start_actual, L_end_actual = actual_bounds

    # -------------------------------------------------------------------------
    # Bootstrap uncertainty on circulation
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("BOOTSTRAP CIRCULATION ANALYSIS")
    print("="*60)

    circ_mean, circ_std, circ_samples = compute_circulation_bootstrap(
        h_abs, L_values, t_total,
        hcenters, Lcenters,
        h_start, h_end, L_start, L_end,
        n_bootstrap=1000,
        block_size=2
    )

    print(f"\nBootstrap results:")
    print(f"  Mean circulation : {circ_mean:+.6e}")
    print(f"  Std deviation    : {circ_std:.6e}")
    print(f"  95% CI           : [{circ_mean - 1.96*circ_std:+.6e}, "
          f"{circ_mean + 1.96*circ_std:+.6e}]")
    z_score = abs(circ_mean) / circ_std if circ_std > 0 else np.inf
    print(f"  Significance     : {z_score:.2f} sigma")

    if z_score > 2.576:
        print("  *** HIGHLY SIGNIFICANT (p < 0.01) ***")
    elif z_score > 1.96:
        print("  *** SIGNIFICANT (p < 0.05) ***")
    else:
        print("  Not significant at 95% level")
    print("="*60)

    plot_bootstrap_distribution(circ_samples, circ_mean, circ_std,
                                tau_p, epsilon, bin_num, save=True)

    # -------------------------------------------------------------------------
    # Flux magnitude + streamlines visualisation
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0.2, 0.2, 0.5, 0.5])

    H_field, L_field = np.meshgrid(hcenters, Lcenters, indexing='ij')
    h_components = prob_flux_vector_field[:, :, 0].T
    L_components = prob_flux_vector_field[:, :, 1].T
    flux_magnitude = np.sqrt(h_components**2 + L_components**2)

    # Set these manually after reading from first run
    vmin = 1.5e-5
    vmax = 5.55e-4
    levels = [0.0, 3.0000000000000004e-05, 6.000000000000001e-05, 9.000000000000002e-05, 0.00012000000000000002, 0.00015000000000000001, 0.00018000000000000004, 
              0.00021000000000000004, 0.00024000000000000003, 0.00027000000000000006, 0.00030000000000000003, 0.00033000000000000005, 0.0003600000000000001, 
              0.00039000000000000005, 0.00042000000000000007, 0.00045000000000000004, 0.00048000000000000007, 0.00051, 0.0005400000000000001, 0.0005700000000000001]

    im = ax.contourf(H_field.T, L_field.T, flux_magnitude, levels=levels,
                    cmap='coolwarm', alpha=1, vmin=vmin, vmax=vmax)

    # im = ax.contourf(H_field.T, L_field.T, flux_magnitude, levels=20,
    #                  cmap='coolwarm', alpha=1)

    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((hcenters, Lcenters), p_alpha,
                                     bounds_error=False, fill_value=0)
    H_stream, L_stream = np.meshgrid(hcenters, Lcenters, indexing='xy')
    p_at_field = interp(
        np.column_stack([H_stream.ravel(), L_stream.ravel()])
    ).reshape(H_stream.shape)

    total_samples = len(h_abs)
    min_samples_per_bin = 15
    threshold = max(min_samples_per_bin / total_samples, 0)
    mask = p_at_field > threshold

    print(f"\nStreamline statistics:")
    print(f"  Threshold       : {threshold:.6e}")
    print(f"  Phase space shown: {np.sum(mask)/mask.size:.1%}")
    print(f"  Probability mass : {np.sum(p_alpha[p_alpha > threshold])/np.sum(p_alpha):.1%}")
    print(f"  Mean |J| (shown) : {np.mean(flux_magnitude[mask.T]):.6f}")

    h_masked = np.where(mask, h_components, np.nan)
    L_masked = np.where(mask, L_components, np.nan)

    ax.streamplot(H_stream, L_stream, h_masked, L_masked,
                  color='white', linewidth=1.2, density=1.2,
                  arrowsize=1.2, arrowstyle='->')

    rect_h = [h_start_actual, h_start_actual, h_end_actual, h_end_actual, h_start_actual]
    rect_L = [L_start_actual, L_end_actual, L_end_actual, L_start_actual, L_start_actual]
    ax.plot(rect_h, rect_L, 'k-', linewidth=3, label='Integration contour')
    ax.plot(rect_h[0], rect_L[0], 'go', markersize=12, label='Start/End')

    arrow_props = dict(arrowstyle='->', lw=2, color='green')
    ax.annotate('', xy=(h_start_actual, (L_start_actual + L_end_actual)/2 + 0.1),
                xytext=(h_start_actual, (L_start_actual + L_end_actual)/2 - 0.1),
                arrowprops=arrow_props)
    ax.annotate('', xy=((h_start_actual + h_end_actual)/2 + 0.1, L_end_actual),
                xytext=((h_start_actual + h_end_actual)/2 - 0.1, L_end_actual),
                arrowprops=arrow_props)

    ax.set_xlabel(r'$|h|$')
    ax.set_ylabel(r'$\mathcal{L}$')

    h_span = hedges[-1] - hedges[0]
    L_span = Ledges[-1] - Ledges[0]
    ax.set_xlim(hedges[0] - 0.02*h_span, hedges[-1] + 0.02*h_span)
    ax.set_ylim(Ledges[0] - 0.02*L_span, Ledges[-1] + 0.02*L_span)

    # Show bootstrap result in legend title
    circ_label = (r'$\Omega = {:+.3e}$'.format(circ_mean) + '\n'
                  + r'$\pm {:.2e}$ ({:.1f}$\sigma$)'.format(circ_std, z_score))
    ax.legend(loc='upper right', title=circ_label, title_fontsize=11)

    cax = fig.add_axes([0.75, 0.2, 0.03, 0.5])
    cb = plt.colorbar(im, cax=cax, orientation='vertical')
    print(f"Contour level boundaries = {im.levels.tolist()}")
    print(f"Colorbar range: vmin={im.get_array().min():.6e}, vmax={im.get_array().max():.6e}")
    cax.set_ylabel(r'$|\vec{j}|$')

    plt.tight_layout()
    plt.savefig(
        rf'bootstrapped_reduced_units_flux_magnitude_streamlines_bin_num={bin_num}'
        rf'_taup={tau_p}_epsilon={epsilon}.svg'
    )
    plt.show()
