"""
Merge multi-radius experiment results and fit scaling laws.

Loads per-radius .npz files and combines into a single summary with
power-law fits for alpha(k) and T_c(k).

Usage:
  python benchmarking/merge_multiradius.py
  python benchmarking/merge_multiradius.py --data-dir outputs/data
"""

import argparse
import glob
import numpy as np
from scipy.optimize import curve_fit
import os


def load_results(data_dir):
    """Load all multiradius_R*.npz files from data_dir."""
    records = []
    pattern = os.path.join(data_dir, 'multiradius_R*.npz')
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No multiradius_R*.npz files found in {data_dir}")

    for fpath in files:
        d = np.load(fpath)
        records.append({
            'radius': int(d['radius']),
            'k': int(d['k']),
            'T_c': float(d['T_c']),
            'alpha': float(d['alpha']),
            'F_k_size': int(d['F_k_size']),
            'variances': np.array(d['variances']),
            'L_fss': np.array(d['L_fss']),
        })

    records.sort(key=lambda x: x['radius'])
    return records


def fit_alpha_k(records):
    """Fit alpha(k) = -2 + c * k^delta."""
    k_arr = np.array([r['k'] for r in records], dtype=float)
    alpha_arr = np.array([r['alpha'] for r in records], dtype=float)

    def model(k, c, delta):
        return -2.0 + c * k**delta

    try:
        popt, pcov = curve_fit(model, k_arr, alpha_arr,
                               p0=[0.1, 0.5], maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
        return popt, perr
    except Exception as e:
        print(f"  WARNING: alpha(k) fit failed: {e}")
        return None, None


def fit_Tc_k(records):
    """Fit T_c(k) = 0.5 - c2 * k^(-beta_k)."""
    k_arr = np.array([r['k'] for r in records], dtype=float)
    Tc_arr = np.array([r['T_c'] for r in records], dtype=float)

    def model(k, c2, beta_k):
        return 0.5 - c2 * k**(-beta_k)

    try:
        popt, pcov = curve_fit(model, k_arr, Tc_arr,
                               p0=[1.0, 0.5], maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
        return popt, perr
    except Exception as e:
        print(f"  WARNING: T_c(k) fit failed: {e}")
        return None, None


def main():
    parser = argparse.ArgumentParser(description="Merge multi-radius results")
    parser.add_argument('--data-dir', type=str, default='outputs/data',
                        help='Directory containing multiradius_R*.npz files')
    args = parser.parse_args()

    records = load_results(args.data_dir)

    print(f"\nLoaded {len(records)} radius results")
    print(f"{'radius':>6} {'k':>5} {'|F_k|':>6} {'T_c':>8} {'alpha':>8}")
    print(f"{'-'*6:>6} {'-'*5:>5} {'-'*6:>6} {'-'*8:>8} {'-'*8:>8}")
    for r in records:
        print(f"{r['radius']:>6d} {r['k']:>5d} {r['F_k_size']:>6d} "
              f"{r['T_c']:>8.4f} {r['alpha']:>8.4f}")

    # ── Fit alpha(k) = -2 + c * k^delta ────────────────────────
    print(f"\n{'='*50}")
    print("Fitting alpha(k) = -2 + c * k^delta")
    popt_a, perr_a = fit_alpha_k(records)
    if popt_a is not None:
        print(f"  c     = {popt_a[0]:.4f} +/- {perr_a[0]:.4f}")
        print(f"  delta = {popt_a[1]:.4f} +/- {perr_a[1]:.4f}")

    # ── Fit T_c(k) = 0.5 - c2 * k^(-beta_k) ───────────────────
    print(f"\nFitting T_c(k) = 0.5 - c2 * k^(-beta_k)")
    popt_t, perr_t = fit_Tc_k(records)
    if popt_t is not None:
        print(f"  c2     = {popt_t[0]:.4f} +/- {perr_t[0]:.4f}")
        print(f"  beta_k = {popt_t[1]:.4f} +/- {perr_t[1]:.4f}")

    # ── Save combined ───────────────────────────────────────────
    os.makedirs(args.data_dir, exist_ok=True)
    outpath = os.path.join(args.data_dir, 'multiradius_combined.npz')

    save_dict = {
        'radii': np.array([r['radius'] for r in records]),
        'k_values': np.array([r['k'] for r in records]),
        'T_c_values': np.array([r['T_c'] for r in records]),
        'alpha_values': np.array([r['alpha'] for r in records]),
        'F_k_sizes': np.array([r['F_k_size'] for r in records]),
    }
    if popt_a is not None:
        save_dict['alpha_fit_c'] = popt_a[0]
        save_dict['alpha_fit_delta'] = popt_a[1]
        save_dict['alpha_fit_c_err'] = perr_a[0]
        save_dict['alpha_fit_delta_err'] = perr_a[1]
    if popt_t is not None:
        save_dict['Tc_fit_c2'] = popt_t[0]
        save_dict['Tc_fit_beta_k'] = popt_t[1]
        save_dict['Tc_fit_c2_err'] = perr_t[0]
        save_dict['Tc_fit_beta_k_err'] = perr_t[1]

    np.savez(outpath, **save_dict)
    print(f"\nSaved {outpath}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
