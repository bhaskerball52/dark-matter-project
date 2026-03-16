import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import cumulative_trapezoid
import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

G = 4.30091e-6

def M_nfw(r, rho0, rs):
    x = np.array(r) / rs
    return 4*np.pi*rho0*rs**3 * (np.log(1 + x) - x/(1 + x))

def V_nfw(r, rho0, rs):
    r = np.array(r)
    rr = np.clip(r, 1e-6, None)
    return np.sqrt(G * M_nfw(rr, rho0, rs) / rr)

def rho_nfw(r, rho0, rs):
    x = np.array(r)/rs
    return rho0 / (x * (1 + x)**2)

def rho_iso(r, rho0, rc):
    r = np.array(r)
    return rho0 / (1 + (r/rc)**2)

def M_iso(r, rho0, rc):
    """
    Vectorized isothermal mass integral — evaluates all radii at once
    on a shared fine grid instead of looping per point.
    ~10x faster than the original per-point loop.
    """
    r = np.array(r)
    r_max = r.max()
    # single shared grid from 0 to r_max
    N = 150
    rsamp = np.linspace(0, r_max, N)
    integrand = rho_iso(rsamp, rho0, rc) * rsamp**2
    # cumulative integral at each grid point
    cum = np.zeros(N)
    cum[1:] = cumulative_trapezoid(integrand, rsamp)
    # interpolate to requested radii
    M = 4*np.pi * np.interp(r, rsamp, cum)
    return M

def V_iso(r, rho0, rc):
    rr = np.array(r)
    rr_safe = np.clip(rr, 1e-6, None)
    return np.sqrt(G * M_iso(rr_safe, rho0, rc) / rr_safe)

# ---------------------------
# Fit wrappers
# ---------------------------

def fit_profile(V_target, r, profile='nfw', p0=None, bounds=None, sigma=None):
    mask = np.isfinite(V_target) & (V_target > 0) & (r > 0)
    if sigma is not None:
        sigma = np.array(sigma)
        mask = mask & np.isfinite(sigma) & (sigma > 0)
    r_fit = r[mask]
    V_fit = V_target[mask]
    sigma_fit = sigma[mask] if sigma is not None else None
    if len(r_fit) < 4:
        raise ValueError("Not enough valid points to fit.")
    if profile == 'nfw':
        if p0 is None: p0 = [1e7, 5.0]
        if bounds is None: bounds = ([1e4, 0.01], [1e12, 100.0])
        popt, pcov = curve_fit(
            lambda rr, rho0, rs: V_nfw(rr, rho0, rs),
            r_fit, V_fit, p0=p0, bounds=bounds, maxfev=20000,
            sigma=sigma_fit, absolute_sigma=True
        )
    elif profile == 'iso':
        if p0 is None: p0 = [1e7, 2.0]
        if bounds is None: bounds = ([1e4, 0.01], [1e12, 100.0])
        popt, pcov = curve_fit(
            lambda rr, rho0, rc: V_iso(rr, rho0, rc),
            r_fit, V_fit, p0=p0, bounds=bounds, maxfev=20000,
            sigma=sigma_fit, absolute_sigma=True
        )
    else:
        raise ValueError("profile must be 'nfw' or 'iso'")
    return popt, pcov

# ---------------------------
# DM extraction
# ---------------------------

def extract_Vdm(Vobs, Vbar, errV=None):
    v2 = Vobs**2 - Vbar**2
    Vdm = np.where(v2 > 0, np.sqrt(v2), np.nan)
    if errV is None:
        errDm = np.full_like(Vdm, np.nan)
    else:
        with np.errstate(invalid='ignore', divide='ignore'):
            deriv = np.where(Vdm>0, Vobs / Vdm, 0.0)
            errDm = deriv * errV
    return Vdm, errDm

# ---------------------------
# Plotting utilities
# ---------------------------

def plot_rotation_curve(r, Vobs, errV, Vbar, Vdm, errDm=None, fits=None, title=None):
    plt.figure(figsize=(9,6))
    plt.errorbar(r, Vobs, yerr=errV, fmt='o', label='V_obs', color='tab:blue', capsize=3)
    plt.plot(r, Vbar, 'r--', label='V_bar (visible)')
    if errDm is not None:
        plt.errorbar(r, Vdm, yerr=errDm, fmt='o', label='V_DM (data)',
                     color='tab:orange', markersize=4, capsize=3)
    else:
        plt.plot(r, Vdm, 'o', label='V_DM (data)', color='tab:orange', markersize=4)
    if fits:
        for label, rr, Vmod in fits:
            plt.plot(rr, Vmod, '-', lw=2, label=label)
    plt.xlabel("Radius [kpc]")
    plt.ylabel("Velocity [km/s]")
    if title: plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

def plot_density(r_plot, rho_models, labels, title=None):
    plt.figure(figsize=(7,6))
    for rho, lab in zip(rho_models, labels):
        plt.loglog(r_plot, rho, label=lab)
    plt.xlabel("Radius [kpc]")
    plt.ylabel(r"$\rho(r)\ \ [M_\odot\ \mathrm{kpc}^{-3}]$")
    if title: plt.title(title)
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()

# ---------------------------
# Main pipeline
# ---------------------------

def analyze_single_galaxy(galaxy_file, save_plots=False):
    data = pd.read_csv(galaxy_file, sep=r'\s+', comment='#',
                       names=['Rad', 'Vobs', 'errV', 'Vgas', 'Vdisk', 'Vbul', 'SBdisk', 'SBbul'])
    r    = data['Rad'].values
    Vobs = data['Vobs'].values
    errV = data['errV'].values
    Vbar = np.sqrt(np.nan_to_num(data['Vgas'])**2 +
                   np.nan_to_num(data['Vdisk'])**2 +
                   np.nan_to_num(data['Vbul'])**2)

    Vdm, errDm = extract_Vdm(Vobs, Vbar, errV)

    try:
        popt_nfw, pcov_nfw = fit_profile(Vdm, r, profile='nfw', sigma=errDm)
        popt_iso, pcov_iso = fit_profile(Vdm, r, profile='iso', sigma=errDm)
    except Exception as e:
        print("Fitting failed:", e)
        popt_nfw = popt_iso = None
        pcov_nfw = pcov_iso = None

    r_plot = np.linspace(max(r.min(), 0.01), r.max(), 100)
    fits = []; rho_models = []; labels = []
    if popt_nfw is not None:
        rho0_nfw, rs_nfw = popt_nfw
        perr_nfw = np.sqrt(np.diag(pcov_nfw))
        fits.append((f"NFW (rho0={rho0_nfw:.2e}, rs={rs_nfw:.2f} kpc)", r_plot, V_nfw(r_plot, rho0_nfw, rs_nfw)))
        rho_models.append(rho_nfw(r_plot, rho0_nfw, rs_nfw))
        labels.append('NFW')
        print(f"NFW fit: rho0 = {rho0_nfw:.3e} ± {perr_nfw[0]:.2e},  rs = {rs_nfw:.3f} ± {perr_nfw[1]:.3f} kpc")
    if popt_iso is not None:
        rho0_iso, rc_iso = popt_iso
        perr_iso = np.sqrt(np.diag(pcov_iso))
        fits.append((f"ISO (rho0={rho0_iso:.2e}, rc={rc_iso:.2f} kpc)", r_plot, V_iso(r_plot, rho0_iso, rc_iso)))
        rho_models.append(rho_iso(r_plot, rho0_iso, rc_iso))
        labels.append('Isothermal')
        print(f"ISO fit: rho0 = {rho0_iso:.3e} ± {perr_iso[0]:.2e},  rc = {rc_iso:.3f} ± {perr_iso[1]:.3f} kpc")

    plot_rotation_curve(r, Vobs, errV, Vbar, Vdm, errDm=errDm, fits=fits,
                        title=os.path.basename(galaxy_file))
    if save_plots:
        plt.savefig(os.path.basename(galaxy_file) + "_rotcurve.png", dpi=150)
    plt.show()

    if rho_models:
        plot_density(r_plot, rho_models, labels, title="Fitted DM density profiles")
        if save_plots:
            plt.savefig(os.path.basename(galaxy_file) + "_rho.png", dpi=150)
        plt.show()

    return {
        'radii': r, 'Vobs': Vobs, 'Vbar': Vbar, 'errV': errV,
        'Vdm': Vdm, 'errDm': errDm,
        'popt_nfw': popt_nfw, 'pcov_nfw': pcov_nfw,
        'popt_iso': popt_iso, 'pcov_iso': pcov_iso
    }

if __name__ == "__main__":
    galaxy_file = 'data/Rotmod_LTG/D631-7_rotmod.dat'
    results = analyze_single_galaxy(galaxy_file, save_plots=False)