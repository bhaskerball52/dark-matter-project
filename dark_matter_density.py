# dm_pipeline.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# Gravitational constant in kpc * (km/s)^2 / Msun
G = 4.30091e-6

# ---------------------------
# Profile definitions
# ---------------------------

def M_nfw(r, rho0, rs):
    """NFW cumulative mass inside r [kpc]. rho0 in Msun/kpc^3, rs in kpc."""
    x = np.array(r) / rs
    return 4*np.pi*rho0*rs**3 * (np.log(1 + x) - x/(1 + x))

def V_nfw(r, rho0, rs):
    """NFW circular velocity at r [km/s]"""
    r = np.array(r)
    # avoid divide-by-zero at r=0
    rr = np.clip(r, 1e-6, None)
    return np.sqrt(G * M_nfw(rr, rho0, rs) / rr)

def rho_nfw(r, rho0, rs):
    x = np.array(r)/rs
    return rho0 / (x * (1 + x)**2)

# Isothermal (cored) profile
def rho_iso(r, rho0, rc):
    r = np.array(r)
    return rho0 / (1 + (r/rc)**2)

def M_iso(r, rho0, rc):
    # integral: 4*pi * int_0^r rho(r') r'^2 dr'
    # no simple elementary closed form for core density simple integrand, compute numerically per radius
    r = np.array(r)
    M = np.zeros_like(r, dtype=float)
    for i, rr in enumerate(r):
        rsamp = np.linspace(0, rr, 300)
        integrand = rho_iso(rsamp, rho0, rc) * rsamp**2
        M[i] = 4*np.pi * np.trapz(integrand, rsamp)
    return M

def V_iso(r, rho0, rc):
    rr = np.array(r)
    rr_safe = np.clip(rr, 1e-6, None)
    return np.sqrt(G * M_iso(rr_safe, rho0, rc) / rr_safe)

# ---------------------------
# Fit wrappers
# ---------------------------

def fit_profile(V_target, r, profile='nfw', p0=None, bounds=None):
    """
    Fit either 'nfw' or 'iso' profile to V_target(r).
    Returns popt, pcov.
    """
    mask = np.isfinite(V_target) & (V_target > 0) & (r > 0)
    r_fit = r[mask]
    V_fit = V_target[mask]
    if len(r_fit) < 4:
        raise ValueError("Not enough valid points to fit.")

    if profile == 'nfw':
        if p0 is None:
            # p0: rho0 [Msun/kpc^3], rs [kpc]
            p0 = [1e7, 5.0]
        if bounds is None:
            bounds = ([1e4, 0.01], [1e12, 100.0])
        popt, pcov = curve_fit(lambda rr, rho0, rs: V_nfw(rr, rho0, rs),
                               r_fit, V_fit, p0=p0, bounds=bounds, maxfev=20000)
    elif profile == 'iso':
        if p0 is None:
            p0 = [1e7, 2.0]
        if bounds is None:
            bounds = ([1e4, 0.01], [1e12, 100.0])
        popt, pcov = curve_fit(lambda rr, rho0, rc: V_iso(rr, rho0, rc),
                               r_fit, V_fit, p0=p0, bounds=bounds, maxfev=20000)
    else:
        raise ValueError("profile must be 'nfw' or 'iso'")
    return popt, pcov

# ---------------------------
# Helpers: safe DM extraction
# ---------------------------

def extract_Vdm(Vobs, Vbar, errV=None):
    """
    Compute V_DM = sqrt(Vobs^2 - Vbar^2) with safety:
    - If Vobs^2 - Vbar^2 <= 0, set to NaN (can't take real sqrt)
    - propagate fractional errors if errV provided
    """
    v2 = Vobs**2 - Vbar**2
    Vdm = np.where(v2 > 0, np.sqrt(v2), np.nan)
    if errV is None:
        errDm = np.full_like(Vdm, np.nan)
    else:
        # crude propagation: dVdm/dVobs = Vobs / Vdm
        with np.errstate(invalid='ignore', divide='ignore'):
            deriv = np.where(Vdm>0, Vobs / Vdm, 0.0)
            errDm = deriv * errV
    return Vdm, errDm

# ---------------------------
# Plotting utilities
# ---------------------------

def plot_rotation_curve(r, Vobs, errV, Vbar, Vdm, fits=None, title=None):
    """
    fits: list of tuples (label, r_model_array, V_model_array)
    """
    plt.figure(figsize=(9,6))
    plt.errorbar(r, Vobs, yerr=errV, fmt='o', label='V_obs', color='tab:blue', capsize=3)
    plt.plot(r, Vbar, 'r--', label='V_bar (visible)')
    # plot Vdm points
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
    plt.ylabel(r"$\rho(r)\ \ [M_\odot\ 1/kpc^{-3}]$")
    if title: plt.title(title)
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()

# ---------------------------
# Example pipeline for one galaxy file (your first script)
# ---------------------------

def analyze_single_galaxy(galaxy_file, save_plots=False):
    data = pd.read_csv(galaxy_file, delim_whitespace=True, comment='#',
                       names=['Rad', 'Vobs', 'errV', 'Vgas', 'Vdisk', 'Vbul', 'SBdisk', 'SBbul'])
    r = data['Rad'].values
    Vobs = data['Vobs'].values
    errV = data['errV'].values
    Vbar = np.sqrt(np.nan_to_num(data['Vgas'])**2 + np.nan_to_num(data['Vdisk'])**2 + np.nan_to_num(data['Vbul'])**2)

    # DM extraction
    Vdm, errDm = extract_Vdm(Vobs, Vbar, errV)

    # Fit NFW and ISO (use only r>0 & finite Vdm)
    try:
        popt_nfw, pcov_nfw = fit_profile(Vdm, r, profile='nfw')
        popt_iso, pcov_iso = fit_profile(Vdm, r, profile='iso')
    except Exception as e:
        print("Fitting failed:", e)
        popt_nfw = popt_iso = None
        pcov_nfw = pcov_iso = None

    # Prepare model curves for plotting
    r_plot = np.linspace(max(r.min(), 0.01), r.max(), 300)
    fits = []
    rho_models = []
    labels = []
    if popt_nfw is not None:
        rho0_nfw, rs_nfw = popt_nfw
        V_nfw_mod = V_nfw(r_plot, rho0_nfw, rs_nfw)
        fits.append((f"NFW (rho0={rho0_nfw:.2e}, rs={rs_nfw:.2f} kpc)", r_plot, V_nfw_mod))
        rho_models.append(rho_nfw(r_plot, rho0_nfw, rs_nfw))
        labels.append('NFW')
    if popt_iso is not None:
        rho0_iso, rc_iso = popt_iso
        V_iso_mod = V_iso(r_plot, rho0_iso, rc_iso)
        fits.append((f"ISO (rho0={rho0_iso:.2e}, rc={rc_iso:.2f} kpc)", r_plot, V_iso_mod))
        rho_models.append(rho_iso(r_plot, rho0_iso, rc_iso))
        labels.append('Isothermal')

    # Plots
    plot_rotation_curve(r, Vobs, errV, Vbar, Vdm, fits, title=os.path.basename(galaxy_file))
    if save_plots:
        plt.savefig(os.path.basename(galaxy_file) + "_rotcurve.png", dpi=200)
    plt.show()

    if rho_models:
        plot_density(r_plot, rho_models, labels, title="Fitted DM density profiles")
        if save_plots:
            plt.savefig(os.path.basename(galaxy_file) + "_rho.png", dpi=200)
        plt.show()

    # return results
    results = {
        'radii': r,
        'Vobs': Vobs,
        'Vbar': Vbar,
        'Vdm': Vdm,
        'popt_nfw': popt_nfw,
        'pcov_nfw': pcov_nfw,
        'popt_iso': popt_iso,
        'pcov_iso': pcov_iso
    }
    return results

# ---------------------------
# Example: run on your file
# ---------------------------

if __name__ == "__main__":
    # change this to your path
    galaxy_file = 'data/Rotmod_LTG/D631-7_rotmod.dat'
    results = analyze_single_galaxy(galaxy_file, save_plots=False)

    # print summary of fits
    if results['popt_nfw'] is not None:
        rho0_nfw, rs_nfw = results['popt_nfw']
        print("NFW fit: rho0 = {:.3e} Msun/kpc^3, rs = {:.3f} kpc".format(rho0_nfw, rs_nfw))
    if results['popt_iso'] is not None:
        rho0_iso, rc_iso = results['popt_iso']
        print("ISO fit: rho0 = {:.3e} Msun/kpc^3, rc = {:.3f} kpc".format(rho0_iso, rc_iso))
