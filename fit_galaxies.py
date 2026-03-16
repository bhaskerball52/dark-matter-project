import sys
sys.path.insert(0, '/Users/bablu/Documents/GitHub/dark matter project')

import numpy as np
import time
import pandas as pd
import glob
import os
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from dark_matter_density import analyze_single_galaxy, V_nfw, V_iso, rho_nfw, rho_iso

# ---------------------------
# Chi-squared + classification
# ---------------------------

def compute_chi2(Vdm, Vmodel, errDm):
    mask = np.isfinite(Vdm) & np.isfinite(errDm) & (errDm > 0) & np.isfinite(Vmodel)
    n = mask.sum()
    if n < 3:
        return np.nan
    residuals = (Vdm[mask] - Vmodel[mask]) / errDm[mask]
    return np.sum(residuals**2) / (n - 2)

def classify(chi2_nfw, chi2_iso, threshold=0.33):
    if np.isnan(chi2_nfw) or np.isnan(chi2_iso):
        return 'failed'
    ratio = np.log(chi2_nfw / chi2_iso)
    if ratio < -threshold:
        return 'NFW'
    elif ratio > threshold:
        return 'ISO'
    else:
        return 'Neither'

# ---------------------------
# Metrics: CSB + V_DM inner slope
# ---------------------------

def extract_csb(galaxy_file):
    try:
        data = pd.read_csv(galaxy_file, sep=r'\s+', comment='#',
                           names=['Rad','Vobs','errV','Vgas','Vdisk','Vbul','SBdisk','SBbul'])
        sb = data['SBdisk'].values
        sb = sb[np.isfinite(sb) & (sb > 0)]
        return float(sb[0]) if len(sb) > 0 else np.nan
    except:
        return np.nan

def compute_vdm_inner_slope(r, Vdm):
    mask = np.isfinite(Vdm) & (Vdm > 0) & (r > 0)
    r_v = r[mask]; v_v = Vdm[mask]
    if len(r_v) < 2:
        return np.nan
    n = min(3, len(r_v))
    log_r = np.log(r_v[:n]); log_v = np.log(v_v[:n])
    slope = np.polyfit(log_r, log_v, 1)[0]
    return round(float(slope), 3)

# ---------------------------
# Mann-Whitney analysis
# ---------------------------

VARIABLES = ['n_points', 'r_max_kpc', 'Vobs_peak', 'fbar_mean',
             'chi2_iso', 'rho0_nfw', 'rs_nfw',
             'rho0_iso', 'rc_iso', 'csb', 'inner_slope']

def run_mannwhitney(rows):
    df = pd.DataFrame(rows)
    iso_df = df[df['winner'] == 'ISO']
    nfw_df = df[df['winner'] == 'NFW']
    results = []
    for var in VARIABLES:
        iso_vals = iso_df[var].dropna().values
        nfw_vals = nfw_df[var].dropna().values
        if len(iso_vals) < 3 or len(nfw_vals) < 3:
            continue
        try:
            stat, p = mannwhitneyu(iso_vals, nfw_vals, alternative='two-sided')
        except:
            continue
        corr_pct  = round((1 - p) * 100, 1)
        iso_med   = round(float(np.median(iso_vals)), 4)
        nfw_med   = round(float(np.median(nfw_vals)), 4)
        favors    = 'ISO' if iso_med > nfw_med else 'NFW'
        highlight = var in ('inner_slope', 'rc_iso')
        results.append({
            'variable': var, 'p_value': round(p, 5),
            'corr_pct': corr_pct, 'iso_median': iso_med,
            'nfw_median': nfw_med, 'favors': favors,
            'significant': p < 0.05, 'highlight': highlight
        })
    results.sort(key=lambda x: x['corr_pct'], reverse=True)
    return results

# ---------------------------
# Baryonic matter analysis for Tab 3
# ---------------------------

def baryonic_analysis(rows):
    """
    For ISO and NFW winners separately, categorise galaxies by
    whether they have high/low f_bar AND shallow/steep inner_slope.
    For ISO: also do high/low f_bar vs large/small rc_iso.
    Thresholds = medians of each group.
    Returns dicts of counts and thresholds.
    """
    df = pd.DataFrame(rows)
    iso = df[df['winner'] == 'ISO'].copy()
    nfw = df[df['winner'] == 'NFW'].copy()

    # thresholds = median of each group
    iso_fbar_thresh  = iso['fbar_mean'].median()
    iso_slope_thresh = iso['inner_slope'].median()
    iso_rc_thresh    = iso['rc_iso'].median()
    nfw_fbar_thresh  = nfw['fbar_mean'].median()
    nfw_slope_thresh = nfw['inner_slope'].median()

    def count_groups(df_in, fbar_t, slope_t):
        groups = {'High f_bar + Shallow slope': 0,
                  'High f_bar + Steep slope':   0,
                  'Low f_bar + Shallow slope':  0,
                  'Low f_bar + Steep slope':    0,
                  'Missing data': 0}
        for _, row in df_in.iterrows():
            fb = row['fbar_mean']; sl = row['inner_slope']
            if pd.isna(fb) or pd.isna(sl):
                groups['Missing data'] += 1; continue
            hi_fbar  = fb > fbar_t
            shallow  = sl < slope_t
            if hi_fbar and shallow:     groups['High f_bar + Shallow slope'] += 1
            elif hi_fbar and not shallow: groups['High f_bar + Steep slope'] += 1
            elif not hi_fbar and shallow: groups['Low f_bar + Shallow slope'] += 1
            else:                         groups['Low f_bar + Steep slope']   += 1
        return groups

    def count_rc_groups(df_in, fbar_t, rc_t):
        groups = {'High f_bar + Large core': 0,
                  'High f_bar + Small core':  0,
                  'Low f_bar + Large core':   0,
                  'Low f_bar + Small core':   0,
                  'Missing data': 0}
        for _, row in df_in.iterrows():
            fb = row['fbar_mean']; rc = row['rc_iso']
            if pd.isna(fb) or pd.isna(rc):
                groups['Missing data'] += 1; continue
            hi_fbar = fb > fbar_t
            large   = rc > rc_t
            if hi_fbar and large:         groups['High f_bar + Large core'] += 1
            elif hi_fbar and not large:   groups['High f_bar + Small core'] += 1
            elif not hi_fbar and large:   groups['Low f_bar + Large core']  += 1
            else:                         groups['Low f_bar + Small core']  += 1
        return groups

    iso_slope_counts = count_groups(iso, iso_fbar_thresh, iso_slope_thresh)
    nfw_slope_counts = count_groups(nfw, nfw_fbar_thresh, nfw_slope_thresh)
    iso_rc_counts    = count_rc_groups(iso, iso_fbar_thresh, iso_rc_thresh)

    return {
        'iso_slope': iso_slope_counts,
        'nfw_slope': nfw_slope_counts,
        'iso_rc':    iso_rc_counts,
        'iso_fbar_thresh':  round(iso_fbar_thresh, 3),
        'iso_slope_thresh': round(iso_slope_thresh, 3),
        'iso_rc_thresh':    round(iso_rc_thresh, 2),
        'nfw_fbar_thresh':  round(nfw_fbar_thresh, 3),
        'nfw_slope_thresh': round(nfw_slope_thresh, 3),
    }

# ---------------------------
# Plot helpers
# ---------------------------

def make_rotcurve_plot(r, Vobs, errV, Vbar, Vdm, errDm, fits, title):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(r, Vobs, yerr=errV, fmt='o', label='V_obs', color='#4C9BE8', capsize=3, ms=4)
    ax.plot(r, Vbar, 'r--', label='V_bar', lw=1.5)
    ax.errorbar(r, Vdm, yerr=errDm, fmt='o', label='V_DM', color='#F5A623', capsize=3, ms=4)
    for label, rr, Vm in fits:
        ax.plot(rr, Vm, '-', lw=2, label=label)
    ax.set_xlabel("Radius [kpc]"); ax.set_ylabel("Velocity [km/s]")
    ax.set_title(title, fontsize=10); ax.legend(fontsize=7); ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig_to_b64(fig)

def make_density_plot(r_plot, rho_models, labels, title):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    colors = ['#4C9BE8', '#F5A623']
    for (rho, lab), col in zip(zip(rho_models, labels), colors):
        ax.loglog(r_plot, rho, label=lab, color=col, lw=2)
    ax.set_xlabel("Radius [kpc]"); ax.set_ylabel(r"$\rho(r)\ [M_\odot\ \mathrm{kpc}^{-3}]$")
    ax.set_title(title, fontsize=10); ax.legend(fontsize=8); ax.grid(True, which='both', alpha=0.2)
    fig.tight_layout()
    return fig_to_b64(fig)

def fig_to_b64(fig):
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=90)
    plt.close(fig); buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# ---------------------------
# HTML builder
# ---------------------------

def build_html(rows, galaxy_plots, mw_results):
    winner_color = {'NFW': '#4C9BE8', 'ISO': '#F5A623', 'Neither': '#888', 'failed': '#e74c3c'}

    # --- Mann-Whitney rows (chi2_nfw removed, inner_slope + rc_iso highlighted) ---
    mw_rows_html = ""
    for mw in mw_results:
        pct       = mw['corr_pct']
        fav       = mw['favors']
        bar_color = '#4C9BE8' if fav == 'NFW' else '#F5A623'
        sig_badge = '<span style="color:#2ecc71;font-size:10px;margin-left:6px">● sig</span>' if mw['significant'] else '<span style="color:#555;font-size:10px;margin-left:6px">○ not sig</span>'
        if mw['highlight']:
            row_style = 'background:rgba(167,139,250,0.07);border-left:3px solid #a78bfa;'
            var_style = 'font-family:monospace;font-size:12px;color:#a78bfa;padding:10px 12px;font-weight:700;'
            badge = '<span style="font-size:9px;background:#a78bfa;color:#fff;padding:2px 6px;border-radius:4px;margin-left:6px;font-family:monospace">KEY</span>'
        else:
            row_style = ''
            var_style = 'font-family:monospace;font-size:12px;color:#6b7494;padding:8px 12px;'
            badge = ''
        mw_rows_html += f"""
        <tr style="{row_style}">
          <td style="{var_style}">{mw['variable']}{badge}</td>
          <td style="padding:8px 12px">
            <div style="display:flex;align-items:center;gap:8px">
              <div style="flex:1;background:#1e2435;border-radius:99px;height:8px;overflow:hidden">
                <div style="width:{min(pct,100)}%;background:{bar_color};height:100%;border-radius:99px"></div>
              </div>
              <span style="font-family:monospace;font-size:12px;min-width:42px">{pct}%</span>
              {sig_badge}
            </div>
          </td>
          <td style="padding:8px 12px;font-size:11px;color:#6b7494">ISO: {mw['iso_median']}</td>
          <td style="padding:8px 12px;font-size:11px;color:#6b7494">NFW: {mw['nfw_median']}</td>
          <td style="padding:8px 12px"><span style="font-size:11px;font-weight:600;color:{bar_color}">{fav} higher</span></td>
        </tr>"""

    # --- Tab 3: baryonic matter analysis ---
    bar_data = baryonic_analysis(rows)

    def make_bar_section(counts_dict, max_val, accent_key=None):
        html = ""
        colors = {
            'High f_bar + Shallow slope': '#2ecc71',
            'High f_bar + Steep slope':   '#e74c3c',
            'Low f_bar + Shallow slope':  '#F5A623',
            'Low f_bar + Steep slope':    '#555',
            'High f_bar + Large core':    '#2ecc71',
            'High f_bar + Small core':    '#e74c3c',
            'Low f_bar + Large core':     '#F5A623',
            'Low f_bar + Small core':     '#555',
            'Missing data':               '#333',
        }
        total = sum(v for k,v in counts_dict.items() if k != 'Missing data')
        for grp, val in counts_dict.items():
            if grp == 'Missing data' and val == 0: continue
            col = colors.get(grp, '#888')
            pct_bar = round(val / max_val * 100) if max_val > 0 else 0
            pct_of_total = round(val / total * 100) if total > 0 else 0
            border = 'border:2px solid #a78bfa;' if grp == accent_key else ''
            html += f"""
        <div style="display:flex;align-items:center;gap:14px;margin-bottom:14px">
          <div style="width:230px;font-size:11px;color:#e8eaf0;text-align:right;line-height:1.4">{grp}</div>
          <div style="flex:1;background:#1e2435;border-radius:6px;height:32px;overflow:hidden;{border}">
            <div style="width:{pct_bar}%;background:{col};height:100%;border-radius:6px;display:flex;align-items:center;padding-left:10px">
              <span style="font-family:monospace;font-size:12px;font-weight:700;color:#fff">{val}</span>
            </div>
          </div>
          <div style="width:40px;font-size:11px;color:#6b7494;font-family:monospace">{pct_of_total}%</div>
        </div>"""
        return html

    all_vals = [v for d in [bar_data['iso_slope'], bar_data['nfw_slope'], bar_data['iso_rc']]
                for k,v in d.items() if k != 'Missing data']
    max_val = max(all_vals) if all_vals else 1

    iso_slope_html = make_bar_section(bar_data['iso_slope'], max_val, 'High f_bar + Shallow slope')
    nfw_slope_html = make_bar_section(bar_data['nfw_slope'], max_val, 'High f_bar + Steep slope')
    iso_rc_html    = make_bar_section(bar_data['iso_rc'],    max_val, 'High f_bar + Large core')

    # --- Galaxy table rows ---
    table_rows = ""
    for r in rows:
        w = r['winner']; wc = winner_color.get(w, '#888')
        def fmt(v):
            return '—' if isinstance(v, float) and np.isnan(v) else v
        table_rows += f"""
        <tr onclick="showGalaxy('{r['galaxy']}')" class="table-row">
            <td class="gname">{r['galaxy']}</td>
            <td>{r['n_points']}</td>
            <td>{r['r_max_kpc']}</td>
            <td>{r['Vobs_peak']}</td>
            <td>{r['fbar_mean']:.3f}</td>
            <td>{fmt(r['chi2_nfw'])}</td>
            <td>{fmt(r['chi2_iso'])}</td>
            <td>{fmt(r['csb'])}</td>
            <td>{fmt(r['inner_slope'])}</td>
            <td><span class="badge" style="background:{wc}">{w}</span></td>
        </tr>"""

    # --- Galaxy detail panels ---
    galaxy_divs = ""
    for r in rows:
        gname   = r['galaxy']
        plots   = galaxy_plots.get(gname, {})
        rot_img = plots.get('rot', ''); den_img = plots.get('den', '')
        w = r['winner']; wc = winner_color.get(w, '#888')
        def fmtv(v, suffix=''):
            return '—' if isinstance(v, float) and np.isnan(v) else f"{v}{suffix}"
        nfw_params = f"ρ₀={r['rho0_nfw']:.2e} M☉/kpc³, rₛ={r['rs_nfw']:.2f} kpc" if not np.isnan(r['rho0_nfw']) else "fit failed"
        iso_params = f"ρ₀={r['rho0_iso']:.2e} M☉/kpc³, r_c={r['rc_iso']:.2f} kpc" if not np.isnan(r['rho0_iso']) else "fit failed"
        galaxy_divs += f"""
        <div class="galaxy-detail" id="detail-{gname}" style="display:none">
            <div class="detail-header">
                <h2>{gname}</h2>
                <span class="badge big-badge" style="background:{wc}">{w}</span>
            </div>
            <div class="stats-row">
                <div class="stat"><span class="stat-label">Data points</span><span class="stat-val">{r['n_points']}</span></div>
                <div class="stat"><span class="stat-label">Max radius</span><span class="stat-val">{r['r_max_kpc']} kpc</span></div>
                <div class="stat"><span class="stat-label">Peak V_obs</span><span class="stat-val">{r['Vobs_peak']} km/s</span></div>
                <div class="stat"><span class="stat-label">Mean f_bar</span><span class="stat-val">{r['fbar_mean']:.3f}</span></div>
                <div class="stat"><span class="stat-label">χ² NFW</span><span class="stat-val">{fmtv(r['chi2_nfw'])}</span></div>
                <div class="stat"><span class="stat-label">χ² ISO</span><span class="stat-val">{fmtv(r['chi2_iso'])}</span></div>
                <div class="stat"><span class="stat-label">CSB</span><span class="stat-val">{fmtv(r['csb'],' L/pc²')}</span></div>
                <div class="stat"><span class="stat-label">V_DM slope</span><span class="stat-val">{fmtv(r['inner_slope'])}</span></div>
            </div>
            <div class="params-row">
                <div class="param-box nfw-box"><strong>NFW</strong> &nbsp; {nfw_params}</div>
                <div class="param-box iso-box"><strong>ISO</strong> &nbsp; {iso_params}</div>
            </div>
            <div class="plots-row">
                <div class="plot-block"><div class="plot-label">Rotation Curve</div><img src="data:image/png;base64,{rot_img}" /></div>
                <div class="plot-block"><div class="plot-label">Density Profile (log-log)</div><img src="data:image/png;base64,{den_img}" /></div>
            </div>
        </div>"""

    winners = [r['winner'] for r in rows]
    n_nfw = winners.count('NFW'); n_iso = winners.count('ISO')
    n_nei = winners.count('Neither'); n_fail = winners.count('failed')
    total = len(rows)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Dark Matter Profile Analysis</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg:#0d0f14; --surface:#161a24; --surface2:#1e2435; --border:#2a3045;
    --text:#e8eaf0; --muted:#6b7494; --nfw:#4C9BE8; --iso:#F5A623;
    --neither:#888; --accent:#a78bfa;
  }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:var(--bg); color:var(--text); font-family:'DM Sans',sans-serif; min-height:100vh; }}
  .header {{ padding:28px 48px 18px; border-bottom:1px solid var(--border); }}
  .header h1 {{ font-family:'Space Mono',monospace; font-size:20px; letter-spacing:-0.5px; }}
  .header p {{ color:var(--muted); font-size:13px; margin-top:4px; }}
  .summary-bar {{ display:flex; gap:12px; padding:14px 48px; border-bottom:1px solid var(--border); flex-wrap:wrap; align-items:center; }}
  .pill {{ padding:5px 14px; border-radius:99px; font-size:11px; font-family:'Space Mono',monospace; font-weight:700; }}
  .pill.nfw {{ background:rgba(76,155,232,0.15); color:var(--nfw); border:1px solid rgba(76,155,232,0.3); }}
  .pill.iso {{ background:rgba(245,166,35,0.15); color:var(--iso); border:1px solid rgba(245,166,35,0.3); }}
  .pill.neither {{ background:rgba(136,136,136,0.12); color:var(--neither); border:1px solid rgba(136,136,136,0.25); }}
  .pill.failed-pill {{ background:rgba(231,76,60,0.12); color:#e74c3c; border:1px solid rgba(231,76,60,0.25); }}
  .pill.total {{ background:rgba(167,139,250,0.12); color:var(--accent); border:1px solid rgba(167,139,250,0.25); }}
  .note {{ font-size:11px; color:var(--muted); font-style:italic; }}
  .tabs {{ display:flex; gap:0; border-bottom:1px solid var(--border); padding:0 48px; }}
  .tab {{ padding:10px 22px; font-size:12px; font-family:'Space Mono',monospace; cursor:pointer;
          color:var(--muted); border-bottom:2px solid transparent; transition:all 0.15s; }}
  .tab.active {{ color:var(--accent); border-bottom:2px solid var(--accent); }}
  .tab-content {{ display:none; }}
  .tab-content.active {{ display:flex; height:calc(100vh - 178px); }}
  .sidebar {{ width:640px; min-width:640px; overflow-y:auto; border-right:1px solid var(--border); }}
  .main {{ flex:1; overflow-y:auto; padding:28px 36px; }}
  table {{ width:100%; border-collapse:collapse; font-size:12px; }}
  thead tr {{ background:var(--surface2); position:sticky; top:0; z-index:10; }}
  th {{ padding:9px 10px; text-align:left; font-family:'Space Mono',monospace; font-size:9px;
        color:var(--muted); text-transform:uppercase; letter-spacing:0.5px; border-bottom:1px solid var(--border); }}
  .table-row {{ cursor:pointer; border-bottom:1px solid rgba(42,48,69,0.6); transition:background 0.15s; }}
  .table-row:hover {{ background:var(--surface2); }}
  .table-row.active {{ background:rgba(167,139,250,0.08); border-left:2px solid var(--accent); }}
  td {{ padding:8px 10px; color:var(--text); }}
  .gname {{ font-family:'Space Mono',monospace; font-size:11px; color:var(--accent); }}
  .badge {{ display:inline-block; padding:2px 7px; border-radius:4px; font-size:10px;
            font-family:'Space Mono',monospace; font-weight:700; color:#fff; }}
  .big-badge {{ font-size:13px; padding:4px 14px; }}
  .galaxy-detail {{ animation:fadeIn 0.2s ease; }}
  @keyframes fadeIn {{ from {{ opacity:0; transform:translateY(6px); }} to {{ opacity:1; transform:translateY(0); }} }}
  .detail-header {{ display:flex; align-items:center; gap:16px; margin-bottom:20px; }}
  .detail-header h2 {{ font-family:'Space Mono',monospace; font-size:18px; }}
  .stats-row {{ display:flex; gap:10px; flex-wrap:wrap; margin-bottom:16px; }}
  .stat {{ background:var(--surface); border:1px solid var(--border); border-radius:8px;
           padding:10px 14px; display:flex; flex-direction:column; gap:3px; min-width:95px; }}
  .stat-label {{ font-size:9px; color:var(--muted); font-family:'Space Mono',monospace; text-transform:uppercase; }}
  .stat-val {{ font-size:14px; font-weight:600; }}
  .params-row {{ display:flex; gap:10px; margin-bottom:20px; flex-wrap:wrap; }}
  .param-box {{ flex:1; min-width:180px; padding:9px 14px; border-radius:8px; font-size:11px; font-family:'Space Mono',monospace; }}
  .nfw-box {{ background:rgba(76,155,232,0.08); border:1px solid rgba(76,155,232,0.25); color:var(--nfw); }}
  .iso-box {{ background:rgba(245,166,35,0.08); border:1px solid rgba(245,166,35,0.25); color:var(--iso); }}
  .plots-row {{ display:flex; gap:16px; flex-wrap:wrap; }}
  .plot-block {{ flex:1; min-width:260px; }}
  .plot-label {{ font-size:10px; color:var(--muted); font-family:'Space Mono',monospace; margin-bottom:6px; text-transform:uppercase; letter-spacing:0.5px; }}
  .plot-block img {{ width:100%; border-radius:8px; border:1px solid var(--border); }}
  .placeholder {{ display:flex; align-items:center; justify-content:center; height:300px;
                  color:var(--muted); font-family:'Space Mono',monospace; font-size:13px; text-align:center; line-height:2; }}
  .mw-panel {{ padding:32px 48px; overflow-y:auto; width:100%; }}
  .mw-panel h2 {{ font-family:'Space Mono',monospace; font-size:16px; margin-bottom:6px; }}
  .mw-panel p {{ color:var(--muted); font-size:13px; margin-bottom:24px; line-height:1.6; }}
  .mw-table {{ width:100%; border-collapse:collapse; }}
  .mw-table th {{ padding:8px 12px; text-align:left; font-family:'Space Mono',monospace; font-size:10px;
                  color:var(--muted); text-transform:uppercase; border-bottom:2px solid var(--border); }}
  .mw-table td {{ border-bottom:1px solid rgba(42,48,69,0.4); vertical-align:middle; }}
  .mw-table tr:hover td {{ background:rgba(30,36,53,0.5); }}
  .chart-panel {{ padding:36px 48px; overflow-y:auto; width:100%; }}
  .chart-panel h2 {{ font-family:'Space Mono',monospace; font-size:16px; margin-bottom:6px; }}
  .chart-panel .sub {{ color:var(--muted); font-size:13px; margin-bottom:8px; line-height:1.6; }}
  .thresholds {{ font-size:11px; color:#a78bfa; font-family:monospace; margin-bottom:32px;
                 background:rgba(167,139,250,0.07); border:1px solid rgba(167,139,250,0.2);
                 padding:10px 16px; border-radius:8px; display:inline-block; }}
  ::-webkit-scrollbar {{ width:5px; }}
  ::-webkit-scrollbar-track {{ background:transparent; }}
  ::-webkit-scrollbar-thumb {{ background:var(--border); border-radius:99px; }}
</style>
</head>
<body>

<div class="header">
  <h1>DARK MATTER PROFILE ANALYSIS</h1>
  <p>NFW (cusp) vs Isothermal (core) — SPARC rotation curve fits · {total} galaxies</p>
</div>

<div class="summary-bar">
  <span class="pill total">TOTAL {total}</span>
  <span class="pill nfw">NFW {n_nfw}</span>
  <span class="pill iso">ISO {n_iso}</span>
  <span class="pill neither">NEITHER {n_nei}</span>
  <span class="pill failed-pill">FAILED {n_fail}</span>
  <span class="note">† Failed = not enough valid V_DM data points (baryon-dominated)</span>
</div>

<div class="tabs">
  <div class="tab active" onclick="switchTab('galaxies', this)">GALAXIES</div>
  <div class="tab" onclick="switchTab('mannwhitney', this)">MANN-WHITNEY ANALYSIS</div>
  <div class="tab" onclick="switchTab('correlation', this)">BARYONIC MATTER ANALYSIS</div>
</div>

<!-- TAB 1: Galaxy browser -->
<div class="tab-content active" id="tab-galaxies">
  <div class="sidebar">
    <table>
      <thead>
        <tr>
          <th>Galaxy</th><th>N pts</th><th>r_max (kpc)</th><th>V_peak (km/s)</th>
          <th>f_bar (0–1)</th><th>χ² NFW</th><th>χ² ISO</th>
          <th>CSB (L/pc²)</th><th>V_DM slope</th><th>Winner</th>
        </tr>
      </thead>
      <tbody id="galaxy-table">{table_rows}</tbody>
    </table>
  </div>
  <div class="main" id="main-panel">
    <div class="placeholder" id="placeholder">← click a galaxy to see its profiles</div>
    {galaxy_divs}
  </div>
</div>

<!-- TAB 2: Mann-Whitney -->
<div class="tab-content" id="tab-mannwhitney">
  <div class="mw-panel">
    <h2>MANN-WHITNEY U TEST — Variable Correlation with ISO vs NFW</h2>
    <p>
      Each row shows how strongly that variable differs between ISO-winning and NFW-winning galaxies.<br>
      <strong>Correlation %</strong> = (1 − p-value) × 100. Higher % = variable changes more between groups.<br>
      <span style="color:#2ecc71">● sig</span> = statistically significant (p &lt; 0.05) &nbsp;·&nbsp;
      Bar color = which group has higher median &nbsp;·&nbsp;
      <span style="color:#a78bfa;font-weight:600">KEY</span> = most physically meaningful variables
    </p>
    <table class="mw-table">
      <thead>
        <tr>
          <th>Variable</th>
          <th style="min-width:300px">Correlation with ISO vs NFW</th>
          <th>ISO median</th><th>NFW median</th><th>Higher in</th>
        </tr>
      </thead>
      <tbody>{mw_rows_html}</tbody>
    </table>
  </div>
</div>

<!-- TAB 3: Baryonic Matter Analysis -->
<div class="tab-content" id="tab-correlation">
  <div class="chart-panel">
    <h2>BARYONIC MATTER vs DARK MATTER PROFILE INDICATORS</h2>
    <p class="sub">
      Does the amount of baryonic matter in a galaxy correlate with its dark matter profile shape?<br>
      <strong style="color:#2ecc71">Green</strong> = the combination you'd expect if baryonic feedback drives core formation &nbsp;·&nbsp;
      Thresholds = median of each group. Outlined bar = predicted key combination.
    </p>

    <div style="display:grid;grid-template-columns:1fr 1fr;gap:40px;max-width:1100px">

      <!-- ISO: f_bar vs inner slope -->
      <div>
        <div style="font-family:monospace;font-size:13px;color:#F5A623;margin-bottom:4px;font-weight:700">ISO GALAXIES — f_bar vs Inner Slope</div>
        <div class="thresholds" style="margin-bottom:16px">
          f_bar threshold: {bar_data['iso_fbar_thresh']} &nbsp;·&nbsp; slope threshold: {bar_data['iso_slope_thresh']}
        </div>
        {iso_slope_html}
      </div>

      <!-- NFW: f_bar vs inner slope -->
      <div>
        <div style="font-family:monospace;font-size:13px;color:#4C9BE8;margin-bottom:4px;font-weight:700">NFW GALAXIES — f_bar vs Inner Slope</div>
        <div class="thresholds" style="margin-bottom:16px">
          f_bar threshold: {bar_data['nfw_fbar_thresh']} &nbsp;·&nbsp; slope threshold: {bar_data['nfw_slope_thresh']}
        </div>
        {nfw_slope_html}
      </div>

      <!-- ISO: f_bar vs rc_iso -->
      <div>
        <div style="font-family:monospace;font-size:13px;color:#F5A623;margin-bottom:4px;font-weight:700">ISO GALAXIES — f_bar vs Core Radius (rc)</div>
        <div class="thresholds" style="margin-bottom:16px">
          f_bar threshold: {bar_data['iso_fbar_thresh']} &nbsp;·&nbsp; rc threshold: {bar_data['iso_rc_thresh']} kpc
        </div>
        {iso_rc_html}
      </div>

      <!-- legend -->
      <div style="padding-top:32px">
        <div style="font-family:monospace;font-size:11px;color:#6b7494;margin-bottom:12px;text-transform:uppercase">Legend</div>
        <div style="font-size:12px;line-height:2.2;color:#e8eaf0">
          <span style="color:#2ecc71">■</span> High f_bar + core indicator → baryonic feedback prediction<br>
          <span style="color:#e74c3c">■</span> High f_bar + cusp indicator → against feedback prediction<br>
          <span style="color:#F5A623">■</span> Low f_bar + core indicator → core without baryonic cause<br>
          <span style="color:#555">■</span> Low f_bar + cusp indicator → expected for DM-dominated galaxies<br>
          <span style="color:#a78bfa">□</span> Outlined bar = the combination most relevant to feedback hypothesis
        </div>
      </div>

    </div>
  </div>
</div>

<script>
  let active = null;
  function showGalaxy(name) {{
    if (active) {{
      document.getElementById('detail-' + active).style.display = 'none';
      document.querySelector('.table-row.active')?.classList.remove('active');
    }}
    document.getElementById('placeholder').style.display = 'none';
    document.getElementById('detail-' + name).style.display = 'block';
    document.querySelectorAll('.table-row').forEach(r => {{
      if (r.querySelector('.gname').textContent === name) r.classList.add('active');
    }});
    active = name;
  }}
  function switchTab(id, el) {{
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.getElementById('tab-' + id).classList.add('active');
    el.classList.add('active');
  }}
</script>
</body>
</html>"""
    return html

# ---------------------------
# Main batch runner
# ---------------------------

def run_batch(data_dir='data/Rotmod_LTG', output_html='galaxy_report.html', output_csv='galaxy_fit_results.csv'):
    files = sorted(glob.glob(os.path.join(data_dir, '*.dat')))
    print(f"Found {len(files)} galaxy files.\n")

    rows = []; galaxy_plots = {}
    t0 = time.time()

    for f in files:
        name = os.path.basename(f).replace('_rotmod.dat', '')
        print(f"  Fitting {name}...", end=' ', flush=True)
        try:
            res = analyze_single_galaxy(f, save_plots=False)
        except Exception as e:
            print(f"FAILED: {e}"); continue

        r     = res['radii'];  Vobs  = res['Vobs']
        Vbar  = res['Vbar'];  Vdm   = res['Vdm']
        errDm = res['errDm']; errV  = res.get('errV', np.full_like(Vobs, np.nan))

        with np.errstate(invalid='ignore'):
            fbar = np.nanmean(Vbar**2 / Vobs**2)

        csb         = extract_csb(f)
        inner_slope = compute_vdm_inner_slope(r, Vdm)

        popt_nfw = res['popt_nfw']; popt_iso = res['popt_iso']
        Vmod_nfw = V_nfw(r, *popt_nfw) if popt_nfw is not None else np.full_like(r, np.nan)
        Vmod_iso = V_iso(r, *popt_iso) if popt_iso is not None else np.full_like(r, np.nan)

        chi2_nfw = compute_chi2(Vdm, Vmod_nfw, errDm)
        chi2_iso = compute_chi2(Vdm, Vmod_iso, errDm)
        winner   = classify(chi2_nfw, chi2_iso)

        r_plot = np.linspace(max(r.min(), 0.01), r.max(), 100)
        fits = []; rho_models = []; rho_labels = []
        if popt_nfw is not None:
            fits.append(("NFW", r_plot, V_nfw(r_plot, *popt_nfw)))
            rho_models.append(rho_nfw(r_plot, *popt_nfw)); rho_labels.append('NFW')
        if popt_iso is not None:
            fits.append(("ISO", r_plot, V_iso(r_plot, *popt_iso)))
            rho_models.append(rho_iso(r_plot, *popt_iso)); rho_labels.append('Isothermal')

        rot_b64 = make_rotcurve_plot(r, Vobs, errV, Vbar, Vdm, errDm, fits, name)
        den_b64 = make_density_plot(r_plot, rho_models, rho_labels, f"{name} — Density")
        galaxy_plots[name] = {'rot': rot_b64, 'den': den_b64}

        row = {
            'galaxy': name, 'n_points': len(r),
            'r_max_kpc': round(r.max(), 2),
            'Vobs_peak': round(np.nanmax(Vobs), 1),
            'fbar_mean': round(fbar, 3),
            'rho0_nfw':  popt_nfw[0] if popt_nfw is not None else np.nan,
            'rs_nfw':    popt_nfw[1] if popt_nfw is not None else np.nan,
            'chi2_nfw':  round(chi2_nfw, 3) if not np.isnan(chi2_nfw) else np.nan,
            'rho0_iso':  popt_iso[0] if popt_iso is not None else np.nan,
            'rc_iso':    popt_iso[1] if popt_iso is not None else np.nan,
            'chi2_iso':  round(chi2_iso, 3) if not np.isnan(chi2_iso) else np.nan,
            'winner':    winner,
            'csb':       round(csb, 2) if not np.isnan(csb) else np.nan,
            'inner_slope': inner_slope if inner_slope is not None and not np.isnan(inner_slope) else np.nan,
        }
        rows.append(row)
        elapsed = time.time() - t0
        print(f"χ²_NFW={chi2_nfw:.2f}  χ²_ISO={chi2_iso:.2f}  → {winner}  ({elapsed:.1f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    mw_results = run_mannwhitney(rows)
    print("\nMann-Whitney results (ranked):")
    for m in mw_results:
        sig = "SIG" if m['significant'] else "   "
        print(f"  {sig} {m['variable']:15s}  corr={m['corr_pct']}%  p={m['p_value']}")

    html = build_html(rows, galaxy_plots, mw_results)
    with open(output_html, 'w') as fh:
        fh.write(html)

    print(f"\n✓ CSV  → {output_csv}")
    print(f"✓ HTML → {output_html}\n")
    print(df['winner'].value_counts().to_string())
    return df

if __name__ == "__main__":
    df = run_batch()