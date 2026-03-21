# Dark Matter Profile Differentiation in SPARC Galaxies

**Author:** Bhasker Vasudevan  
**Date:** March 2026  
**Status:** Research complete — paper in progress

---

## Overview

This project investigates the core-cusp problem in dark matter astrophysics 
using rotation curve data from 175 late-type galaxies in the 
[SPARC database](http://astroweb.cwru.edu/SPARC/). We fit both the 
NFW (cuspy) and pseudo-isothermal (cored) dark matter density profiles 
to each galaxy, identify which physical properties best distinguish 
core from cusp galaxies, and examine how baryonic matter content 
relates to dark matter core size.

---

## Key Findings

- The **isothermal (cored) profile** is preferred in 91 of 175 galaxies (52%), 
  while NFW is preferred in only 15 (8.6%)
- **Inner V_DM slope** and **ISO core radius r_c** are the only statistically 
  significant differentiators between ISO and NFW galaxies 
  (Mann-Whitney p = 0.003 and p = 0.028 respectively)
- Among ISO galaxies, **baryonic fraction correlates strongly with core radius**: 
  43% of ISO galaxies show both high baryonic fraction and large core radius, 
  while only 7% show the opposite — consistent with the baryonic feedback hypothesis

---
## Interactive Dashboard
🔗 [View the live results dashboard](https://bhaskerball52.github.io/dark-matter-project/galaxy_report.html)

The analysis generates a self-contained `galaxy_report.html` file that can 
be opened in any browser with no internet connection required. It contains 
three tabs:

**GALAXIES** — A searchable sidebar table listing all 175 galaxies with their 
fit parameters, chi-squared values, baryonic fraction, inner slope, and 
classification. Click any galaxy row to see its rotation curve and log-log 
density profile side by side in the main panel, along with all fitted 
parameters for both NFW and ISO models.

**MANN-WHITNEY ANALYSIS** — A ranked horizontal bar chart showing the 
correlation strength of each variable with ISO vs NFW classification. 
Variables are sorted from most to least correlated, with statistical 
significance indicated. Inner slope and core radius are highlighted as 
the key differentiating variables.

**BARYONIC MATTER ANALYSIS** — Three bar charts examining how baryonic 
fraction relates to inner slope and core radius separately for ISO and 
NFW galaxy groups, revealing the diagonal correlation between baryonic 
content and core size among ISO-preferred galaxies.

All plots are embedded directly in the HTML file so nothing needs to be 
installed or hosted; just open in web browser.

## Repository Structure
```
dark-matter-project/
├── data/
│   └── Rotmod_LTG/          # SPARC rotation curve files (175 .dat files)
├── dark_matter_density.py   # Core pipeline: NFW/ISO fitting, DM extraction
├── fit_galaxies.py          # Batch runner: fits all 175 galaxies, generates HTML report
├── baryonic_analysis.py     # Baryonic fraction vs core radius analysis + plot
├── galaxy_fit_results.csv   # Output: fit parameters and classifications per galaxy
├── galaxy_report.html       # Interactive HTML dashboard (open in browser)
└── rotation_curves.py       # Early exploratory rotation curve script
```

---

## How to Run

**Requirements:**
```bash
pip install numpy pandas matplotlib scipy
```

**Run the full batch analysis (all 175 galaxies):**
```bash
python3 fit_galaxies.py
```
This generates `galaxy_report.html` and `galaxy_fit_results.csv`.  
Open `galaxy_report.html` in any browser to explore results interactively.

**Run the baryonic analysis plot:**
```bash
python3 baryonic_analysis.py
```

---

## Data Source

Lelli, F., McGaugh, S.S., & Schombert, J.M. (2016).  
*SPARC: Mass Models for 175 Disk Galaxies with Spitzer Photometry and Accurate Rotation Curves.*  
The Astronomical Journal, 152, 157.  
[http://astroweb.cwru.edu/SPARC/](http://astroweb.cwru.edu/SPARC/)

---

*This project was conducted independently*
