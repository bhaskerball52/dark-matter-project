import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Gravitational constant in convenient units
G = 4.302e-6  # kpc * (km/s)^2 / solar mass

# Read the file - skip all the header/metadata lines
# The actual data starts after the long header section
with open('data/MassModels_Lelli2016c.mrt.txt', 'r') as f:
    lines = f.readlines()

# Find where the actual data starts (after all the header lines)
data_start = 0
for i, line in enumerate(lines):
    if line.strip() and not line.startswith(('Title:', 'Authors:', 'Table:', '=', 'Byte', '-', 'Note', ' Bytes')):
        # Check if line has numeric data
        parts = line.split()
        if len(parts) >= 10:
            try:
                float(parts[1])  # Try to convert second column to float
                data_start = i
                break
            except:
                continue

print(f"Data starts at line {data_start}")

# Read the data starting from the correct line
data = pd.read_csv('data/MassModels_Lelli2016c.mrt.txt', 
                   sep=r'\s+',
                   skiprows=data_start,
                   names=['ID', 'D', 'R', 'Vobs', 'e_Vobs', 'Vgas', 'Vdisk', 'Vbul', 'SBdisk', 'SBbul'],
                   on_bad_lines='skip')

print(f"Loaded {len(data)} data points")
print("\nFirst few rows:")
print(data.head())

# Calculate baryonic (visible matter) velocity for each point
data['Vbar'] = np.sqrt(data['Vgas']**2 + data['Vdisk']**2 + data['Vbul']**2)

# Calculate masses
data['M_total'] = data['R'] * data['Vobs']**2 / G  # Total mass from observations
data['M_visible'] = data['R'] * data['Vbar']**2 / G  # Visible matter mass
data['M_DM'] = data['M_total'] - data['M_visible']  # Dark matter mass

# Calculate dark matter fraction
data['f_DM'] = data['M_DM'] / data['M_total'] * 100  # as percentage

# Create summary table: one row per galaxy with values at outermost radius
summary = data.groupby('ID').last().reset_index()

# Select relevant columns for summary
summary_table = summary[['ID', 'D', 'R', 'Vobs', 'Vbar', 'M_total', 'M_visible', 'M_DM', 'f_DM']]

# Rename columns for clarity
summary_table.columns = ['Galaxy', 'Distance_Mpc', 'Max_Radius_kpc', 
                         'Vobs_outer_km/s', 'Vbar_outer_km/s',
                         'M_total_Msun', 'M_visible_Msun', 'M_DM_Msun', 'DM_fraction_%']

print("\n" + "="*80)
print("SUMMARY TABLE: Dark Matter Properties at Outermost Radius")
print("="*80)
print(summary_table.head(20))

# Save to CSV
summary_table.to_csv('dark_matter_summary.csv', index=False)
print("\nSummary table saved to 'dark_matter_summary.csv'")

# Calculate statistics
print("\n" + "="*80)
print("OVERALL STATISTICS")
print("="*80)
print(f"Number of galaxies analyzed: {len(summary_table)}")
print(f"Average dark matter fraction: {summary_table['DM_fraction_%'].mean():.1f}%")
print(f"Median dark matter fraction: {summary_table['DM_fraction_%'].median():.1f}%")
print(f"Min dark matter fraction: {summary_table['DM_fraction_%'].min():.1f}%")
print(f"Max dark matter fraction: {summary_table['DM_fraction_%'].max():.1f}%")

# Plot histogram of dark matter fractions
plt.figure(figsize=(10, 6))
plt.hist(summary_table['DM_fraction_%'], bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Dark Matter Fraction (%)', fontsize=12)
plt.ylabel('Number of Galaxies', fontsize=12)
plt.title('Distribution of Dark Matter Fractions in SPARC Galaxies', fontsize=14)
plt.axvline(summary_table['DM_fraction_%'].mean(), color='red', 
            linestyle='--', linewidth=2, label=f'Mean: {summary_table["DM_fraction_%"].mean():.1f}%')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('dark_matter_histogram.png', dpi=300)
plt.show()

print("\nHistogram saved to 'dark_matter_histogram.png'")