import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Pick a galaxy file to start with
galaxy_file = 'data/Rotmod_LTG/D631-7_rotmod.dat'  # Change this to whatever file you want

# Read the data (skip comment lines starting with #)
data = pd.read_csv(galaxy_file, delim_whitespace=True, comment='#',
                   names=['Rad', 'Vobs', 'errV', 'Vgas', 'Vdisk', 'Vbul', 'SBdisk', 'SBbul'])

# Calculate velocity from visible matter (baryonic matter)
data['Vbar'] = np.sqrt(data['Vgas']**2 + data['Vdisk']**2 + data['Vbul']**2)

# Print first few rows to see what we have
print("First few data points:")
print(data[['Rad', 'Vobs', 'Vbar']].head())

# Create the plot
plt.figure(figsize=(10, 6))

# Plot observed velocities with error bars
plt.errorbar(data['Rad'], data['Vobs'], yerr=data['errV'], 
             fmt='o', color='blue', label='Observed velocity', 
             capsize=3, markersize=5)

# Plot predicted velocity from visible matter only
plt.plot(data['Rad'], data['Vbar'], 'r--', linewidth=2, 
         label='Visible matter only')

# Labels and formatting
plt.xlabel('Radius (kpc)', fontsize=12)
plt.ylabel('Rotation Velocity (km/s)', fontsize=12)
plt.title(f'Rotational velocity to radius from galactic center: {os.path.basename(galaxy_file)}', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Show the plot
plt.tight_layout()
plt.show()

# Calculate how much dark matter is needed
print("\nDark matter evidence:")
print(f"At largest radius ({data['Rad'].iloc[-1]:.2f} kpc):")
print(f"  Observed velocity: {data['Vobs'].iloc[-1]:.2f} km/s")
print(f"  Predicted from visible matter: {data['Vbar'].iloc[-1]:.2f} km/s")
print(f"  Discrepancy: {data['Vobs'].iloc[-1] - data['Vbar'].iloc[-1]:.2f} km/s")