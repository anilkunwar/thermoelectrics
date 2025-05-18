import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Parameters
A = -3.5
B = 2
D = 0.45
E = 1.5
F = -0.2
c_eq = 1.0

# Free energy function
def free_energy(c):
    return A * (B * c - c_eq)**2 + D * (B * c - c_eq) + E * (B * c - c_eq)**6 + F

# Streamlit app
def main():
    st.title("Free Energy Function Plot")

    # Concentration range between 0 and 1
    c_values = np.linspace(0, 1, 200)

    # Calculate free energy values, scaled to *10^9 J/m^3
    f_values = free_energy(c_values) * 1e9

    # Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(c_values, f_values, color='b', linewidth=4)

    # Labels and styling
    ax.set_xlabel('$c$', fontsize=25)
    ax.set_ylabel('$f_{bulk}  \ \ [J/m^3]$', fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=20, width=5.0, size=8)

    # Set y-axis limits to be negative or 0
    ax.set_ylim(min(f_values) * 1.1, 0)

    # Grid and spine styling
    ax.grid(True, linestyle='--', linewidth=1.0)
    ax.spines['top'].set_linewidth(4)
    ax.spines['right'].set_linewidth(4)
    ax.spines['bottom'].set_linewidth(4)
    ax.spines['left'].set_linewidth(4)

    # Display the plot in Streamlit
    st.pyplot(fig)

if __name__ == "__main__":
    main()

