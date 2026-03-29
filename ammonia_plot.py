"""
Ammonia Synthesis — All Plots
Chapter 1: Fe Catalyst, Temkin-Dyson-Simon Model

FIXES applied vs original:
  1. Output folder is now a local 'plots_output' folder beside this script
     (original used /mnt/user-data/outputs/ which is a Linux-only path)
  2. AutoMinorLocator removed from log-scale y-axis in plot_arrhenius()
     and dashboard panel A (matplotlib raises UserWarning and ignores it)
  3. style_ax() now accepts an optional log_y flag to skip minor locator
     on logarithmic axes
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
from ammonia_kinetics import (
    arrhenius, temkin_rate, simulate_reactor,
    rate_vs_temperature, equilibrium_conversion,
    R, k0, Ea, alpha, P_bar
)

# ─────────────────────────────────────────────
# OUTPUT FOLDER — saves plots beside this script
# ─────────────────────────────────────────────

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def out(filename):
    """Return full path for a plot file inside OUTPUT_DIR."""
    return os.path.join(OUTPUT_DIR, filename)


# ─────────────────────────────────────────────
# STYLE SETUP
# ─────────────────────────────────────────────

plt.rcParams.update({
    'figure.facecolor'  : '#0f1117',
    'axes.facecolor'    : '#161b25',
    'axes.edgecolor'    : '#2e3650',
    'axes.labelcolor'   : '#c8d0e0',
    'axes.titlecolor'   : '#e8edf5',
    'axes.grid'         : True,
    'grid.color'        : '#2e3650',
    'grid.linewidth'    : 0.6,
    'grid.alpha'        : 0.8,
    'xtick.color'       : '#7a8299',
    'ytick.color'       : '#7a8299',
    'text.color'        : '#c8d0e0',
    'font.family'       : 'DejaVu Sans',
    'font.size'         : 10,
    'axes.titlesize'    : 12,
    'axes.labelsize'    : 10,
    'legend.facecolor'  : '#1e2535',
    'legend.edgecolor'  : '#2e3650',
    'legend.fontsize'   : 9,
    'lines.linewidth'   : 2.0,
})

# Color palette
COLORS = ['#4fc3f7', '#81c784', '#ffb74d', '#e57373', '#ce93d8', '#80deea']
ACCENT = '#4fc3f7'

T_RANGE = np.linspace(548, 773, 300)   # K (experimental range from paper)


def style_ax(ax, title, xlabel, ylabel, log_y=False):
    """
    Apply consistent styling to an axes object.
    Set log_y=True when the y-axis is logarithmic — this skips
    AutoMinorLocator on y (which doesn't work on log scales).
    """
    ax.set_title(title, pad=10, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # x minor ticks are always safe
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    # y minor ticks only on linear axes
    if not log_y:
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='minor', length=3, color='#2e3650')


# ─────────────────────────────────────────────
# PLOT 1 — Arrhenius: k vs Temperature
# ─────────────────────────────────────────────

def plot_arrhenius():
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0f1117')

    k_vals = arrhenius(T_RANGE)

    ax.semilogy(T_RANGE, k_vals, color=ACCENT, linewidth=2.5)
    ax.fill_between(T_RANGE, k_vals * 0.01, k_vals,
                    alpha=0.12, color=ACCENT)

    # Mark key temperatures
    for T_mark, label in [(548, '548 K'), (673, '673 K'), (773, '773 K')]:
        k_mark = arrhenius(T_mark)
        ax.plot(T_mark, k_mark, 'o', color=ACCENT, markersize=8, zorder=5)
        ax.annotate(f'{label}\nk={k_mark:.2e}',
                    xy=(T_mark, k_mark),
                    xytext=(T_mark + 15, k_mark * 2),
                    fontsize=8, color='#c8d0e0',
                    arrowprops=dict(arrowstyle='->', color='#4a5568'))

    # log_y=True — skips AutoMinorLocator on the log y-axis
    style_ax(ax,
             f'Rate Constant vs Temperature (Arrhenius)\nk₀={k0:.2e}, Eₐ={Ea/1000:.1f} kJ/mol, α={alpha}',
             'Temperature (K)',
             'Rate Constant k (kmol/m³/h)',
             log_y=True)

    ax.set_xlim(540, 785)
    plt.tight_layout()
    plt.savefig(out('plot1_arrhenius.png'), dpi=150,
                bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print("  ✓ Plot 1 saved: Arrhenius")


# ─────────────────────────────────────────────
# PLOT 2 — Reaction Rate vs Temperature
# ─────────────────────────────────────────────

def plot_rate_vs_temp():
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0f1117')

    compositions = [
        (0.25, 0.74, 0.01, 'NH₃=1%  (fresh feed)'),
        (0.22, 0.68, 0.10, 'NH₃=10% (mid-reactor)'),
        (0.18, 0.55, 0.27, 'NH₃=27% (near outlet)'),
    ]

    for i, (y_N2, y_H2, y_NH3, label) in enumerate(compositions):
        rates = rate_vs_temperature(T_RANGE, y_N2, y_H2, y_NH3)
        ax.plot(T_RANGE, rates, color=COLORS[i], label=label)
        ax.fill_between(T_RANGE, 0, rates, alpha=0.07, color=COLORS[i])

    style_ax(ax,
             'Net Reaction Rate vs Temperature\n(Temkin-Dyson-Simon model, P=90 bar)',
             'Temperature (K)',
             'Net Rate r (kmol/m³/h)')

    ax.set_xlim(540, 785)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out('plot2_rate_vs_temp.png'), dpi=150,
                bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print("  ✓ Plot 2 saved: Rate vs Temperature")


# ─────────────────────────────────────────────
# PLOT 3 — Conversion vs Catalyst Weight
# ─────────────────────────────────────────────

def plot_conversion_profile():
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0f1117')

    temps = [548, 598, 648, 698, 748, 773]

    for i, T in enumerate(temps):
        W, X, _ = simulate_reactor(T, ratio_H2_N2=3.0)
        label = f'T = {T} K  ({T-273}°C)'
        ax.plot(W, X * 100, color=COLORS[i % len(COLORS)], label=label)

        # Equilibrium line (dashed)
        X_eq = equilibrium_conversion(T)
        ax.axhline(X_eq * 100, color=COLORS[i % len(COLORS)],
                   linestyle='--', linewidth=0.8, alpha=0.5)

    ax.set_ylim(0, 105)
    style_ax(ax,
             'N₂ Conversion Along Reactor\n(dashed = equilibrium limit, H₂/N₂=3, P=90 bar)',
             'Catalyst Weight W (kg)',
             'N₂ Conversion X (%)')

    ax.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig(out('plot3_conversion_profile.png'), dpi=150,
                bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print("  ✓ Plot 3 saved: Conversion Profile")


# ─────────────────────────────────────────────
# PLOT 4 — Effect of H2/N2 Ratio
# ─────────────────────────────────────────────

def plot_ratio_effect():
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0f1117')

    ratios = [1.0, 1.5, 2.0, 3.0, 4.0]
    T_fixed = 673  # K

    for i, ratio in enumerate(ratios):
        W, X, _ = simulate_reactor(T_fixed, ratio_H2_N2=ratio)
        ax.plot(W, X * 100, color=COLORS[i], label=f'H₂/N₂ = {ratio:.1f}')

    style_ax(ax,
             f'Effect of H₂/N₂ Feed Ratio on N₂ Conversion\n(T={T_fixed} K = {T_fixed-273}°C, P=90 bar)',
             'Catalyst Weight W (kg)',
             'N₂ Conversion X (%)')

    ax.set_ylim(0, 105)
    ax.legend()

    # Annotation explaining the inhibition effect
    ax.annotate('Higher H₂/N₂ → initially faster\nbut H₂ can inhibit at high coverage',
                xy=(60, 55), fontsize=8, color='#a0aec0',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#1e2535',
                          edgecolor='#2e3650', alpha=0.8))

    plt.tight_layout()
    plt.savefig(out('plot4_ratio_effect.png'), dpi=150,
                bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print("  ✓ Plot 4 saved: H₂/N₂ Ratio Effect")


# ─────────────────────────────────────────────
# PLOT 5 — Equilibrium Conversion vs Temperature
# ─────────────────────────────────────────────

def plot_equilibrium():
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0f1117')

    T_eq_range = np.linspace(550, 780, 60)
    X_eq = [equilibrium_conversion(T) * 100 for T in T_eq_range]

    ax.plot(T_eq_range, X_eq, color='#e57373', linewidth=2.5,
            label='Equilibrium conversion')
    ax.fill_between(T_eq_range, 0, X_eq, alpha=0.1, color='#e57373')

    # Shade the "kinetically limited" and "equilibrium limited" zones
    ax.axvspan(548, 640, alpha=0.05, color='#4fc3f7',
               label='Kinetically limited zone')
    ax.axvspan(700, 773, alpha=0.05, color='#e57373',
               label='Equilibrium limited zone')
    ax.axvline(670, color='#ffb74d', linestyle='--', linewidth=1.2,
               label='Optimal T window (~670 K)')

    style_ax(ax,
             'Equilibrium N₂ Conversion vs Temperature\n(P=90 bar, H₂/N₂=3)',
             'Temperature (K)',
             'Equilibrium Conversion X_eq (%)')

    ax.set_ylim(0, 100)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out('plot5_equilibrium.png'), dpi=150,
                bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print("  ✓ Plot 5 saved: Equilibrium Conversion")


# ─────────────────────────────────────────────
# PLOT 6 — Summary Dashboard (all 4 key plots)
# ─────────────────────────────────────────────

def plot_dashboard():
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor('#0f1117')
    gs = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.35,
                           left=0.08, right=0.96, top=0.92, bottom=0.08)

    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    fig.suptitle('Ammonia Synthesis Kinetic Model — Fe Catalyst (Chapter 1)\n'
                 'Temkin–Dyson–Simon Rate Equation | P = 90 bar',
                 fontsize=13, fontweight='bold', color='#e8edf5', y=0.97)

    # ── Panel A: Arrhenius (log y-axis) ──
    ax = axes[0]
    k_vals = arrhenius(T_RANGE)
    ax.semilogy(T_RANGE, k_vals, color=ACCENT, linewidth=2)
    ax.fill_between(T_RANGE, k_vals * 0.001, k_vals, alpha=0.1, color=ACCENT)
    # log_y=True — avoids AutoMinorLocator warning on log scale
    style_ax(ax, 'A)  Rate Constant k vs T (Arrhenius)', 'T (K)', 'k (kmol/m³/h)', log_y=True)

    # ── Panel B: Rate vs Temperature ──
    ax = axes[1]
    for i, (y_N2, y_H2, y_NH3, label) in enumerate([
        (0.25, 0.74, 0.01, 'NH₃=1%'),
        (0.22, 0.68, 0.10, 'NH₃=10%'),
        (0.18, 0.55, 0.27, 'NH₃=27%'),
    ]):
        rates = rate_vs_temperature(T_RANGE, y_N2, y_H2, y_NH3)
        ax.plot(T_RANGE, rates, color=COLORS[i], label=label, linewidth=1.8)
    style_ax(ax, 'B)  Net Rate vs Temperature', 'T (K)', 'r (kmol/m³/h)')
    ax.legend(fontsize=8)

    # ── Panel C: Conversion Profile ──
    ax = axes[2]
    for i, T in enumerate([548, 623, 698, 773]):
        W, X, _ = simulate_reactor(T)
        ax.plot(W, X * 100, color=COLORS[i], label=f'{T} K', linewidth=1.8)
        X_eq = equilibrium_conversion(T)
        ax.axhline(X_eq * 100, color=COLORS[i], linestyle='--',
                   linewidth=0.7, alpha=0.5)
    ax.set_ylim(0, 105)
    style_ax(ax, 'C)  Conversion Along Reactor\n(dashed = equilibrium)',
             'Catalyst W (kg)', 'X_N₂ (%)')
    ax.legend(fontsize=8)

    # ── Panel D: H2/N2 Ratio Effect ──
    ax = axes[3]
    for i, ratio in enumerate([1.0, 2.0, 3.0, 4.0]):
        W, X, _ = simulate_reactor(673, ratio_H2_N2=ratio)
        ax.plot(W, X * 100, color=COLORS[i], label=f'H₂/N₂={ratio}', linewidth=1.8)
    ax.set_ylim(0, 105)
    style_ax(ax, 'D)  Effect of H₂/N₂ Feed Ratio\n(T=673 K)',
             'Catalyst W (kg)', 'X_N₂ (%)')
    ax.legend(fontsize=8)

    plt.savefig(out('plot6_dashboard.png'), dpi=150,
                bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print("  ✓ Plot 6 saved: Full Dashboard")


# ─────────────────────────────────────────────
# RUN ALL PLOTS
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n Generating all plots...\n")
    print(f" Saving to: {OUTPUT_DIR}\n")
    plot_arrhenius()
    plot_rate_vs_temp()
    plot_conversion_profile()
    plot_ratio_effect()
    plot_equilibrium()
    plot_dashboard()
    print(f"\n All plots saved to: {OUTPUT_DIR}")