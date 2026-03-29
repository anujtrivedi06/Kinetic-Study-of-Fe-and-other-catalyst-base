import numpy as np
from scipy.integrate import solve_ivp

R        = 8.314          # Universal gas constant, J/mol·K
k0       = 6.5e13         # Pre-exponential factor, kmol/m³/h
Ea       = 159400         # Activation energy, J/mol (159.4 kJ/mol)
alpha    = 0.654          # Temkin parameter
P_bar    = 90.0           # Operating pressure, bar
P_Pa     = P_bar * 1e5    # Pressure in Pascals


# STEP 1 — ARRHENIUS: Rate constant vs Temperature
def arrhenius(T):
    """
    Returns the rate constant k at temperature T (K).
    k = k0 * exp(-Ea / RT)
    Units: kmol/m³/h
    """
    return k0 * np.exp(-Ea / (R * T))

# STEP 2 — TEMKIN RATE EQUATION
def temkin_rate(T, y_N2, y_H2, y_NH3, P=P_bar):
    """
    Temkin-Dyson-Simon rate equation.
    
    r = k*(aN2 * aH2^3 / aNH3^2)^alpha  -  k*(aNH3^2 / aH2^3)^(1-alpha)

    Activities approximated as partial pressures (Pi = yi * P).
    Returns rate in kmol/m³/h.
    """
    k = arrhenius(T)

    # Partial pressures (bar)
    P_N2  = y_N2  * P
    P_H2  = y_H2  * P
    P_NH3 = y_NH3 * P

    # Avoid division by zero / negative values
    P_N2  = max(P_N2,  1e-12)
    P_H2  = max(P_H2,  1e-12)
    P_NH3 = max(P_NH3, 1e-12)

    forward = k * ((P_N2 * P_H2**3) / P_NH3**2) ** alpha
    reverse = k * ((P_NH3**2)       / P_H2**3)   ** (1 - alpha)

    return forward - reverse   # net rate


# STEP 3 — ODE SYSTEM (Fixed-Bed Reactor)
def reactor_odes(W, F, T, P, F_total_in):
    """
    Molar flow balance along catalyst weight W (kg).
    
    State vector F = [F_N2, F_H2, F_NH3]  in mol/s
    
    Stoichiometry:  N2 + 3H2 -> 2NH3
      dF_N2  / dW = -r
      dF_H2  / dW = -3r
      dF_NH3 / dW = +2r
    
    r is converted from kmol/m³/h to mol/s/kg_cat using bulk density.
    """
    F_N2, F_H2, F_NH3 = F
    F_total = F_N2 + F_H2 + F_NH3

    # Mole fractions
    y_N2  = F_N2  / F_total
    y_H2  = F_H2  / F_total
    y_NH3 = F_NH3 / F_total

    # Rate in kmol/m³/h → convert to mol/s/kg_cat
    # Using bulk density of magnetite ~ 2850 kg/m³
    rho_cat  = 2850          # kg/m³
    r_vol    = temkin_rate(T, y_N2, y_H2, y_NH3, P)   # kmol/m³/h
    r_mass   = r_vol * 1000 / 3600 / rho_cat           # mol/s/kg_cat

    dF_N2  = -r_mass
    dF_H2  = -3 * r_mass
    dF_NH3 = +2 * r_mass

    return [dF_N2, dF_H2, dF_NH3]



def simulate_reactor(T, ratio_H2_N2=3.0, P=P_bar, W_max=100.0, n_points=500):
    """
    Simulates the fixed-bed reactor at temperature T (K).
    
    Parameters:
        T           : Temperature in K
        ratio_H2_N2 : H2/N2 molar feed ratio (default 3.0)
        P           : Pressure in bar (default 90)
        W_max       : Total catalyst weight in kg (default 100)
        n_points    : Resolution of output
    
    Returns:
        W_span  : catalyst weight array (kg)
        X_N2    : nitrogen conversion array (0 to 1)
        F_out   : final molar flows [F_N2, F_H2, F_NH3]
    """
    # Feed molar flows (basis: 1 mol/s total)
    y_N2_in = 1.0 / (1.0 + ratio_H2_N2)
    y_H2_in = ratio_H2_N2 / (1.0 + ratio_H2_N2)

    F_N2_0  = y_N2_in   # mol/s
    F_H2_0  = y_H2_in   # mol/s
    F_NH3_0 = 1e-10     # tiny seed to avoid log(0)
    F0      = [F_N2_0, F_H2_0, F_NH3_0]
    F_total_in = F_N2_0 + F_H2_0 + F_NH3_0

    W_span  = [0, W_max]
    W_eval  = np.linspace(0, W_max, n_points)

    sol = solve_ivp(
        reactor_odes,
        W_span,
        F0,
        args=(T, P, F_total_in),
        t_eval=W_eval,
        method='RK45',
        rtol=1e-6,
        atol=1e-9
    )

    F_N2  = sol.y[0]
    X_N2  = (F_N2_0 - F_N2) / F_N2_0   # Conversion of N2

    return sol.t, np.clip(X_N2, 0, 1), sol.y



# STEP 5 — HELPER: Rate vs Temperature at fixed composition
def rate_vs_temperature(T_range, y_N2=0.25, y_H2=0.75, y_NH3=0.01, P=P_bar):
    """Returns reaction rate at each temperature for fixed composition."""
    return np.array([temkin_rate(T, y_N2, y_H2, y_NH3, P) for T in T_range])


# STEP 6 — HELPER: Equilibrium approximation
def equilibrium_conversion(T, P=P_bar, ratio_H2_N2=3.0):
    """
    Approximate equilibrium conversion of N2 using Temkin's correlation.
    ln(Keq) = -2*(delta_G / RT)  simplified form.
    Uses empirical correlation: log10(Keq) = -2.691122*log10(T) - 5.519265e-5*T + 1.848863e-7*T^2 + 2001.6/T + 2.6899
    Valid ~300-600°C range.
    """
    log10_Keq = (-2.691122 * np.log10(T)
                 - 5.519265e-5 * T
                 + 1.848863e-7 * T**2
                 + 2001.6 / T
                 + 2.6899)
    Keq = 10 ** log10_Keq   # Keq in atm^-1

    # Convert pressure to atm
    P_atm = P / 1.01325

    # Solve for equilibrium conversion (simplified, stoichiometric feed H2/N2=3)
    # At equilibrium: Keq = (2x)^2 / ((1-x)(3-3x)^3) * (1/(P*(4-2x)))^-2  [approx]
    # Numerical scan is cleaner
    X_vals = np.linspace(0.001, 0.999, 5000)
    best_X = 0.0
    best_err = 1e18
    for X in X_vals:
        n_N2  = 1 - X
        n_H2  = ratio_H2_N2 - 3*X
        n_NH3 = 2*X
        n_tot = n_N2 + n_H2 + n_NH3
        if n_H2 <= 0:
            break
        P_N2  = (n_N2  / n_tot) * P_atm
        P_H2  = (n_H2  / n_tot) * P_atm
        P_NH3 = (n_NH3 / n_tot) * P_atm
        Keq_calc = P_NH3**2 / (P_N2 * P_H2**3)
        err = abs(Keq_calc - Keq)
        if err < best_err:
            best_err = err
            best_X   = X
    return best_X


if __name__ == "__main__":
    T_test = 673  
    k_test = arrhenius(T_test)
    r_test = temkin_rate(T_test, 0.25, 0.75, 0.01)
    print(f"T = {T_test} K")
    print(f"k = {k_test:.4e} kmol/m³/h")
    print(f"r = {r_test:.4e} kmol/m³/h")

    W, X, _ = simulate_reactor(T_test)
    print(f"Final N2 conversion at W=100 kg: {X[-1]*100:.2f}%")
    # print("Model OK.")