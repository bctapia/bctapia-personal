# -*- coding: utf-8 -*-

'''
This code solve the rate equation
r = k * P_A**-0.94027 * P_B**0.968224
which was found for the heterogeneous CO oxidation reaction:
2CO + O2 -> 2CO2 simplified as 2A + B -> 2C
For explanation of any variables (e.g., alpha), see:
Fogler, H.S. Elements of Chemical Reaction Engineering (2020)
'''

# Imports
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
from matplotlib import pyplot as plt


# Reaction parameters
alpha = 0     # assuming no pressure drop
P_0 = 101325  # Pa; Inlet pressure
A = 61698     # 1/(Pa^(a+b)*s*g); Pre-exponential factor
Ea = 95947    # J/mol; Activation Energy
R = 8.314     # J/mol/K; Ideal gas constant
T = np.arange(130, 280) + 273.15  # K; temperature range studied
F_C0 = 0                          # mol/s; No inlet CO2
F_A0 =  3.71813E-7                # mol/s; Inlet CO
F_B0 =  3.71813E-6                # mol/s; Inlet O2
F_inerts = 3.30913E-5             # mol/s; Inlet inters
F_T0 = F_A0 + F_B0 + F_inerts     # mol/s; Total inlet flow
exp_cat = 0.0003924               # g; catalyst weight studied

# define the reaction phase
gas_phase = True
liquid_phase = False


def dFdW(Wspan, F, *args):
      
    '''
    function to simulate differential equation
    Wspan: catalyst weight (independent variable)
    F: flowrates & pressure drop (dependent variables)
    Size of F = number of components + 1
    *args: additional arguments to pass (e.g., T)
    '''

    F_A = F[0]
    F_B = F[1]
    F_C = F[2]
    F_T = F_A + F_B + F_C + F_inerts

    y = F[3] # pressure drop term

    if gas_phase is True and liquid_phase is False:
        v = v_0 * (F_T / (y*F_T0)) # IF GAS-PHASE REACTION
    elif gas_phase is False and liquid_phase is True:
        v = v_0 # IF LIQUID-PHASE REACTION 
    else:
        print('Must modify v \
            Only account for the volume change in gas phase \
            e.g. v = v_0_g * (F_T_g / (y*F_T0_g)) + v_0_l')
        return


    c_A = F_A / v
    c_B = F_B / v
    c_C = F_C / v

    # converting concentrations to partial pressures based on ideal gas law
    P_A = c_A*R*T[i]
    P_B = c_B*R*T[i]
    P_C = c_C*R*T[i]

    # ensure no errors due to negative pressures during iterations
    P_A = max(P_A, 1e-20)
    P_B = max(P_B, 1e-20)
    P_C = max(P_C, 1e-20)

    # rate law from experimental measurements
    r_A = -(k * P_A**-0.94027 * P_B**0.968224)

    dFdW = [r_A,                       # r_CO
            (1/2)*r_A,                 # r_O2
            -1 * r_A,                  # r_CO2
            -alpha * F_T / (2*y*F_T0)] # pressure drop

    return dFdW

# Arguments for solve_ivp
initial_conditions = [F_A0, F_B0, F_C0, 1]
w_bounds = [0, 0.001]
w_span = np.linspace(w_bounds[0], w_bounds[1], 100000)

X_A_array = []

for i, _ in enumerate(T):

    v_0 = F_T0*R*T[i]/P_0    # m^3/s
    k = A*np.exp(-Ea/R/T[i]) # 1/(Pa^(a+b)*s*g)

    # solving ODE
    sol = solve_ivp(dFdW, w_bounds, initial_conditions,
                    t_eval=w_span, dense_output=True, method='Radau')

    # unpacking molar flow rates
    F_A = sol.y[0,:]
    F_B = sol.y[1,:]
    F_C = sol.y[2,:]
    y = sol.y[3,:]
    F_T = F_A + F_B + F_C + F_inerts

    v = v_0 * (F_T / F_T0)  # unpacking volumetric flow rate

    # calculating concentrations
    c_A1 = F_A / v
    c_B1 = F_B / v
    c_C1 = F_C / v

    X_A = 1-F_A/F_A0  # calculating conversion

    w1 = sol.t                         # unpacking catalyst weight
    index = np.argmin(abs(w1-exp_cat)) # finding index of exp_cat in W
    X_A_single = X_A[index]            # finding X at exp_cat
    X_A_array = np.append(X_A_array, X_A_single)

plt.plot(T-273.15, X_A_array*100)
plt.title('Conversion vs Temperature')
plt.xlabel('Temperature ($^o$C)')
plt.ylabel('Conversion (%)')
matplotlib.pyplot.show()

print('Done!')
