# SpilloverGS_code
# 0D overtopping-breach model


# lake_infor_mode: 0=z-V block, 1=A0 build z-V
# lake_shape_mode: 0=linear, 1=box-like (only when lake_infor_mode=1)
# flow_velocity_mode: 0=Manning, 1=Critical (U=sqrt(g*h))
# export_results_mode: 0=NOT export, 1=export results to file "Export results.txt"

import numpy as np
import matplotlib.pyplot as plt
import re

# File name of Input parameters
PARAM_FILE = "SpilloverGS_params.txt"

# 1) Read parameters from .txt (SpilloverGS_params.txt)
def load_params_from_txt(filepath: str) -> dict:
    # required parameters from external .txt file
    required = {"z_sill0", "z_sill_min", "z_l0", "z_bed", "L_dam", "W0", "Qin", "D", "ap", "tmax", "dt"}

    # if missing, default values will be assigned later
    optional = {"lake_infor_mode", "A0", "lake_shape_mode", "flow_velocity_mode", "export_results"}

    # create empty dictionary to store parameters
    params = {}

    # open txt file for reading
    with open(filepath, "r", encoding="utf-8") as f:
        # Read file line by line
        for raw in f:

            # remove whitespace
            line = raw.strip()
            if (not line) or line.startswith("#"):
                continue

            # remove inline comments:  # Qin = 1680   # inflow, becomes: # Qin = 1680
            if "#" in line:
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue

            # use regex to detect parameter assignment:
            m = re.search(r"([A-Za-z_]\w*)\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line)
            # If line doesn't match parameter format, will be ignored
            if not m:
                continue

            # extract parameter name and value
            key = m.group(1).strip()
            val = float(m.group(2))

            # store recognized parameters
            if (key in required) or (key in optional):
                params[key] = val
    # check whether any required parameter is missing
    missing = sorted(list(required - set(params.keys())))
    if missing:
        raise ValueError(f"Missing required parameters in '{filepath}': {missing}")
    # assign default values for optional parameters if .txt file is not provided
    params.setdefault("lake_shape_mode", 0)
    params.setdefault("flow_velocity_mode", 0)
    params.setdefault("export results", 0)

    return params


# 2) Read z-V block from .txt if initial lake information mode 0 is used
def load_stage_storage_from_same_txt(filepath: str):
    z_list, v_list = [], []
    reading = False

    with open(filepath, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if (not line) or line.startswith("#"):
                continue
            if "#" in line:
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue

            parts = line.split()
            if len(parts) >= 2:
                try:
                    z_val = float(parts[0])
                    v_val = float(parts[1])
                    z_list.append(z_val)
                    v_list.append(v_val)
                    reading = True
                    continue
                except:
                    pass

            if reading:
                break

    if len(z_list) < 2:
        raise ValueError("No valid z-V block found in txt. Please append two-column numeric data (z V).")

    z_arr = np.array(z_list, dtype=float)
    v_arr = np.array(v_list, dtype=float)

    if np.any(np.diff(z_arr) <= 0):
        raise ValueError("z in z-V block must be strictly increasing.")
    if np.any(np.diff(v_arr) < 0):
        raise ValueError("V in z-V block should be non-decreasing.")

    return z_arr, v_arr


# 3) Build synthetic z-V from A0 with shape_mode
def build_stage_storage_from_A0(A0: float, z_bed: float, z_l0: float,
                               shape_mode: int = 0, npts: int = 200, z_top_extra: float = 50.0):
    if A0 <= 0:
        raise ValueError("A0 must be > 0 when lake_infor_mode = 1.")
    if z_l0 <= z_bed:
        raise ValueError("z_l0 must be greater than z_bed.")

    shape_mode = int(shape_mode)
    if shape_mode not in (0, 1):
        raise ValueError("lake_shape_mode must be 0 (linear) or 1 (box-like).")

    z_top = float(z_l0 + z_top_extra)

    # uniform elevation samples from lake bottom upward
    z = np.linspace(float(z_bed), z_top, int(npts))

    # mode 1: box-like lake (constant area)
    if shape_mode == 1:
        V = A0 * (z - z_bed)
        return z.astype(float), V.astype(float)


    # mode 0: linear widening lake
    dz0 = (z_l0 - z_bed)
    A = np.empty_like(z)
    below = z <= z_l0
    A[below] = A0 * (z[below] - z_bed) / dz0
    A[~below] = A0

    # integrate area to obtain volume
    V = np.zeros_like(z)

    # trapezoidal integration layer by layer
    for i in range(1, len(z)):
        dz = z[i] - z[i - 1]
        V[i] = V[i - 1] + 0.5 * (A[i] + A[i - 1]) * dz

    V = np.maximum.accumulate(V)
    return z.astype(float), V.astype(float)


# 4) Read input parameters
p = load_params_from_txt(PARAM_FILE)
lake_infor_mode = int(round(float(p.get("lake_infor_mode", 0.0))))
lake_shape_mode = int(round(float(p.get("lake_shape_mode", 0.0))))
A0_user = float(p.get("A0", 0.0))
flow_velocity_mode = int(round(float(p.get("flow_velocity_mode", 0.0))))
export_results = int(round(float(p.get("export_results", 0.0))))


# Constants
g = 9.81         # acceleration of gravity
rho = 1000.0     # water density
rhos = 2650.0    # sediment density
nu = 1.0e-6      # kinematic viscosity of water at 20°C

Qin = float(p["Qin"])                  # inflow discharge
L_dam = float(p["L_dam"])              # dam length
z_bed = float(p["z_bed"])              # lake bottom elevation
z_l0 = float(p["z_l0"])                # lake initial elevation
z_sill0 = float(p["z_sill0"])          # dam initial elevation
z_sill_min = float(p["z_sill_min"])    # dam final elevation
W0 = float(p["W0"])                    # Initial breach width
D = float(p["D"])                      # Grain size of dam materials
ap = float(p["ap"])                    # A empirical parameter, suggested give *×10^(-4)
tmax = float(p["tmax"]) * 3600.0       # Breach total time
dt = float(p['dt'])


# Two modes are provided of intical lake information as follows
# 0: read 'Lake Level_z - Storage_V Data (z-V)' block in .txt
# 1:build z-V from initial lake area A0
if lake_infor_mode == 0:
    zV_pts, V_pts_m3 = load_stage_storage_from_same_txt(PARAM_FILE)
else:
    zV_pts, V_pts_m3 = build_stage_storage_from_A0(
        A0=A0_user, z_bed=z_bed, z_l0=z_l0,
        shape_mode=lake_shape_mode
    )

# 5) z-Area preprocessing from z-V; calculate z from V
# z is lake elevation
# V is lake volume
def prepare_hypsometry(z_pts: np.ndarray, v_pts: np.ndarray) -> np.ndarray:
    z_pts = np.asarray(z_pts, dtype=float)
    v_pts = np.asarray(v_pts, dtype=float)
    if np.any(np.diff(z_pts) <= 0):
        raise ValueError("zV_pts must be strictly increasing.")

    areas = np.zeros_like(z_pts, dtype=float)

# A0 = (V1-v0)/(z1-z0)
    dz0 = z_pts[1] - z_pts[0]
    areas[0] = (v_pts[1] - v_pts[0]) / dz0

# solve lake area

# ΔVi = (A_{i-1} + A_{i})/2 * Δz
# drive A_{i} = 2 * dV/dz - A_{i-1}
    for i in range(1, len(z_pts)):
        dz = z_pts[i] - z_pts[i - 1]
        dv = v_pts[i] - v_pts[i - 1]
        a_new = (2.0 * dv / dz) - areas[i - 1]
        areas[i] = max(a_new, areas[i - 1] * 0.1)
    return areas

A_pts = prepare_hypsometry(zV_pts, V_pts_m3)


# Assume lake area varies linearly within each elevation layer:
#     A(z) = A_{i-1} + s(z - z_{i-1}),  s = (A_i - A_{i-1})/dz
# Volume from the layer bottom to height dz:
#     V = A_{i-1}*dz + (s/2)*dz^2
# Setting V = target_dv gives a quadratic equation:
#     (s/2)*dz^2 + A_{i-1}*dz - target_dv = 0
# Solve for dz and obtain water level:
#     z = z_{i-1} + dz
# If s≈0, use linear solution: dz = target_dv / A_{i-1}


def get_level_from_volume(vol_target: float, z_pts: np.ndarray, a_pts: np.ndarray) -> float:
    if vol_target <= 0.0:
        return float(z_pts[0])

    vol_acc = 0.0
    for i in range(1, len(z_pts)):
        dz_layer = float(z_pts[i] - z_pts[i - 1])
        dvol = dz_layer * float(a_pts[i] + a_pts[i - 1]) / 2.0

        if vol_acc + dvol >= vol_target:
            target_dv = vol_target - vol_acc
            slope = float(a_pts[i] - a_pts[i - 1]) / dz_layer

            a_coeff = slope / 2.0
            b_coeff = float(a_pts[i - 1])
            c_coeff = -target_dv

            if abs(a_coeff) < 1e-12:
                dz = -c_coeff / max(b_coeff, 1e-12)
            else:
                delta = b_coeff * b_coeff - 4.0 * a_coeff * c_coeff
                dz = (-b_coeff + np.sqrt(max(0.0, delta))) / (2.0 * a_coeff)

            return float(z_pts[i - 1] + dz)

        vol_acc += dvol

    return float(z_pts[-1] + (vol_target - vol_acc) / max(float(a_pts[-1]), 1e-12))


# 6) Main calculation
W_max = 10000.0   # max breach width (m)
np_exp = 1.5      # power law index of Lamb (2015) erosion equation
n = 0.047 * D ** (1.0 / 6.0)     # Manning’s roughness coefficient
R = rhos / rho - 1.0     # related density
D_star = D * (((rhos - rho) * g) / (rho * nu**2)) ** (1.0 / 3.0)  # dimensionless particle diameter
tau_c_star = 0.5 * (0.22 * (D_star ** (-0.9)) + 0.06 * (10.0 ** (-7.7 * (D_star ** (-0.9)))))  # critical Shields number from Garcia and parker (1991)

# Time discretization
# dt: time step size (s), read from input file
# tmax: total simulation duration (s) , and from input file (unit h)
# Nt: total number of time points in the simulation
# - the first time point is exactly t = 0
# - the last time point does not exceed tmax
# - both initial and final states are included
# t_axis: time array in seconds, from 0 to tmax
# t_h: same time array converted to hours (used for plotting/output)
Nt = int(np.floor(tmax / dt) + 1)
t_axis = np.linspace(0.0, tmax, Nt)
t_h = t_axis / 3600.0

# Initialize state variables
zl = np.zeros(Nt, dtype=float)           # lake level time series
zs = np.zeros(Nt, dtype=float)           # sill elevation time series
W  = np.zeros(Nt, dtype=float)           # breach width time series
Q  = np.zeros(Nt, dtype=float)           # discharge time series
S_hist = np.zeros(Nt, dtype=float)       # hydraulic slope time series
dzsdt_hist = np.zeros(Nt, dtype=float)   # sill erosion rate time series

# set initial conditions at t = 0
zl[0] = z_l0
zs[0] = z_sill0
W[0]  = W0


# Compute initial lake volume from stage–storage curve
# Interpolate V(z) using the provided z–V relationship
# so that the mass balance equation can be solved:
#     dV/dt = Qin - Qout
current_vol = float(np.interp(zl[0], zV_pts, V_pts_m3))

print("Starting calculation...")
print("Time(h)\tLake(m)\tSill(m)\th(m)\tW(m)\tQout(m3/s)\tQin(m3/s)\tS(-)\tdzsdt(m/s)")

# print output interval
print_interval = 1/6 * 3600.0
last_print_time = -print_interval


# Time marching loop (explicit Euler update)
# At each time step i:
#   1) compute dynamic slope S(t) and water depth h(t)
#   2) compute breach velocity U(t) and discharge Qout(t)
#   3) update lake volume by mass balance equation: dV/dt = Qin - Qout
#   4) invert V -> z_l using stage–storage curve
#   5) compute Shields stress and erosion rate dzs/dt
#   6) update sill elevation zs and breach width W

for i in range(0, Nt - 1):
    # dynamic slope
    S_dyn = max((zl[i] - z_bed) / L_dam, 1e-6)
    S_hist[i] = S_dyn

    # breach flow depth over sill / dam (head above sill / dam)
    h_curr = max(zl[i] - zs[i], 0.0)

    # If no water overtops the sill, skip hydraulic/erosion update
    # Here we keep the state unchanged and move to next step
    if h_curr <= 0.0 and i > 0:
        zl[i + 1] = zl[i]
        zs[i + 1] = zs[i]
        W[i + 1]  = W[i]
        Q[i + 1]  = Q[i]
        dzsdt_hist[i] = 0.0
        continue

    # Compute breach flow velocity
    # Mode 0: Manning U = (1/n)*h^(2/3)*S^(1/2)
    # Mode 1: Critical-flow-like U = sqrt(g*h)
    if flow_velocity_mode == 0:
        v_eff = (1.0 / n) * (h_curr ** (2.0 / 3.0)) * np.sqrt(S_dyn)
    elif flow_velocity_mode == 1:
        v_eff = np.sqrt(g * h_curr)
    else:
        raise ValueError("flow_velocity_mode must be 0 (Manning) or 1 (Critical).")

    # Breach discharge
    # Qout = U * A = U * W * h
    Q[i] = W[i] * h_curr * v_eff

    # Lake volume update by mass balance equation
    # dV/dt = Qin - Qout
    # Explicit Euler: V^{i+1} = V^{i} + (Qin - Qout^i)*dt
    current_vol += (Qin - Q[i]) * dt
    current_vol = max(0.0, current_vol)

    # Convert updated volume to updated lake level: z_l = z(V)
    zl[i + 1] = get_level_from_volume(current_vol, zV_pts, A_pts)

    # Compute Shields stress and vertical erosion rate
    # Bed shear stress: tau = rho*g*h*S
    # Shields: tau* = tau / [(rho_s-rho) g D] = h*S/(R*D)
    # Erosion equation: dzs/dt = -ap*(tau* - tau_c*)^np_exp * sqrt(R*g*D) from Lamb (2015)
    tau_star = h_curr * S_dyn / (R * D)
    dzsdt = 0.0
    if (tau_star > tau_c_star) and (zs[i] > z_sill_min + 1e-12):
        dzsdt = -ap * ((tau_star - tau_c_star) ** np_exp) * np.sqrt(R * g * D)

    dzsdt_hist[i] = dzsdt

    # Update sill/dam elevation (incision) and breach width (widening)
    # Sill/dam: z_s^{i+1} = z_s^i + dzsdt*dt, limited by z_sill_min
    zs[i + 1] = max(zs[i] + dzsdt * dt, z_sill_min)

    # Width widening: dW/dt = -2 * dzsdt (2 times is assumes symmetric widening on both sides)
    W[i + 1] = min(W[i] + 2.0 * max(-dzsdt, 0.0) * dt, W_max)

    # Print status every print_interval seconds
    if (t_axis[i] - last_print_time) >= print_interval:
        print(f"{t_axis[i]/3600.0:.2f}\t{zl[i]:.2f}\t{zs[i]:.2f}\t{h_curr:.2f}\t"
              f"{W[i]:.1f}\t{Q[i]:.1f}\t{Qin:.1f}\t{S_dyn:.4g}\t{dzsdt:.3e}")
        last_print_time = t_axis[i]


# Fill the last time-step values (Nt-1) for plotting continuity
S_hist[-1] = S_hist[-2]
Q[-1] = Q[-2]
W[-1] = W[-2]
zs[-1] = zs[-2]
zl[-1] = zl[-2]

# because the loop mainly writes up to index Nt-2.
dzsdt_hist[-1] = dzsdt_hist[-2]


# 7) Plotting
# lines color in four subplot
COL_LAKE  = "tab:blue"
COL_Q     = "tab:orange"
COL_W     = "tab:green"
COL_EROS  = "tab:purple"

# create a 2x2 subplot figure
fig, axes = plt.subplots(2, 2, figsize=(12, 7))
plt.gcf().set_facecolor('w')


# panel 1: Lake level
axes[0, 0].plot(t_h, zl, color=COL_LAKE, lw=2.0, label="Lake level")
axes[0, 0].set_ylabel("Lake elevation (m)", fontsize=14)
axes[0, 0].set_xlabel("Time since breach (h)", fontsize=14)
axes[0, 0].tick_params(labelsize=12)
axes[0, 0].grid(True)
axes[0, 0].legend(fontsize=13)

# panel 2: Discharge
axes[0, 1].plot(t_h, Q, color=COL_Q, lw=2.0, label="Discharge")
axes[0, 1].set_ylabel("Discharge (m$^3$/s)", fontsize=14)
axes[0, 1].set_xlabel("Time since breach (h)", fontsize=14)
axes[0, 1].tick_params(labelsize=12)
axes[0, 1].grid(True)
axes[0, 1].legend(fontsize=13)

# panel 3: Breach width
axes[1, 0].plot(t_h, W, color=COL_W, lw=2.0, label="Breach width")
axes[1, 0].set_ylabel("Breach width (m)", fontsize=14)
axes[1, 0].set_xlabel("Time since breach (h)", fontsize=14)
axes[1, 0].tick_params(labelsize=12)
axes[1, 0].grid(True)
axes[1, 0].legend(fontsize=13)

# panel 4: Incision rate dzs/dt
axes[1, 1].plot(t_h, dzsdt_hist, color=COL_EROS, lw=2.0, label="Incision rate dzs/dt")
axes[1, 1].set_ylabel(r"d$z_s$/dt (m/s)", fontsize=14)
axes[1, 1].set_xlabel("Time since breach (h)", fontsize=14)
axes[1, 1].tick_params(labelsize=12)
axes[1, 1].grid(True)
axes[1, 1].legend(fontsize=13)

plt.tight_layout()


# 8) Outputs
A0_out = float(np.interp(zl[0], zV_pts, A_pts))
final_erosion_depth = float(z_sill0 - zs[-1])
Q_peak = float(np.max(Q))
i_peak = int(np.argmax(Q))
t_peak_h = float(t_axis[i_peak] / 3600.0)


print("\n================== Model Outputs ==================")
print(f"1) Lake area at breach start A0 = {A0_out:.3e} m^2  (at zl0 = {zl[0]:.3f} m)")
print(f"2) Final erosion depth (z_sill0 - z_sill_end) = {final_erosion_depth:.3f} m")
print(f"3) Grain size D used = {float(D):.6g} m")
print(f"4) Peak discharge Q_peak = {Q_peak:.3f} m^3/s  (at t = {t_peak_h:.3f} h)")
print(f"5) ap used = {float(ap):.6g}")
print(f"6) tau_c*={tau_c_star:.5f}")
print(f"7) Grain size D={D:.4f} m")
if flow_velocity_mode == 0:
    print(f"8) Manning’s roughness coefficient n={n:.4g}")
else:
    print("8) Manning’s roughness coefficient is not calculated.")

print(f"9) Qin={Qin:.1f} m3/s")
print("===================================================\n")

print(f"[MODE INFO] lake_infor_mode={lake_infor_mode} (0=z-V, 1=A0), lake_shape_mode={lake_shape_mode} (0=linear,1=box)")
print(f"[MODE INFO] flow_velocity_mode={flow_velocity_mode} (0=Manning's formular, 1=Critical flow formular)")
print(f"[MODE INFO] export_results={export_results} (0=no export, 1=export)")



# Export results to .txt file (optional)
if export_results == 1:

    with open("Export_results.txt", "w", encoding="utf-8") as file:

        # header
        file.write("Time (h)\tLake Elevation (m)\tDischarge (m^3/s)\tBreach Width (m)\tIncision dzs/dt (m/s)\n")

        # data rows
        for i in range(Nt):
            file.write(f"{t_h[i]:.4f}\t{zl[i]:.3f}\t{Q[i]:.3f}\t{W[i]:.3f}\t{dzsdt_hist[i]:.6e}\n")

    print("[EXPORT INFO] Results exported to 'Export_results.txt'")

else:
    print("[EXPORT INFO] Result export disabled (export_results = 0)")


plt.show()
