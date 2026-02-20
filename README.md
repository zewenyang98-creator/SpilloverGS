# SpilloverGS model (0D overtopping breach model)

'SpilloverGS' is a lightweight 0D numerical model to simulate 'overtopping-driven breach erosion' of a dam (e.g., landslide dam / natural dam) and the resulting flood hydrograph

This code was written by Zewen Yang and Daniel Garcia-Castellanos, and is developed as an extension of the original 'spillover' code (Author: Daniel Garcia-Castellanos, https://github.com/danigeos/spillover)



1. Model purpose
SpilloverGS is designed to calculate and simulate 'overtopping breach processes' for dams or sill, including the evolution of:
- lake level / lake storage,
- breach discharge,
- breach widening,
- incision rate

2. Key features

 (1) Overtopping breach simulation
   Simulates overtopping-driven breach erosion for dams / sill.

 (2) Model uses dam/sill grain size (D50)
   The model integrates the grain size D50 of the dam / sill material, improving the simulation by considering the particle size distribution of the material. 
   This makes the model more realistic and adaptive to various dam material types, reflecting their specific erosive behavior.

 (3) Dynamic slope computation  
   The hydraulic slope is updated dynamically at each time step

 (4) Grain-size-dependent roughness 
   When using the Manning's velocity formulation, the Manning roughness coefficient is computed from dam-material grain size D    

 (5) Vertical incision + lateral widening 
   - Vertical incision is computed using an erosion law based on Shields stress exceedance
   - Lateral widening is linked to vertical incision (symmetric widening on both sides)

 (6) Simple and lightweight
   The code is compact, easy to run, and produces 'four key plots':
   - Lake elevation vs time
   - Discharge vs time
   - Breach width vs time
   - Incision rate vs time

 (7) Discharge time series output  
   The program computes a full 'breach discharge time series'
   (Optional: you can easily save `t_h` and `Q` to a text/CSV file; see “Export results” below.)

 (8) Extended from 'spillover' model
   SpilloverGS is developed on top of the original 'spillover' model by Daniel Garcia-Castellanos (2009–2018, d.g.c@csic.es).


3. Files
- SpilloverGS_code.py (main script)
- SpilloverGS_params.txt (input parameter file, including optional z–V data block)


4. How to run

 (1) Make sure `SpilloverGS_code.py` and `SpilloverGS_params.txt` are in the same folder
 
 (2) Edit numbers in `SpilloverGS_params.txt` (do not change parameter names)
 
 (3) Run: bash python SpilloverGS_code.py
