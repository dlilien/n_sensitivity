# Flowline simulations

## Steady state
Run true_flowline.py at the desired temperatures (-20, -10, -5 in the paper).

Run initialize_identical.py.

## Inversions
Run initialize_standard.py for each n and temperature.

Plot results with analyze_initialization.py.

## Forward modeling
Run transient_simulation.py for each forcing (retreat, unperturbed), n, and initialization (standard, identical).

Run analyze_transient.py to plot the results.
