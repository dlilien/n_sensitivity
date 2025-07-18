# MISMIP+ suite

## Steady state
Run mismip_ssa_init.py followed by mismip_hybrid_init.py. Note, you probably want a lot of cores for the latter...


## Initialization
Run initialize_mismip.py and initialize_ssa_mismip.py for the different conditions (sliding laws, values of n)

Then, run postprocess_inversions.py and postprocess_ssa_inversion.py

Run plot_init.py to analyze the initialized states.

## Transient

Run transient_mismip.py for each forcing (retreat, unperturbed), each sliding law, each n, and initialization (standard, identical)..

Run transient_ssa_mismip.py for each forcing (retreat, unperturbed), each n, and initialization (standard, identical); it automatically does all sliding laws for each.


## Plotting

Run transient_plots.py to analyze the results.
