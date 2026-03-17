# Custom inference procedures for LV and Nonlinear SSM probabilistic models

This repository contains Julia code for experimenting with inference in two probabilistic models implemented with `Gen.jl`:

- `LV/`: Lotka-Volterra predator-prey model.
- `Nonlinear_SSM/`: nonlinear state-space model.

For each model, the repository includes:

- a `*_custom_inference.jl` file with custom inference procedure and helper utilities,
- a `*_compare_inferences.jl` file for running and comparing custom made inference procedure with out-of-the-box procedures.

The code is focused on posterior inference, runtime comparison, and summary statistics for the model parameters under a custom made inference procedure.
Description of the custom procedure and results with out-of-the-box procedures are available in report.pdf.