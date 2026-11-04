# Endpoint readout follow-up

Exploratory: previous test outcomes were already known. Native TRIBE is unchanged.

| Model | Normalized MSE | Mean signed correlation |
|---|---:|---:|
| tribe_endpoint | 1.075506 | -0.01347 |
| full_endpoint | 0.906187 | 0.00743 |
| tribe_calibrated | 0.785250 | 0.03698 |
| full_calibrated | 0.789026 | 0.02903 |
| previous_mean | 0.776981 | undefined |
| previous_acoustic_ridge | 0.785250 | 0.03698 |
| previous_tribe_tcn | 1.106498 | 0.01125 |
| previous_full_tcn | 0.949433 | -0.07320 |

Full blend neural weight: 0.2. Zero means acoustic-only.
Promotion gate passed: False.

All development weights, individual listeners, clip results and wrong-content controls are in results.json.
No measured-fMRI improvement, educational benefit or broad brain-prediction claim is established.

Calibration detail: full uses 20% endpoint network and 80% acoustic ridge; TRIBE calibration selected 0% network and is acoustic-only.

The average correlation improved over the old full network, but remains weak; clip 11 is negative. The new blend does not beat acoustic ridge or the mean-error baseline. It is not promoted to the application.
