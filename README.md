## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -e .
```


## Project Structure

```
src/
  pricing/
    fourier_pricing.py     # Lewis formula pricers (trapezoid + GK), BSM, IV inversion
    heston_model.py        # Single-maturity HestonModel with MC simulation and plots
  calibration/
    heston_calibration.py  # HestonCalibrator — multi-maturity, FD + analytical Gradient
  data/
    data_utils.py          # data cleaning, arbitrage filters, r/q fitting
  scripts/
    calibrate.py           # End to End calibration
    benchmark_calibration.py  # calibration speed experiment code

data/
  optionsdata.csv          # S&P 500 options snapshot 2019-11-13 (S0 = 3094)

bachelors latex/
  bachelors.tex            # Thesis source
  chapters/                # Chapter .tex files
```


