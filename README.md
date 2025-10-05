# ExoScan AI 🔭

AI-powered exoplanet classification system using XGBoost machine learning across three NASA missions: Kepler, K2, and TESS.

![ExoScan AI](https://img.shields.io/badge/Accuracy-99.27%25-success)
![Missions](https://img.shields.io/badge/Missions-3-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

ExoScan AI automates exoplanet validation by classifying candidates from NASA's transit survey missions. The system analyzes transit photometry data to distinguish between confirmed planets, planet candidates, and false positives with high accuracy.

### Model Performance

| Mission | Test Accuracy | CV Accuracy | Classes |
|---------|--------------|-------------|---------|
| Kepler  | 99.27%       | ~99%        | 3       |
| K2      | 87.64%       | 86.83%      | 4       |
| TESS    | 70.78%       | 70.22%      | 6       |

## Features

- **Multi-Mission Support**: Kepler, K2, and TESS datasets
- **Interactive Web Interface**: Upload CSV, select mission, view results
- **Real-Time Analysis**: Instant classification with confidence scores
- **Multiple Export Formats**: CSV, TXT reports, ZIP archives
- **Mission-Specific Models**: Optimized for each telescope's characteristics

## Tech Stack

**Backend:**
- FastAPI (REST API)
- XGBoost (Classification)
- scikit-learn (Preprocessing)
- pandas, NumPy (Data handling)
- Python 3.8+

**Frontend:**
- HTML5, CSS3, JavaScript
- Vanilla JS (no frameworks)
- Responsive design

## Setup
download the index.html, models and main.py 
Install the dependencies using requirements.txt 
or 
pip install fastapi uvicorn pandas numpy scikit-learn xgboost joblib python-multipart

## Ensure all model files are in the root directory:
exoplanet_xgboost_model.pkl
label_encoder.pkl
selected_features.pkl
k2_exoplanet_xgboost_model.pkl
k2_label_encoder.pkl
k2_selected_features.pkl
tess_exoplanet_xgboost_model.pkl
tess_label_encoder.pkl
tess_selected_features.pkl

## Start the Backend
python main.py
Open index.html in a web browser

Using the Interface

Select Mission: Choose Kepler, K2, or TESS
Upload CSV: Upload your dataset with required columns
View Results: See classification statistics and confidence scores
Download: Export full results, confirmed exoplanets, or summary reports

## Required CSV Columns
Kepler
koi_period, koi_duration, koi_depth, koi_prad, koi_teq, koi_insol, 
koi_sma, koi_incl, koi_impact, koi_dor, koi_ror, koi_srho, 
koi_model_snr, koi_num_transits, koi_count, koi_steff, koi_slogg, 
koi_srad, koi_smass, koi_fpflag_nt, koi_fpflag_ss, koi_fpflag_co, 
koi_fpflag_ec, koi_kepmag, koi_tce_plnt_num
K2
pl_orbper, pl_trandur, pl_trandep, pl_rade, pl_insol, pl_eqt, 
pl_tranmid, pl_orbincl, pl_imppar, pl_ratdor, pl_ratror, st_teff, 
st_logg, st_rad, st_mass, sy_kepmag, ra, dec, sy_pmra, sy_pmdec
TESS
pl_orbper, pl_trandurh, pl_trandep, pl_rade, pl_insol, pl_eqt, 
pl_tranmid, st_tmag, st_dist, st_teff, st_logg, st_rad, st_pmra, 
st_pmdec, ra, dec

## Data Sources
All datasets from NASA Exoplanet Archive:
- [Kepler Objects of Interest](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative)
- [K2 Targets and Candidates](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=k2pandc)
- [TESS Objects of Interest](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=TOI)

## model evaluation 
k2
<img width="4467" height="3569" alt="k2_model_evaluation" src="https://github.com/user-attachments/assets/1b850194-f19e-4bef-9cca-415588145cfc" />
kepler
<img width="4467" height="3569" alt="exoplanet_model_evaluation" src="https://github.com/user-attachments/assets/906c5dd1-4500-45a1-aafd-100b9a364daa" />
tess
<img width="4467" height="3568" alt="tess_model_evaluation" src="https://github.com/user-attachments/assets/6f1694f8-cb8b-4b03-861e-d363e7d2856f" />

## presentation 
https://www.canva.com/design/DAG0qzuIupA/nAE5I4Wyx6r3N_2kmSovGw/edit?utm_content=DAG0qzuIupA&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton
