from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
import io
import os
from datetime import datetime
from typing import Dict, List, Optional
import zipfile
from pathlib import Path

app = FastAPI(
    title="ExoScan AI API",
    description="Exoplanet Detection API using XGBoost",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store models for each mission
models = {}
label_encoders = {}
selected_features = {}

# Mission configurations
MISSIONS = {
    'kepler': {
        'model_file': 'exoplanet_xgboost_model.pkl',
        'encoder_file': 'label_encoder.pkl',
        'features_file': 'selected_features.pkl'
    },
    'k2': {
        'model_file': 'k2_xgboost_model.pkl',
        'encoder_file': 'k2_label_encoder.pkl',
        'features_file': 'k2_selected_features.pkl'
    },
    'tess': {
        'model_file': 'tess_xgboost_model.pkl',
        'encoder_file': 'tess_label_encoder.pkl',
        'features_file': 'tess_selected_features.pkl'
    }
}

# Load all available models
print("Loading XGBoost models...")
for mission, files in MISSIONS.items():
    try:
        models[mission] = joblib.load(files['model_file'])
        label_encoders[mission] = joblib.load(files['encoder_file'])
        selected_features[mission] = joblib.load(files['features_file'])
        print(f"✓ {mission.upper()} model loaded successfully!")
    except Exception as e:
        print(f"⚠ Error loading {mission.upper()} model: {e}")
        models[mission] = None

# Required features for prediction (Kepler default)
REQUIRED_FEATURES = [
    'koi_period', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_teq',
    'koi_insol', 'koi_sma', 'koi_incl', 'koi_impact', 'koi_dor',
    'koi_ror', 'koi_srho', 'koi_model_snr', 'koi_num_transits',
    'koi_count', 'koi_steff', 'koi_slogg', 'koi_srad', 'koi_smass',
    'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec',
    'koi_kepmag', 'koi_tce_plnt_num'
]

def preprocess_data(df: pd.DataFrame, mission: str = 'kepler') -> pd.DataFrame:
    """Preprocess uploaded data to match training format"""
    # Use mission-specific features if available
    required_features = selected_features.get(mission, REQUIRED_FEATURES)
    
    # Check which required features are present
    missing_features = [f for f in required_features if f not in df.columns]
    
    if missing_features:
        print(f"Missing features for {mission}: {missing_features}")
    
    # Keep only required features that exist
    available_features = [col for col in required_features if col in df.columns]
    df_processed = df[available_features].copy()
    
    # Add missing features with NaN (XGBoost can handle them)
    for feature in missing_features:
        df_processed[feature] = np.nan
    
    # Ensure correct order
    df_processed = df_processed[required_features]
    
    # Fill critical missing values
    for col in df_processed.columns:
        if df_processed[col].isnull().sum() > 0:
            if col in ['koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec']:
                df_processed[col].fillna(0, inplace=True)
            else:
                df_processed[col].fillna(df_processed[col].median(), inplace=True)
    
    return df_processed

def create_summary_report(results_df: pd.DataFrame, processing_time: float, mission: str) -> str:
    """Generate text summary report"""
    total = len(results_df)
    
    # Count each class (case-insensitive)
    predictions_upper = results_df['PREDICTION'].str.upper()
    confirmed = (predictions_upper == 'CONFIRMED').sum()
    candidate = (predictions_upper == 'CANDIDATE').sum()
    false_positive = (predictions_upper == 'FALSE POSITIVE').sum()
    
    # Find the confidence column for CONFIRMED (handle different formats)
    conf_cols = [col for col in results_df.columns if 'CONFIDENCE' in col and 'CONFIRMED' in col]
    conf_col = conf_cols[0] if conf_cols else None
    
    # Get high confidence discoveries (>90%)
    if conf_col and len(results_df[results_df[conf_col] > 0.90]) > 0:
        high_conf = results_df[results_df[conf_col] > 0.90].sort_values(
            conf_col, ascending=False
        ).head(10)
    else:
        high_conf = pd.DataFrame()
    
    report = f"""
╔════════════════════════════════════════════════════════════╗
║            EXOPLANET DETECTION SUMMARY REPORT              ║
║                      ExoScan AI v1.0                       ║
║                   Mission: {mission.upper()}                        ║
╚════════════════════════════════════════════════════════════╝

Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Processing Time: {processing_time:.2f} seconds
Model: XGBoost Classifier ({mission.upper()})

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 ANALYSIS RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Total KOIs Analyzed: {total}

Classification Breakdown:
  ✓ Confirmed Exoplanets:  {confirmed:4d} ({confirmed/total*100:5.1f}%)
  ⚠ Planetary Candidates:  {candidate:4d} ({candidate/total*100:5.1f}%)
  ✗ False Positives:       {false_positive:4d} ({false_positive/total*100:5.1f}%)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🌟 HIGH CONFIDENCE DISCOVERIES (>90% Confidence)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
    
    if len(high_conf) > 0:
        for idx, row in high_conf.iterrows():
            kepid = row.get('kepid', row.get('rowid', idx))
            conf = row[conf_col] if conf_col else 0
            report += f"  • KOI-{kepid}: {conf*100:.1f}% confidence\n"
    else:
        report += "  No high confidence discoveries in this dataset.\n"
    
    # Calculate average confidence scores dynamically
    conf_confirmed = results_df[conf_cols[0]].mean() if conf_cols else 0
    conf_candidate_cols = [col for col in results_df.columns if 'CONFIDENCE' in col and 'CANDIDATE' in col]
    conf_candidate = results_df[conf_candidate_cols[0]].mean() if conf_candidate_cols else 0
    conf_fp_cols = [col for col in results_df.columns if 'CONFIDENCE' in col and 'FALSE' in col]
    conf_fp = results_df[conf_fp_cols[0]].mean() if conf_fp_cols else 0
    
    # Get signal-to-noise range safely
    snr_min = results_df['koi_model_snr'].min() if 'koi_model_snr' in results_df.columns else 0
    snr_max = results_df['koi_model_snr'].max() if 'koi_model_snr' in results_df.columns else 0
    
    report += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📈 STATISTICAL SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Average Confidence Scores:
  Confirmed:      {conf_confirmed*100:.1f}%
  Candidate:      {conf_candidate*100:.1f}%
  False Positive: {conf_fp*100:.1f}%

Discovery Rate: {confirmed/total*100:.1f}%
Signal-to-Noise Range: {snr_min:.1f} - {snr_max:.1f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📁 OUTPUT FILES GENERATED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. full_results.csv          - Complete analysis with predictions
2. confirmed_exoplanets.csv  - Only confirmed exoplanets
3. summary_report.txt        - This report

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ℹ️  NOTES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• Confirmed: High confidence exoplanet detections
• Candidate: Potential exoplanets requiring follow-up observation
• False Positive: Likely stellar eclipses or instrumental artifacts

For questions or support, visit: https://exoplanetarchive.ipac.caltech.edu

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Generated by ExoScan AI - Powered by NASA Open Data
"""
    
    return report

@app.get("/")
async def root():
    """API health check"""
    available_missions = [m for m, model in models.items() if model is not None]
    return {
        "message": "ExoScan AI API",
        "status": "running",
        "available_missions": available_missions,
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    mission_status = {
        mission: {
            "loaded": models[mission] is not None,
            "classes": label_encoders[mission].classes_.tolist() if models[mission] else []
        }
        for mission in MISSIONS.keys()
    }
    
    return {
        "status": "healthy",
        "missions": mission_status
    }

@app.post("/predict")
async def predict_exoplanets(
    file: UploadFile = File(...),
    mission: str = Form(default='kepler')
):
    """
    Upload CSV file with KOI data and get predictions
    
    Parameters:
    - file: CSV file with KOI data
    - mission: Mission name (kepler, k2, or tess)
    
    Returns:
    - Summary statistics
    - Download links for results
    """
    # Validate mission
    mission = mission.lower()
    if mission not in MISSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid mission. Must be one of: {', '.join(MISSIONS.keys())}"
        )
    
    # Check if model is loaded
    if models[mission] is None:
        raise HTTPException(
            status_code=500, 
            detail=f"Model for {mission.upper()} is not loaded. Please ensure model files exist."
        )
    
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        start_time = datetime.now()
        
        # Read uploaded CSV
        contents = await file.read()
        print(f"File size: {len(contents)} bytes")
        print(f"Mission: {mission.upper()}")
        
        try:
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')), comment='#')
        except Exception as csv_error:
            print(f"CSV parsing error: {csv_error}")
            raise HTTPException(status_code=400, detail=f"CSV parsing failed: {str(csv_error)}")
        
        print(f"Received file: {file.filename}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()[:10]}")
        
        # Store original data
        df_original = df.copy()
        
        # Preprocess data for specific mission
        try:
            df_processed = preprocess_data(df, mission)
            print(f"Preprocessed shape: {df_processed.shape}")
        except Exception as prep_error:
            print(f"Preprocessing error: {prep_error}")
            raise HTTPException(status_code=400, detail=f"Data preprocessing failed: {str(prep_error)}")
        
        # Make predictions using mission-specific model
        try:
            predictions = models[mission].predict(df_processed)
            predictions_proba = models[mission].predict_proba(df_processed)
            print(f"Predictions generated: {len(predictions)}")
        except Exception as pred_error:
            print(f"Prediction error: {pred_error}")
            raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(pred_error)}")
        
        # Create results dataframe
        results_df = df_original.copy()
        results_df['PREDICTION'] = [label_encoders[mission].classes_[p] for p in predictions]
        
        # Add confidence scores for each class
        for i, class_name in enumerate(label_encoders[mission].classes_):
            clean_name = class_name.upper().replace(" ", "_")
            results_df[f'CONFIDENCE_{clean_name}'] = predictions_proba[:, i]
        
        print(f"Classes found: {label_encoders[mission].classes_}")
        print(f"Result columns: {results_df.columns.tolist()}")
        
        # Calculate statistics
        total = len(results_df)
        predictions_upper = results_df['PREDICTION'].str.upper()
        confirmed = (predictions_upper == 'CONFIRMED').sum()
        candidate = (predictions_upper == 'CANDIDATE').sum()
        false_positive = (predictions_upper == 'FALSE POSITIVE').sum()
        
        # Save results to files
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path('results')
        output_dir.mkdir(exist_ok=True)
        
        # 1. Full results CSV
        full_results_path = output_dir / f'{mission}_full_results_{timestamp}.csv'
        results_df.to_csv(full_results_path, index=False, encoding='utf-8')
        
        # 2. Confirmed exoplanets only CSV
        confirmed_df = results_df[predictions_upper == 'CONFIRMED'].copy()
        confirmed_path = output_dir / f'{mission}_confirmed_exoplanets_{timestamp}.csv'
        confirmed_df.to_csv(confirmed_path, index=False, encoding='utf-8')
        
        # 3. Summary report
        processing_time = (datetime.now() - start_time).total_seconds()
        summary_text = create_summary_report(results_df, processing_time, mission)
        summary_path = output_dir / f'{mission}_summary_report_{timestamp}.txt'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        
        # Create ZIP file with all results
        zip_path = output_dir / f'{mission}_exoscan_results_{timestamp}.zip'
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(full_results_path, full_results_path.name)
            zipf.write(confirmed_path, confirmed_path.name)
            zipf.write(summary_path, summary_path.name)
        
        # Get top discoveries for preview
        conf_cols = [col for col in confirmed_df.columns if 'CONFIDENCE' in col and 'CONFIRMED' in col]
        top_discoveries_list = []
        
        if len(confirmed_df) > 0 and conf_cols:
            if 'kepid' in confirmed_df.columns:
                top_discoveries_list = [
                    {
                        'kepid': int(row['kepid']) if pd.notna(row['kepid']) else 0,
                        'confidence': float(row[conf_cols[0]])
                    }
                    for _, row in confirmed_df.nlargest(5, conf_cols[0]).iterrows()
                ]
        
        return {
            "success": True,
            "mission": mission,
            "summary": {
                "total_analyzed": int(total),
                "confirmed": int(confirmed),
                "candidates": int(candidate),
                "false_positives": int(false_positive),
                "confirmed_percentage": round(float(confirmed) / float(total) * 100, 1),
                "processing_time": round(processing_time, 2)
            },
            "top_discoveries": top_discoveries_list,
            "download_links": {
                "full_results": f"/download/{full_results_path.name}",
                "confirmed_only": f"/download/{confirmed_path.name}",
                "summary_report": f"/download/{summary_path.name}",
                "zip_all": f"/download/{zip_path.name}"
            },
            "timestamp": timestamp
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download result files"""
    file_path = Path('results') / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/octet-stream'
    )

@app.get("/template/{mission}")
async def download_template(mission: str = 'kepler'):
    """Download a sample CSV template with required columns for specified mission"""
    mission = mission.lower()
    if mission not in MISSIONS:
        raise HTTPException(status_code=400, detail="Invalid mission")
    
    template_data = {
        'kepid': [10797460, 10811496],
        'koi_period': [9.488, 3.522],
        'koi_duration': [2.9575, 3.1234],
        'koi_depth': [615.8, 450.2],
        'koi_prad': [2.26, 1.85],
        'koi_teq': [793.0, 1050.0],
        'koi_insol': [93.59, 180.5],
        'koi_sma': [0.0853, 0.0512],
        'koi_incl': [89.66, 88.5],
        'koi_impact': [0.146, 0.234],
        'koi_dor': [24.81, 18.6],
        'koi_ror': [0.022344, 0.018],
        'koi_srho': [3.20796, 2.8],
        'koi_model_snr': [35.80, 28.5],
        'koi_num_transits': [142, 98],
        'koi_count': [2, 1],
        'koi_steff': [5455.0, 6100.0],
        'koi_slogg': [4.467, 4.2],
        'koi_srad': [0.927, 1.15],
        'koi_smass': [0.919, 1.08],
        'koi_fpflag_nt': [0, 0],
        'koi_fpflag_ss': [0, 0],
        'koi_fpflag_co': [0, 0],
        'koi_fpflag_ec': [0, 0],
        'koi_kepmag': [15.347, 14.2],
        'koi_tce_plnt_num': [1, 1]
    }
    
    template_df = pd.DataFrame(template_data)
    
    # Save to temporary file
    template_path = Path('results') / f'{mission}_koi_template.csv'
    template_path.parent.mkdir(exist_ok=True)
    template_df.to_csv(template_path, index=False)
    
    return FileResponse(
        path=template_path,
        filename=f'{mission}_koi_template.csv',
        media_type='text/csv'
    )

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("🚀 Starting ExoScan AI API Server")
    print("=" * 60)
    print("📡 Server will run at: http://localhost:8000")
    print("📚 API Docs at: http://localhost:8000/docs")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)
