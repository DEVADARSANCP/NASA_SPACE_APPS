import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("TESS EXOPLANET CLASSIFICATION WITH XGBOOST")
print("=" * 60)

# Load the TESS dataset
df = pd.read_csv('TOI_2025.09.30_07.53.33.csv', comment='#')
print(f"\n✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Check target variable distribution
print("\n" + "=" * 60)
print("TARGET VARIABLE DISTRIBUTION (tfopwg_disp)")
print("=" * 60)
print(df['tfopwg_disp'].value_counts())
print(f"\nClass percentages:")
print(df['tfopwg_disp'].value_counts(normalize=True) * 100)

# FEATURE SELECTION
print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

# Select features available in TESS TOI catalog
# Based on transit detection and stellar characterization
selected_features = [
    # Orbital characteristics
    'pl_orbper',           # Orbital period
    'pl_trandurh',         # Transit duration (hours)
    'pl_trandep',          # Transit depth (ppm) - very important!
    'pl_rade',             # Planet radius (Earth radii)
    'pl_insol',            # Insolation flux
    'pl_eqt',              # Equilibrium temperature
    
    # Transit timing
    'pl_tranmid',          # Transit midpoint
    
    # Stellar properties
    'st_tmag',             # TESS magnitude - very important!
    'st_dist',             # Stellar distance
    'st_teff',             # Stellar effective temperature
    'st_logg',             # Stellar surface gravity
    'st_rad',              # Stellar radius
    
    # Proper motion (can indicate false positives)
    'st_pmra',             # Proper motion RA
    'st_pmdec',            # Proper motion Dec
    
    # Coordinate information
    'ra',                  # Right ascension
    'dec',                 # Declination
]

# Create feature dataframe
X = df[selected_features].copy()
y = df['tfopwg_disp'].copy()

print(f"✓ Selected {len(selected_features)} features")
print(f"✓ Missing values per feature:")
print(X.isnull().sum().sort_values(ascending=False).head(10))

# DATA PREPROCESSING
print("\n" + "=" * 60)
print("DATA PREPROCESSING")
print("=" * 60)

# Handle missing values
print("Handling missing values...")

for col in X.columns:
    if X[col].isnull().sum() > 0:
        # Fill with median for numerical features
        X[col].fillna(X[col].median(), inplace=True)

print(f"✓ Missing values handled")
print(f"✓ Remaining missing values: {X.isnull().sum().sum()}")

# Filter out any rows with missing target
valid_mask = y.notna()
X = X[valid_mask]
y = y[valid_mask]

print(f"✓ Filtered dataset: {X.shape[0]} samples with valid targets")

# Encode target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"\n✓ Target encoding: {dict(enumerate(le.classes_))}")

# TRAIN TEST SPLIT
print("\n" + "=" * 60)
print("SPLITTING DATA")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_encoded
)

print(f"✓ Training set: {X_train.shape[0]} samples")
print(f"✓ Test set: {X_test.shape[0]} samples")
print(f"✓ Training class distribution:")
train_dist = pd.Series(y_train).map(lambda x: le.classes_[x]).value_counts()
print(train_dist)

# TRAINING XGBOOST MODEL
print("\n" + "=" * 60)
print("TRAINING XGBOOST MODEL")
print("=" * 60)

# Create XGBoost classifier with optimized parameters
model = xgb.XGBClassifier(
    n_estimators=300,           # Number of trees
    max_depth=8,                # Maximum tree depth
    learning_rate=0.1,          # Learning rate
    subsample=0.8,              # Subsample ratio
    colsample_bytree=0.8,       # Feature sampling ratio
    min_child_weight=3,         # Minimum sum of instance weight
    gamma=0.1,                  # Minimum loss reduction
    reg_alpha=0.1,              # L1 regularization
    reg_lambda=1.0,             # L2 regularization
    objective='multi:softmax',  # Multi-class classification
    num_class=len(le.classes_), # Number of classes
    random_state=42,
    eval_metric='mlogloss',     # Evaluation metric
    use_label_encoder=False,
    n_jobs=-1                   # Use all CPU cores
)

print("Training model... (this may take a minute)")
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False
)

print("✓ Model training completed!")

# MODEL EVALUATION
print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Calculate accuracies
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"\nACCURACY SCORES:")
print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Detailed classification report
print(f"\nDETAILED CLASSIFICATION REPORT:")
print("=" * 60)
print(classification_report(y_test, y_pred_test, target_names=le.classes_))

# Confusion matrix
print(f"\nCONFUSION MATRIX:")
print("=" * 60)
cm = confusion_matrix(y_test, y_pred_test)
print(cm)
print(f"\nRows = Actual, Columns = Predicted")
print(f"Classes: {le.classes_}")

# Cross-validation score
print(f"\nCROSS-VALIDATION (5-FOLD):")
print("=" * 60)
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
print(f"Individual fold scores: {cv_scores}")

# FEATURE IMPORTANCE
print("\n" + "=" * 60)
print("TOP 15 MOST IMPORTANT FEATURES")
print("=" * 60)

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(15).to_string(index=False))

# SAVE MODEL
print("\n" + "=" * 60)
print("SAVING MODEL")
print("=" * 60)

# Save the trained model
joblib.dump(model, 'tess_exoplanet_xgboost_model.pkl')
joblib.dump(le, 'tess_label_encoder.pkl')
joblib.dump(selected_features, 'tess_selected_features.pkl')

print("✓ Model saved as 'tess_exoplanet_xgboost_model.pkl'")
print("✓ Label encoder saved as 'tess_label_encoder.pkl'")
print("✓ Feature list saved as 'tess_selected_features.pkl'")

# VISUALIZATIONS
print("\n" + "=" * 60)
print("CREATING VISUALIZATIONS")
print("=" * 60)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=le.classes_, 
            yticklabels=le.classes_,
            ax=axes[0, 0])
axes[0, 0].set_title('Confusion Matrix (TESS)', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Actual')
axes[0, 0].set_xlabel('Predicted')

# 2. Feature Importance (Top 15)
top_features = feature_importance.head(15)
axes[0, 1].barh(top_features['feature'], top_features['importance'], color='teal')
axes[0, 1].set_xlabel('Importance')
axes[0, 1].set_title('Top 15 Feature Importances', fontsize=14, fontweight='bold')
axes[0, 1].invert_yaxis()

# 3. Class Distribution
class_counts = pd.Series(y_test).map(lambda x: le.classes_[x]).value_counts()
colors_palette = plt.cm.Set3(range(len(class_counts)))
axes[1, 0].bar(class_counts.index, class_counts.values, color=colors_palette)
axes[1, 0].set_title('Test Set Class Distribution', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('Count')
axes[1, 0].tick_params(axis='x', rotation=45)

# 4. Accuracy Comparison
accuracies = ['Train', 'Test', 'CV']
scores = [train_accuracy, test_accuracy, cv_scores.mean()]
colors = ['#2ECC71', '#3498DB', '#E74C3C']
axes[1, 1].bar(accuracies, scores, color=colors)
axes[1, 1].set_title('Model Performance Comparison (TESS)', fontsize=14, fontweight='bold')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].set_ylim([0, 1])
for i, v in enumerate(scores):
    axes[1, 1].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('tess_model_evaluation.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved as 'tess_model_evaluation.png'")

# EXAMPLE PREDICTION
print("\n" + "=" * 60)
print("EXAMPLE PREDICTION")
print("=" * 60)

def predict_toi(sample_data):
    """
    Predict if a TOI is CP (Confirmed Planet), FP (False Positive), KP (Known Planet), or PC (Planet Candidate)
    
    Parameters:
    sample_data: dict with feature values
    
    Returns:
    prediction: string (CP, FP, KP, or PC)
    probabilities: dict with probability for each class
    """
    # Create dataframe from input
    sample_df = pd.DataFrame([sample_data])
    
    # Make prediction
    pred = model.predict(sample_df)[0]
    pred_proba = model.predict_proba(sample_df)[0]
    
    # Get class name
    prediction = le.classes_[pred]
    
    # Get probabilities
    probabilities = {le.classes_[i]: float(prob) for i, prob in enumerate(pred_proba)}
    
    return prediction, probabilities

# Test with a sample from test set
sample_idx = 0
sample = X_test.iloc[sample_idx].to_dict()
actual = le.classes_[y_test[sample_idx]]

prediction, probabilities = predict_toi(sample)

print(f"Sample TOI prediction:")
print(f"  Actual: {actual}")
print(f"  Predicted: {prediction}")
print(f"  Probabilities:")
for cls, prob in probabilities.items():
    print(f"    {cls}: {prob:.4f} ({prob*100:.2f}%)")

# SUMMARY
print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print(f"\nFinal Model Performance:")
print(f"   • Test Accuracy: {test_accuracy*100:.2f}%")
print(f"   • Cross-Validation: {cv_scores.mean()*100:.2f}%")
print(f"\nGenerated Files:")
print(f"   • tess_exoplanet_xgboost_model.pkl")
print(f"   • tess_label_encoder.pkl")
print(f"   • tess_selected_features.pkl")
print(f"   • tess_model_evaluation.png")
print("=" * 60)
