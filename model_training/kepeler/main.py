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
print("EXOPLANET CLASSIFICATION WITH XGBOOST")
print("=" * 60)

# Load the dataset
df = pd.read_csv('cumulative_2025.09.30_07.53.58.csv', comment='#')
print(f"\n✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Check target variable distribution
print("\n" + "=" * 60)
print("TARGET VARIABLE DISTRIBUTION (koi_pdisposition)")
print("=" * 60)
print(df['koi_pdisposition'].value_counts())
print(f"\nClass percentages:")
print(df['koi_pdisposition'].value_counts(normalize=True) * 100)

# FEATURE SELECTION

print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

# Select the most important features for exoplanet detection
# These are based on domain knowledge and transit method physics
selected_features = [
    # Orbital characteristics
    'koi_period',           # Orbital period
    'koi_duration',         # Transit duration
    'koi_depth',            # Transit depth (very important!)
    'koi_prad',             # Planetary radius
    'koi_teq',              # Equilibrium temperature
    'koi_insol',            # Insolation flux
    'koi_sma',              # Semi-major axis
    'koi_incl',             # Inclination
    'koi_impact',           # Impact parameter
    'koi_dor',              # Planet-star distance ratio
    
    # Transit shape characteristics
    'koi_ror',              # Planet-star radius ratio
    'koi_srho',             # Fitted stellar density
    
    # Statistical measures
    'koi_model_snr',        # Signal-to-noise ratio (very important!)
    'koi_num_transits',     # Number of observed transits
    'koi_count',            # Number of planets in system
    
    # Stellar properties
    'koi_steff',            # Stellar temperature
    'koi_slogg',            # Stellar surface gravity
    'koi_srad',             # Stellar radius
    'koi_smass',            # Stellar mass
    
    # False positive flags (very important!)
    'koi_fpflag_nt',        # Not transit-like flag
    'koi_fpflag_ss',        # Stellar eclipse flag
    'koi_fpflag_co',        # Centroid offset flag
    'koi_fpflag_ec',        # Contamination flag
    
    # Additional useful features
    'koi_kepmag',           # Kepler magnitude
    'koi_tce_plnt_num',     # TCE planet number
]

# Create feature dataframe
X = df[selected_features].copy()
y = df['koi_pdisposition'].copy()

print(f"✓ Selected {len(selected_features)} features")
print(f"✓ Missing values per feature:")
print(X.isnull().sum().sort_values(ascending=False).head(10))

# 3. DATA PREPROCESSING

print("\n" + "=" * 60)
print("DATA PREPROCESSING")
print("=" * 60)

# Handle missing values
# For XGBoost, we can keep NaN as it handles them natively
# But let's fill some critical ones for better performance
print("Handling missing values...")

# Fill critical missing values with median
for col in X.columns:
    if X[col].isnull().sum() > 0:
        if col in ['koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec']:
            # False positive flags - fill with 0 (no flag)
            X[col].fillna(0, inplace=True)
        else:
            # Numerical features - fill with median
            X[col].fillna(X[col].median(), inplace=True)

print(f"✓ Missing values handled")
print(f"✓ Remaining missing values: {X.isnull().sum().sum()}")

# Encode target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"\n✓ Target encoding: {dict(enumerate(le.classes_))}")


# train test split
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
print(pd.Series(y_train).value_counts())


# training xgboost model

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

print("Model training completed!")

#Model evaluation
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

# feature importance

print("\n" + "=" * 60)
print("TOP 15 MOST IMPORTANT FEATURES")
print("=" * 60)

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(15).to_string(index=False))

#save model

print("\n" + "=" * 60)
print("SAVING MODEL")
print("=" * 60)

# Save the trained model
joblib.dump(model, 'exoplanet_xgboost_model.pkl')
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(selected_features, 'selected_features.pkl')

print("✓ Model saved as 'exoplanet_xgboost_model.pkl'")
print("✓ Label encoder saved as 'label_encoder.pkl'")
print("✓ Feature list saved as 'selected_features.pkl'")

print("\n" + "=" * 60)
print("CREATING VISUALIZATIONS")
print("=" * 60)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, 
            yticklabels=le.classes_,
            ax=axes[0, 0])
axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Actual')
axes[0, 0].set_xlabel('Predicted')

# 2. Feature Importance (Top 15)
top_features = feature_importance.head(15)
axes[0, 1].barh(top_features['feature'], top_features['importance'], color='skyblue')
axes[0, 1].set_xlabel('Importance')
axes[0, 1].set_title('Top 15 Feature Importances', fontsize=14, fontweight='bold')
axes[0, 1].invert_yaxis()

# 3. Class Distribution
class_counts = pd.Series(y_test).map(lambda x: le.classes_[x]).value_counts()
axes[1, 0].bar(class_counts.index, class_counts.values, color=['#FF6B6B', '#4ECDC4', '#95E1D3'])
axes[1, 0].set_title('Test Set Class Distribution', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('Count')
axes[1, 0].tick_params(axis='x', rotation=45)

# 4. Accuracy Comparison
accuracies = ['Train', 'Test', 'CV']
scores = [train_accuracy, test_accuracy, cv_scores.mean()]
colors = ['#6C5CE7', '#00B894', '#FDCB6E']
axes[1, 1].bar(accuracies, scores, color=colors)
axes[1, 1].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].set_ylim([0, 1])
for i, v in enumerate(scores):
    axes[1, 1].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('exoplanet_model_evaluation.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved as 'exoplanet_model_evaluation.png'")

# ============================================
# 10. EXAMPLE PREDICTION FUNCTION
# ============================================
print("\n" + "=" * 60)
print("EXAMPLE PREDICTION")
print("=" * 60)

def predict_exoplanet(sample_data):
    """
    Predict if a KOI is a confirmed exoplanet, candidate, or false positive
    
    Parameters:
    sample_data: dict with feature values
    
    Returns:
    prediction: string (CANDIDATE, CONFIRMED, or FALSE POSITIVE)
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

prediction, probabilities = predict_exoplanet(sample)

print(f"Sample KOI prediction:")
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
print(f"\n Generated Files:")
print(f"   • exoplanet_xgboost_model.pkl")
print(f"   • label_encoder.pkl")
print(f"   • selected_features.pkl")
print(f"   • exoplanet_model_evaluation.png")
print("=" * 60)
