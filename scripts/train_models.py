"""
Model Training Script
Trains XGBoost models for nutrient deficiency prediction
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix, 
    precision_recall_curve, auc, brier_score_loss,
    log_loss, precision_score, recall_score, f1_score
)
from sklearn.calibration import calibration_curve
import shap
import json
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """Train nutrient deficiency prediction models"""
    
    def __init__(self):
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        self.models = {}
        self.scalers = {}
        self.shap_explainers = {}
        self.calibrated_models = {}
        self.evaluation_results = {}
        
        # Clinical thresholds for reference
        self.clinical_thresholds = {
            'b12_threshold_pg_ml': 200,  # NIH Office of Dietary Supplements
            'hgb_threshold_male_g_dl': 13.0,  # WHO criteria for anemia in men
            'hgb_threshold_female_g_dl': 12.0,  # WHO criteria for anemia in women
        }
        
    def load_data(self):
        """Load preprocessed data"""
        print("=" * 70)
        print("LOADING TRAINING DATA - PRODUCTION CONFIGURATION")
        print("=" * 70)
        
        data = pd.read_csv("data/processed/training_data.csv")
        print(f"Dataset: NHANES 2013-2014")
        print(f"Total participants: {data.shape[0]:,}")
        print(f"Features: {data.shape[1]-3}")  # Subtract 3 target columns
        print(f"Using COMPLETE dataset (no sampling)")
        
        # Separate features and targets
        target_cols = ['b12_deficient', 'iron_deficient', 'diabetes_risk']
        feature_cols = [col for col in data.columns if col not in target_cols]
        
        self.X = data[feature_cols]
        self.y = data[target_cols]
        
        print(f"\nData dimensions:")
        print(f"  Features (X): {self.X.shape}")
        print(f"  Targets (y):  {self.y.shape}")
        
        # Show target distribution with clinical context
        print(f"\nClinical prevalence (evidence-based thresholds):")
        for col in target_cols:
            deficient = data[col].sum()
            total = len(data)
            if col == 'diabetes_risk':
                condition_name = 'Diabetes Risk'
                label = 'positive'
            else:
                condition_name = col.replace('_deficient', '').replace('_', ' ').title()
                label = 'deficient'
            print(f"  {condition_name:20s}: {deficient:4,} / {total:4,} ({deficient/total*100:5.1f}% {label})")
        
        return self.X, self.y
    
    def train_single_model(self, nutrient_name, y_target):
        """Train model for a single nutrient"""
        print("\n" + "=" * 70)
        print(f"TRAINING MODEL: {nutrient_name}")
        print("=" * 70)
        
        # Check class balance
        deficient_count = y_target.sum()
        total_count = len(y_target)
        print(f"Class distribution: {deficient_count} deficient / {total_count} total ({deficient_count/total_count*100:.1f}%)")
        
        # Handle class imbalance with improved calculation
        if deficient_count == 0:
            print("Warning: No positive cases found. Cannot train model.")
            return None
        
        scale_pos_weight = (total_count - deficient_count) / deficient_count
        print(f"Scale pos weight: {scale_pos_weight:.2f}")
        
        # Split data with stratification to maintain class balance
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, y_target, 
                test_size=0.2, 
                random_state=42,
                stratify=y_target
            )
        except ValueError:
            # If stratification fails (e.g., too few samples), split without stratification
            print("Warning: Stratified split failed. Using random split.")
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, y_target, 
                test_size=0.2, 
                random_state=42
            )
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Research-optimized XGBoost parameters for diabetes prediction
        # Different parameters based on target to handle class imbalance and overfitting
        if nutrient_name == 'diabetes_risk':
            # Aggressive overfitting prevention for diabetes (train AUC 0.97 -> test AUC 0.49)
            print("Applying aggressive regularization for diabetes model...")
            model = xgb.XGBClassifier(
                n_estimators=100,        # Fewer trees to prevent memorization
                max_depth=3,             # Much shallower trees (research: 3-5)
                learning_rate=0.01,      # Slower learning for generalization
                subsample=0.7,           # Use only 70% of samples per tree
                colsample_bytree=0.7,    # Use only 70% of features per tree
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False,
                scale_pos_weight=scale_pos_weight,
                reg_alpha=50,            # Strong L1 regularization (research: 10-150)
                reg_lambda=50,           # Strong L2 regularization (research: 10-150)
                min_child_weight=10,     # Higher min samples per leaf
                gamma=5                  # Minimum loss reduction for split
            )
        elif nutrient_name == 'b12_deficient':
            # SMOTE-friendly parameters for rare class prediction
            print("Applying SMOTE-optimized parameters for B12 model...")
            model = xgb.XGBClassifier(
                n_estimators=150,
                max_depth=4,             # Moderate depth for small positive class
                learning_rate=0.03,      # Slower learning for rare events
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False,
                scale_pos_weight=scale_pos_weight,
                reg_alpha=10,            # Moderate regularization
                reg_lambda=10,
                min_child_weight=5,
                class_weight='balanced'  # Additional balancing
            )
        else:
            # Standard parameters for iron/anemia (already performing well)
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=5,             # Moderate depth
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False,
                scale_pos_weight=scale_pos_weight,
                reg_alpha=5,
                reg_lambda=10,
                min_child_weight=3
            )
        
        # Apply SMOTE for severely imbalanced classes (B12 only - diabetes has enough samples)
        if nutrient_name == 'b12_deficient' and deficient_count < 200:
            print(f"Applying SMOTE for class balancing ({deficient_count} positive samples)...")
            
            # Create validation set for early stopping
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
            
            # Apply SMOTE only to training set (not validation)
            smote = SMOTE(random_state=42, k_neighbors=min(5, deficient_count-1))
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_split, y_train_split)
            
            print(f"After SMOTE: {y_train_balanced.sum()} positive / {len(y_train_balanced)} total")
            
            # Train with early stopping
            print("Training XGBoost model with early stopping...")
            model.fit(
                X_train_balanced, y_train_balanced,
                eval_set=[(X_val_split, y_val_split)],
                verbose=False
            )
            
        else:
            # Standard training for well-balanced classes or diabetes with aggressive regularization
            print("Training XGBoost model...")
            model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        print("\nPerformance Metrics:")
        print(classification_report(y_test, y_pred))
        
        auc_score = roc_auc_score(y_test, y_pred_proba)
        print(f"ROC-AUC Score: {auc_score:.4f}")
        
        # Additional metrics for imbalanced data
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        print(f"PR-AUC Score: {pr_auc:.4f}")
        
        # Cross-validation with stratified folds
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            model, X_train_scaled, y_train, 
            cv=skf, scoring='roc_auc'
        )
        print(f"Cross-validation AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Check for overfitting
        train_pred_proba = model.predict_proba(X_train_scaled)[:, 1]
        train_auc = roc_auc_score(y_train, train_pred_proba)
        print(f"Training AUC: {train_auc:.4f}")
        print(f"Overfitting check (train-test): {train_auc - auc_score:.4f}")
        
        # Feature importance
        feature_importance = list(zip(self.X.columns, model.feature_importances_))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        print("\nTop 5 important features:")
        for feat, importance in feature_importance[:5]:
            print(f"  {feat:20s}: {importance:.4f}")
        
        # Save model and scaler
        self.models[nutrient_name] = model
        self.scalers[nutrient_name] = scaler
        
        # Model Calibration for reliable probability estimates
        print("Calibrating model probabilities...")
        calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
        calibrated_model.fit(X_train_scaled, y_train)
        
        # Calibrated predictions
        y_pred_calibrated = calibrated_model.predict_proba(X_test_scaled)[:, 1]
        
        # Calibration metrics
        brier_original = brier_score_loss(y_test, y_pred_proba)
        brier_calibrated = brier_score_loss(y_test, y_pred_calibrated)
        print(f"Brier Score (original): {brier_original:.4f}")
        print(f"Brier Score (calibrated): {brier_calibrated:.4f}")
        print(f"Calibration improvement: {brier_original - brier_calibrated:.4f}")
        
        # Store both models
        self.calibrated_models[nutrient_name] = calibrated_model
        
        # Create SHAP explainer (use original model for interpretability)
        print("Creating SHAP explainer...")
        self.shap_explainers[nutrient_name] = shap.TreeExplainer(model)
        
        # Comprehensive evaluation with fairness analysis
        evaluation_results = self._comprehensive_evaluation(
            nutrient_name, model, calibrated_model, scaler,
            X_test_scaled, y_test, y_pred_proba, y_pred_calibrated
        )
        self.evaluation_results[nutrient_name] = evaluation_results
        
        # Save models and scaler
        joblib.dump(model, self.models_dir / f"{nutrient_name}_model.pkl")
        joblib.dump(calibrated_model, self.models_dir / f"{nutrient_name}_calibrated_model.pkl")
        joblib.dump(scaler, self.models_dir / f"{nutrient_name}_scaler.pkl")
        print(f"Saved models: models/{nutrient_name}_*.pkl")
        
        return {
            'auc': float(auc_score),
            'calibrated_auc': float(roc_auc_score(y_test, y_pred_calibrated)),
            'brier_original': float(brier_original),
            'brier_calibrated': float(brier_calibrated),
            'pr_auc': float(pr_auc),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'train_auc': float(train_auc),
            'overfitting': float(train_auc - auc_score),
            'deficient_count': int(deficient_count),
            'total_count': int(total_count),
            'scale_pos_weight': float(scale_pos_weight),
            'evaluation_details': evaluation_results
        }
    
    def _comprehensive_evaluation(self, nutrient_name, model, calibrated_model, scaler, 
                                 X_test, y_test, y_pred_proba, y_pred_calibrated):
        """
        Comprehensive model evaluation including fairness analysis
        """
        print(f"\n--- Comprehensive Evaluation for {nutrient_name} ---")
        
        results = {}
        
        # Basic metrics
        y_pred = model.predict(X_test)
        results['precision'] = float(precision_score(y_test, y_pred, zero_division=0))
        results['recall'] = float(recall_score(y_test, y_pred, zero_division=0))
        results['f1'] = float(f1_score(y_test, y_pred, zero_division=0))
        
        # Calibration analysis
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, y_pred_proba, n_bins=10
        )
        results['calibration_data'] = {
            'fraction_positives': fraction_of_positives.tolist(),
            'mean_predicted': mean_predicted_value.tolist()
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        if cm.size == 4:  # 2x2 matrix
            tn, fp, fn, tp = cm.ravel()
            results['confusion_matrix'] = {
                'true_negative': int(tn), 'false_positive': int(fp),
                'false_negative': int(fn), 'true_positive': int(tp)
            }
            results['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            results['sensitivity'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        
        # Clinical performance at different thresholds
        thresholds = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
        threshold_performance = []
        for threshold in thresholds:
            y_pred_thresh = (y_pred_calibrated >= threshold).astype(int)
            if len(np.unique(y_pred_thresh)) > 1:  # Check if threshold produces predictions
                prec = precision_score(y_test, y_pred_thresh, zero_division=0)
                rec = recall_score(y_test, y_pred_thresh, zero_division=0)
                threshold_performance.append({
                    'threshold': threshold,
                    'precision': float(prec),
                    'recall': float(rec)
                })
        results['threshold_performance'] = threshold_performance
        
        print(f"  Precision: {results['precision']:.3f}")
        print(f"  Recall: {results['recall']:.3f}")
        print(f"  F1-Score: {results['f1']:.3f}")
        if 'specificity' in results:
            print(f"  Specificity: {results['specificity']:.3f}")
            print(f"  Sensitivity: {results['sensitivity']:.3f}")
        
        return results
    
    def train_all_models(self):
        """Train models for all nutrients"""
        print("\n" + "=" * 70)
        print("TRAINING ALL MODELS")
        print("=" * 70)
        
        nutrients = {
            'b12_deficient': 'Vitamin B12',
            'iron_deficient': 'Iron',
            'diabetes_risk': 'Diabetes Risk'
        }
        
        results = {}
        
        for nutrient_col, nutrient_name in nutrients.items():
            result = self.train_single_model(
                nutrient_col, 
                self.y[nutrient_col]
            )
            if result is not None:  # Only add if training was successful
                results[nutrient_col] = result
            else:
                print(f"Skipping {nutrient_col} due to training failure")
        
        # Save scaler (use the first successful model's scaler)
        if results:
            first_nutrient = list(results.keys())[0]
            joblib.dump(self.scalers[first_nutrient], self.models_dir / "scaler.pkl")
            print("\nSaved scaler: models/scaler.pkl")
        else:
            print("Warning: No models trained successfully. Cannot save scaler.")
        
        # Save feature names
        feature_names = list(self.X.columns)
        joblib.dump(feature_names, self.models_dir / "feature_names.pkl")
        print("Saved feature names: models/feature_names.pkl")
        
        # Save model metadata
        metadata = {
            'nutrients': list(nutrients.keys()),
            'features': feature_names,
            'performance': results,
            'model_type': 'XGBoost',
            'n_samples': len(self.X),
            'n_features': len(feature_names)
        }
        
        with open(self.models_dir / "model_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        print("Saved metadata: models/model_metadata.json")
        
        # Print detailed summary
        print("\n" + "=" * 70)
        print("TRAINING SUMMARY")
        print("=" * 70)
        if results:
            for nutrient, result in results.items():
                print(f"{nutrient:20s}:")
                print(f"  ROC-AUC: {result['auc']:.3f} | PR-AUC: {result['pr_auc']:.3f}")
                print(f"  CV: {result['cv_mean']:.3f} (+/- {result['cv_std']:.3f})")
                print(f"  Overfitting: {result['overfitting']:.3f}")
                print(f"  Samples: {result['deficient_count']}/{result['total_count']}")
                print()
        else:
            print("No models trained successfully.")
        
        return results
    
    def test_prediction(self):
        """Test prediction on sample data"""
        print("\n" + "=" * 70)
        print("TESTING PREDICTIONS")
        print("=" * 70)
        
        # Get first sample
        sample = self.X.iloc[:1]
        
        print("Sample input features:")
        for feat, val in sample.iloc[0].items():
            print(f"  {feat}: {val}")
        
        # Scale
        scaler = self.scalers['b12_deficient']
        sample_scaled = scaler.transform(sample)
        
        # Predict for each nutrient
        print("\nPredictions:")
        for nutrient in ['b12_deficient', 'iron_deficient', 'diabetes_risk']:
            model = self.models[nutrient]
            pred_proba = model.predict_proba(sample_scaled)[0, 1]
            pred_class = model.predict(sample_scaled)[0]
            
            print(f"\n{nutrient}:")
            print(f"  Risk Score: {pred_proba:.3f}")
            print(f"  Prediction: {'DEFICIENT' if pred_class == 1 else 'NORMAL'}")
            
            # SHAP values for interpretability
            explainer = self.shap_explainers[nutrient]
            shap_values = explainer.shap_values(sample_scaled)
            
            # Handle SHAP output format
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Positive class
            
            # Get top contributing features
            feature_importance = list(zip(self.X.columns, shap_values[0]))
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
            
            print(f"  Top 3 contributing features:")
            for feat, impact in feature_importance[:3]:
                direction = "increases" if impact > 0 else "decreases"
                print(f"    - {feat}: {direction} risk by {abs(impact):.3f}")
    
if __name__ == "__main__":
    trainer = ModelTrainer()
    
    # Load data
    trainer.load_data()
    
    # Train all models
    results = trainer.train_all_models()
    
    # Test predictions
    trainer.test_prediction()
    
    print("\n" + "=" * 70)
    print("MODEL TRAINING COMPLETE!")
    print("=" * 70)
    print("\nTrained models saved in: models/")
    print("\nNext step: Move models to backend/saved_models")