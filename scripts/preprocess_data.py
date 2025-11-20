"""
NHANES Data Preprocessing Pipeline
Prepares training data for nutrient deficiency prediction
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

class NHANESPreprocessor:
    """Preprocess Kaggle NHANES dataset"""
    
    def __init__(self):
        self.data_dir = Path("data/raw/kaggle")
        self.output_dir = Path("data/processed")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """Load all CSV files"""
        print("=" * 70)
        print("LOADING DATA")
        print("=" * 70)
        
        self.demo = pd.read_csv(self.data_dir / "demographic.csv")
        self.diet = pd.read_csv(self.data_dir / "diet.csv")
        self.exam = pd.read_csv(self.data_dir / "examination.csv")
        self.labs = pd.read_csv(self.data_dir / "labs.csv")
        self.quest = pd.read_csv(self.data_dir / "questionnaire.csv")
        
        print(f"Demographics: {self.demo.shape}")
        print(f"Diet:         {self.diet.shape}")
        print(f"Examination:  {self.exam.shape}")
        print(f"Labs:         {self.labs.shape}")
        print(f"Questionnaire:{self.quest.shape}")
        
    def explore_labs(self):
        """Find nutrient columns in labs.csv"""
        print("\n" + "=" * 70)
        print("EXPLORING LAB COLUMNS")
        print("=" * 70)
        
        # Show all columns
        print(f"\nTotal columns in labs: {len(self.labs.columns)}")
        
        # Search for nutrient-related columns
        keywords = ['b12', 'vitamin', 'iron', 'hemoglobin', 'ferritin', 'folate']
        
        print("\nNutrient-related columns:")
        found_cols = {}
        
        for col in self.labs.columns:
            col_lower = col.lower()
            for keyword in keywords:
                if keyword in col_lower:
                    non_null = self.labs[col].notna().sum()
                    if non_null > 100:  # Only show if enough data
                        found_cols[col] = {
                            'keyword': keyword,
                            'count': non_null,
                            'mean': self.labs[col].mean(),
                            'min': self.labs[col].min(),
                            'max': self.labs[col].max()
                        }
                        print(f"\n  {col}")
                        print(f"    Non-null: {non_null:,}")
                        print(f"    Mean: {self.labs[col].mean():.2f}")
                        print(f"    Range: [{self.labs[col].min():.2f}, {self.labs[col].max():.2f}]")
                    break
        
        return found_cols
    
    def create_targets(self):
        """
        Create target variables for nutrient deficiencies using evidence-based clinical thresholds
        
        Clinical thresholds based on:
        - B12: NIH Office of Dietary Supplements (<200 pg/mL indicates deficiency)
        - Iron/Anemia: WHO criteria (Hb <12.0 g/dL women, <13.0 g/dL men)
        - Diabetes: Self-reported diagnosis or prediabetes (NHANES DIQ010)
        
        Note: This predicts anemia risk (via hemoglobin), not iron deficiency specifically.
        True iron deficiency requires ferritin + transferrin saturation assessment.
        """
        print("\n" + "=" * 70)
        print("CREATING TARGET VARIABLES WITH EVIDENCE-BASED THRESHOLDS")
        print("=" * 70)
        
        # NHANES 2013-2014 column names
        b12_col = 'LBDB12'        # Vitamin B12 (pg/mL)
        hgb_col = 'LBXHGB'        # Hemoglobin (g/dL)
        diabetes_col = 'DIQ010'   # Ever told you have diabetes (1=yes, 2=no, 3=prediabetes)
        iron_sat_col = 'LBXSATSI' # Transferrin saturation (%)
        gender_col = 'RIAGENDR'   # Gender (1=male, 2=female)
        
        # Clinical thresholds with citations
        self.clinical_thresholds = {
            'b12_threshold_pg_ml': 200,  # NIH Office of Dietary Supplements
            'hgb_threshold_male_g_dl': 13.0,  # WHO criteria for anemia in men
            'hgb_threshold_female_g_dl': 12.0,  # WHO criteria for anemia in women  
            'transferrin_sat_threshold_pct': 15  # Iron deficiency threshold
        }
        
        print(f"Clinical thresholds used:")
        print(f"  B12 deficiency: <{self.clinical_thresholds['b12_threshold_pg_ml']} pg/mL (NIH)")
        print(f"  Anemia (male): <{self.clinical_thresholds['hgb_threshold_male_g_dl']} g/dL (WHO)")
        print(f"  Anemia (female): <{self.clinical_thresholds['hgb_threshold_female_g_dl']} g/dL (WHO)")
        print(f"  Diabetes: Self-reported diagnosis or prediabetes (NHANES DIQ010)")
        print(f"  Note: Predicting anemia risk, not iron deficiency specifically")
        
        # Verify all columns exist
        missing_cols = []
        for col_name, col in [('B12', b12_col), ('Hemoglobin', hgb_col), ('Iron Sat', iron_sat_col)]:
            if col not in self.labs.columns:
                missing_cols.append(f"{col_name} ({col})")
        
        # Check diabetes column in questionnaire
        if diabetes_col not in self.quest.columns:
            missing_cols.append(f"Diabetes ({diabetes_col})")
        
        if missing_cols:
            raise ValueError(f"Missing critical columns: {', '.join(missing_cols)}")
        
        # Merge gender data for iron deficiency analysis
        labs_with_gender = self.labs.merge(
            self.demo[['SEQN', gender_col]], 
            on='SEQN', 
            how='left'
        )
        
        # Initialize targets DataFrame
        targets = pd.DataFrame()
        targets['SEQN'] = self.labs['SEQN']
        
        # B12 deficiency using NIH threshold
        b12_values = self.labs[b12_col]
        targets['b12_deficient'] = (b12_values < self.clinical_thresholds['b12_threshold_pg_ml']).astype(int)
        valid_b12 = b12_values.notna().sum()
        print(f"B12 deficiency: {b12_col}")
        print(f"  Valid measurements: {valid_b12:,}")
        print(f"  Deficient (<{self.clinical_thresholds['b12_threshold_pg_ml']} pg/mL): {targets['b12_deficient'].sum()} / {valid_b12} ({(targets['b12_deficient'].sum()/valid_b12)*100:.1f}%)")
        print(f"  Range: [{b12_values.min():.1f} - {b12_values.max():.1f}] pg/mL")
        
        # Anemia risk using WHO hemoglobin criteria (NOT iron deficiency specifically)
        hgb_values = labs_with_gender[hgb_col]
        sat_values = labs_with_gender[iron_sat_col]
        gender_values = labs_with_gender[gender_col]
        
        # Anemia criteria using WHO thresholds
        anemia_risk = np.where(
            gender_values == 1,  # Male
            hgb_values < self.clinical_thresholds['hgb_threshold_male_g_dl'],   
            hgb_values < self.clinical_thresholds['hgb_threshold_female_g_dl']    
        )
        
        # Note: We keep the original column name for compatibility but clarify it predicts anemia
        targets['iron_deficient'] = anemia_risk.astype(int)
        
        valid_iron = hgb_values.notna().sum()
        print(f"Anemia risk: {hgb_col}")
        print(f"  Valid measurements: {valid_iron:,}")
        print(f"  Anemic (WHO criteria): {targets['iron_deficient'].sum()} / {valid_iron} ({(targets['iron_deficient'].sum()/valid_iron)*100:.1f}%)")
        print(f"  Hemoglobin range: [{hgb_values.min():.1f} - {hgb_values.max():.1f}] g/dL")
        print(f"  Note: Predicts anemia risk, not iron deficiency specifically")
        
        # Diabetes risk using self-reported diagnosis (NHANES DIQ010)
        diabetes_values = self.quest[diabetes_col]
        
        # Create diabetes risk target (includes diabetes AND prediabetes)
        # 1 = Yes (diabetes), 3 = Prediabetes → Both count as positive
        # 2 = No → Negative
        targets['diabetes_risk'] = ((diabetes_values == 1) | (diabetes_values == 3)).astype(int)
        
        valid_diabetes = diabetes_values.notna().sum()
        diabetes_count = (diabetes_values == 1).sum()
        prediabetes_count = (diabetes_values == 3).sum()
        total_risk_count = targets['diabetes_risk'].sum()
        
        print(f"Diabetes risk: {diabetes_col}")
        print(f"  Valid responses: {valid_diabetes:,}")
        print(f"  Diabetes (code 1): {diabetes_count:,} ({diabetes_count/valid_diabetes*100:.1f}%)")
        print(f"  Prediabetes (code 3): {prediabetes_count:,} ({prediabetes_count/valid_diabetes*100:.1f}%)")
        print(f"  Total risk (diabetes + prediabetes): {total_risk_count:,} / {valid_diabetes:,} ({total_risk_count/valid_diabetes*100:.1f}%)")
        print(f"  Note: Predicts combined diabetes and prediabetes risk")
        
        # Remove rows where all targets are missing
        targets = targets.dropna(subset=['b12_deficient', 'iron_deficient', 'diabetes_risk'], how='all')
        
        print(f"\nTARGET SUMMARY:")
        print(f"  Total samples: {len(targets):,}")
        for col in ['b12_deficient', 'iron_deficient', 'diabetes_risk']:
            valid_count = targets[col].notna().sum()
            deficient_count = targets[col].sum()
            if valid_count > 0:
                label = 'positive' if col == 'diabetes_risk' else 'deficient'
                print(f"  {col:20s}: {deficient_count:4,} / {valid_count:4,} ({deficient_count/valid_count*100:5.1f}% {label})")
        
        self.targets = targets
        return targets
    
    def _find_column(self, df, patterns):
        """Find column matching any of the patterns"""
        for pattern in patterns:
            for col in df.columns:
                if pattern.lower() in col.lower():
                    return col
        return None
    
    def merge_datasets(self):
        """Merge all datasets"""
        print("\n" + "=" * 70)
        print("MERGING DATASETS")
        print("=" * 70)
        
        # Start with demographics
        merged = self.demo.copy()
        print(f"Starting: {merged.shape}")
        
        # Merge examination
        merged = merged.merge(self.exam, on='SEQN', how='inner', suffixes=('', '_exam'))
        print(f"+ Examination: {merged.shape}")
        
        # Merge diet
        merged = merged.merge(self.diet, on='SEQN', how='inner', suffixes=('', '_diet'))
        print(f"+ Diet: {merged.shape}")
        
        # Merge questionnaire
        merged = merged.merge(self.quest, on='SEQN', how='inner', suffixes=('', '_quest'))
        print(f"+ Questionnaire: {merged.shape}")
        
        # Merge targets
        merged = merged.merge(self.targets, on='SEQN', how='inner')
        print(f"+ Targets: {merged.shape}")
        
        # Remove rows with missing targets
        merged = merged.dropna(subset=['b12_deficient', 'iron_deficient', 'diabetes_risk'])
        print(f"After removing missing targets: {merged.shape}")
        
        self.merged = merged
        return merged
    
    def select_features(self):
        """Select relevant features"""
        print("\n" + "=" * 70)
        print("SELECTING FEATURES")
        print("=" * 70)
        
        # Define potential feature columns
        feature_patterns = {
            'age': ['RIDAGEYR', 'age', 'AGE'],
            'gender': ['RIAGENDR', 'gender', 'sex', 'GENDER'],
            'race': ['RIDRETH3', 'race', 'ethnicity'],
            'weight': ['BMXWT', 'weight', 'WEIGHT'],
            'height': ['BMXHT', 'height', 'HEIGHT'],
            'bmi': ['BMXBMI', 'bmi', 'BMI'],
        }
        
        selected_features = []
        
        # Find features
        for feat_name, patterns in feature_patterns.items():
            col = self._find_column(self.merged, patterns)
            if col and col in self.merged.columns:
                selected_features.append(col)
                print(f"{feat_name:15s}: {col}")
        
        # Add any numeric columns that might be useful
        numeric_cols = self.merged.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in selected_features and col not in ['SEQN', 'b12_deficient', 'iron_deficient', 'vitd_deficient']:
                # Only add if not too many missing values
                if self.merged[col].notna().sum() > len(self.merged) * 0.5:
                    selected_features.append(col)
        
        # Limit to reasonable number
        selected_features = selected_features[:20]
        
        print(f"\nTotal features selected: {len(selected_features)}")
        
        self.feature_cols = selected_features
        return selected_features
    
    def prepare_training_data(self, use_full_dataset=True, sample_size=None):
        """
        Prepare final training dataset
        
        Args:
            use_full_dataset (bool): Use complete NHANES dataset for production-quality models
            sample_size (int, optional): Sample size for development/testing. Ignored if use_full_dataset=True
        """
        print("\n" + "=" * 70)
        print("PREPARING TRAINING DATA")
        print("=" * 70)
        
        # Get features and targets
        X = self.merged[self.feature_cols].copy()
        y = self.merged[['b12_deficient', 'iron_deficient', 'diabetes_risk']].copy()
        
        print(f"Initial dataset size: {X.shape[0]:,} participants")
        
        # Handle missing values in features
        print("\nHandling missing values...")
        for col in X.columns:
            if X[col].dtype in [np.float64, np.int64]:
                X[col].fillna(X[col].median(), inplace=True)
            else:
                X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown', inplace=True)
        
        # Apply sampling strategy based on configuration
        if use_full_dataset:
            print(f"Using FULL NHANES dataset ({X.shape[0]:,} participants) for production-quality models")
        elif sample_size and len(X) > sample_size:
            print(f"WARNING: Sampling to {sample_size:,} rows for development/testing")
            indices = np.random.choice(len(X), sample_size, replace=False)
            X = X.iloc[indices]
            y = y.iloc[indices]
        else:
            print(f"Using complete available dataset ({X.shape[0]:,} participants)")
        
        print(f"\nFinal dataset:")
        print(f"  Features (X): {X.shape}")
        print(f"  Targets (y):  {y.shape}")
        
        print(f"\nTarget distribution:")
        for col in y.columns:
            print(f"  {col:20s}: {y[col].sum():,} / {len(y):,} ({y[col].mean()*100:.1f}% deficient)")
        
        # Save
        output_file = self.output_dir / "training_data.csv"
        combined = pd.concat([X, y], axis=1)
        combined.to_csv(output_file, index=False)
        print(f"\nSaved to: {output_file}")
        
        # Save feature names
        feature_file = self.output_dir / "feature_names.json"
        with open(feature_file, 'w') as f:
            json.dump(self.feature_cols, f, indent=2)
        print(f"Saved feature names to: {feature_file}")
        
        return X, y
    
    def run_full_pipeline(self):
        """Run complete preprocessing pipeline"""
        print("\n" + "=" * 70)
        print("NHANES DATA PREPROCESSING PIPELINE")
        print("=" * 70 + "\n")
        
        self.load_data()
        self.explore_labs()
        self.create_targets()
        self.merge_datasets()
        self.select_features()
        X, y = self.prepare_training_data()
        
        print("\n" + "=" * 70)
        print("PREPROCESSING COMPLETE!")
        print("=" * 70)
        print(f"\nNext step: Train models using data/processed/training_data.csv")
        
        return X, y


if __name__ == "__main__":
    preprocessor = NHANESPreprocessor()
    X, y = preprocessor.run_full_pipeline()