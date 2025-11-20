# Nutrient Deficiency Predictor - Model Development

Machine learning model development pipeline for predicting nutrient deficiencies using NHANES 2013-2014 dataset. This repository contains the complete data preprocessing, model training, and evaluation pipeline for developing XGBoost models that power the nutrient deficiency prediction system.

## Overview

This repository implements a comprehensive machine learning pipeline that transforms raw NHANES (National Health and Nutrition Examination Survey) data into production-ready models for predicting B12 deficiency, anemia risk, and diabetes risk. The models use evidence-based clinical thresholds and advanced techniques like SMOTE for class imbalance and isotonic regression for probability calibration.

## Features

- **NHANES Data Processing**: Complete pipeline for cleaning and preprocessing NHANES 2013-2014 survey data
- **Evidence-Based Targets**: Clinical thresholds from NIH and WHO guidelines for deficiency classification
- **Advanced ML Pipeline**: XGBoost models with SMOTE balancing and isotonic calibration
- **Model Interpretability**: SHAP (SHapley Additive exPlanations) integration for feature importance
- **Comprehensive Evaluation**: ROC-AUC, PR-AUC, Brier score, and calibration analysis
- **Production Ready**: Serialized models and scalers compatible with FastAPI backend

## Technology Stack

- **Data Processing**: Pandas, NumPy
- **Machine Learning**: XGBoost, Scikit-learn, Imbalanced-learn
- **Model Interpretability**: SHAP
- **Evaluation**: Scikit-learn metrics, Matplotlib, Seaborn
- **Serialization**: Joblib for model persistence

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- NHANES 2013-2014 dataset (see Data Requirements)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/umairleo1/nutrient-predictor-model-development.git
cd nutrient-predictor-model-development
```

2. Create and activate virtual environment:
```bash
python -m venv model_env
source model_env/bin/activate  # On Windows: model_env\\Scripts\\activate
```

3. Install dependencies:
```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn shap matplotlib seaborn joblib
```

## Data Requirements

### NHANES 2013-2014 Dataset

Download the following CSV files from NHANES 2013-2014 and place them in `data/raw/kaggle/`:

- `demographic.csv` - Demographic data (age, gender, race, etc.)
- `examination.csv` - Physical examination data (height, weight, BMI)
- `labs.csv` - Laboratory results (B12, hemoglobin, etc.)
- `questionnaire.csv` - Survey responses (diabetes history)
- `diet.csv` - Dietary intake data

### Clinical Thresholds Used

The models use evidence-based clinical thresholds:

- **B12 Deficiency**: < 200 pg/mL (NIH Office of Dietary Supplements)
- **Anemia Risk**: Hemoglobin < 13.0 g/dL (men) or < 12.0 g/dL (women) (WHO criteria)
- **Diabetes Risk**: Self-reported diabetes or prediabetes diagnosis (NHANES DIQ010)

## Usage

### 1. Data Preprocessing

Process raw NHANES data into training format:

```bash
python scripts/preprocess_data.py
```

This script:
- Loads and merges all NHANES datasets
- Creates evidence-based target variables
- Handles missing values and feature selection
- Saves processed data to `data/processed/training_data.csv`

### 2. Model Training

Train all nutrient deficiency prediction models:

```bash
python scripts/train_models.py
```

This script:
- Loads preprocessed training data
- Trains XGBoost models with optimization for each nutrient
- Applies SMOTE for severe class imbalance (B12)
- Performs isotonic probability calibration
- Saves models to `models/` directory
- Generates comprehensive evaluation metrics

### 3. Model Artifacts

After training, the following files are generated:

```
models/
├── b12_deficient_model.pkl           # B12 deficiency XGBoost model
├── b12_deficient_calibrated_model.pkl # Calibrated B12 model
├── b12_deficient_scaler.pkl          # B12 feature scaler
├── iron_deficient_model.pkl          # Anemia risk XGBoost model
├── iron_deficient_calibrated_model.pkl # Calibrated anemia model
├── iron_deficient_scaler.pkl         # Anemia feature scaler
├── diabetes_risk_model.pkl           # Diabetes XGBoost model
├── diabetes_risk_calibrated_model.pkl # Calibrated diabetes model
├── diabetes_risk_scaler.pkl          # Diabetes feature scaler
├── scaler.pkl                        # General feature scaler
├── feature_names.pkl                 # Feature column names
└── model_metadata.json              # Performance metrics and metadata
```

## Model Architecture

### XGBoost Configuration

Different hyperparameters optimized for each prediction task:

**B12 Deficiency Model:**
- SMOTE oversampling for rare events (< 5% prevalence)
- Moderate regularization (alpha=10, lambda=10)
- Deeper trees (max_depth=4) with balanced class weights

**Anemia Risk Model:**
- Standard XGBoost configuration for moderate prevalence
- Conservative regularization (alpha=5, lambda=10)
- Balanced approach with moderate tree depth

**Diabetes Risk Model:**
- Aggressive regularization to prevent overfitting
- Strong L1/L2 penalties (alpha=50, lambda=50)
- Shallow trees (max_depth=3) with high min_child_weight

### Probability Calibration

All models use isotonic regression for reliable probability estimates:
- Improved Brier scores for clinical decision support
- Better calibrated risk scores for patient communication
- Enhanced reliability for threshold-based interventions

## Evaluation Metrics

The pipeline provides comprehensive model evaluation:

- **ROC-AUC**: Area under ROC curve for binary classification performance
- **PR-AUC**: Precision-Recall AUC for imbalanced dataset evaluation
- **Brier Score**: Calibration quality for probability predictions
- **Cross-Validation**: 5-fold stratified CV for robust performance estimation
- **Calibration Curves**: Visual assessment of probability calibration
- **Feature Importance**: SHAP values for model interpretability

## Project Structure

```
├── scripts/                    # Main processing scripts
│   ├── preprocess_data.py      # NHANES data preprocessing pipeline
│   ├── train_models.py         # Model training and evaluation
│   └── __init__.py
├── data/                       # Data directories
│   ├── raw/                    # Raw NHANES CSV files
│   ├── processed/              # Preprocessed training data
│   └── sample/                 # Sample data for testing
├── models/                     # Trained model artifacts
├── results/                    # Evaluation results and reports
│   ├── metrics/                # Performance metrics
│   ├── reports/                # Analysis reports
│   └── shap/                   # SHAP explanations
├── notebooks/                  # Jupyter notebooks for analysis
├── tests/                      # Unit tests
├── experiments/                # Experiment tracking
└── README.md                   # This file
```

## Performance Benchmarks

Based on NHANES 2013-2014 complete dataset:

### Model Performance (Test Set)
- **B12 Deficiency**: ROC-AUC 0.75+ (rare event prediction)
- **Anemia Risk**: ROC-AUC 0.80+ (moderate prevalence)
- **Diabetes Risk**: ROC-AUC 0.70+ (complex multifactorial condition)

### Clinical Relevance
- **Sensitivity**: Optimized for early detection of deficiencies
- **Specificity**: Balanced to minimize false alarms
- **Calibration**: Reliable probability estimates for risk communication

## Clinical Validation

### Evidence-Based Thresholds
- B12 thresholds align with NIH clinical guidelines
- Anemia criteria follow WHO international standards
- Diabetes classification uses validated NHANES questionnaire items

### Population Representativeness
- NHANES provides nationally representative US population sample
- Includes diverse demographics and socioeconomic backgrounds
- Validated survey methodology with clinical measurements

## Deployment Integration

Models are designed for seamless integration with the production backend:

1. **Copy models** to backend repository:
   ```bash
   cp models/*.pkl ../backend/saved_models/
   ```

2. **Feature compatibility**: Models expect the same feature format as the web API
3. **Calibrated predictions**: Use calibrated models for user-facing risk scores
4. **SHAP explanations**: Models include explainability for transparency

## Development Workflow

### Adding New Features

1. Modify feature selection in `scripts/preprocess_data.py`
2. Update model training pipeline in `scripts/train_models.py`
3. Retrain models and evaluate performance
4. Update backend API to handle new features

### Hyperparameter Tuning

1. Modify model parameters in `ModelTrainer.train_single_model()`
2. Run training pipeline to evaluate changes
3. Use cross-validation to assess generalization
4. Update documentation with new configurations

### Data Updates

1. Replace files in `data/raw/kaggle/` with new NHANES cycles
2. Verify column names and format compatibility
3. Run complete preprocessing and training pipeline
4. Compare performance with previous model versions

## Quality Assurance

### Model Validation
- Stratified cross-validation for unbiased performance estimates
- Overfitting detection through train/test AUC comparison
- Calibration assessment using reliability diagrams
- Statistical significance testing for model comparisons

### Code Quality
- Comprehensive error handling for data processing
- Logging throughout the pipeline for debugging
- Modular design for maintainability
- Documentation for all major functions

## Limitations and Considerations

### Data Limitations
- NHANES 2013-2014 represents specific time period
- Self-reported diabetes may have recall bias
- Missing laboratory values reduce sample size
- Population may not generalize to all demographics

### Model Limitations
- Anemia model predicts hemoglobin-based risk, not iron stores specifically
- Cross-sectional data limits causal inference
- Models may require recalibration for different populations
- Clinical decision support requires medical professional oversight

## Research Applications

This pipeline supports various research applications:

- **Population Health Studies**: Analyzing nutrient deficiency patterns
- **Risk Factor Analysis**: Identifying predictive demographics and lifestyle factors
- **Clinical Decision Support**: Developing screening tools for healthcare settings
- **Public Health Policy**: Evidence-based nutrition intervention targeting

## Contributing

1. Fork the repository
2. Create a feature branch
3. Follow coding standards and add tests
4. Update documentation for changes
5. Submit a pull request

## Testing

```bash
# Run unit tests
python -m pytest tests/

# Validate data processing
python scripts/preprocess_data.py --validate

# Test model training
python scripts/train_models.py --test
```

## License

This project is for educational and research purposes only. Not intended for medical diagnosis or treatment decisions.

## Disclaimer

This machine learning pipeline is designed for research and educational purposes only. The models should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical decisions.

## Citation

If you use this work in research, please cite:

```
Nutrient Deficiency Predictor - Model Development
Muhammad Umair, 2024
NHANES 2013-2014 Machine Learning Pipeline
https://github.com/umairleo1/nutrient-predictor-model-development
```

## Contact

**Author**: Muhammad Umair  
**LinkedIn**: [muhammad-umair-amin](https://www.linkedin.com/in/muhammad-umair-amin/)  
**GitHub**: [umairleo1](https://github.com/umairleo1)

## Related Repositories

- **Backend API**: [nutrient-predictor-backend](https://github.com/umairleo1/nutrient-predictor-backend)
- **Frontend Application**: [nutrient-predictor-frontend](https://github.com/umairleo1/nutrient-predictor-frontend)

---

**Note**: This repository contains the complete model development pipeline for a healthcare ML research project. The models integrate with a full-stack web application for educational demonstration of machine learning in clinical decision support.