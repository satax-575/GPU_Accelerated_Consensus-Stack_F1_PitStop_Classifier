F1 Pit Stop Prediction - Machine Learning Project

PROJECT OVERVIEW

This project predicts Formula 1 pit stop decisions using machine learning models trained on historical F1 race data. The goal is to predict whether a driver will pit on the next lap based on tyre degradation, race strategy, lap times, and other telemetry features.

Competition: Kaggle Playground Series Season 6 Episode 5
Task: Binary classification to predict PitNextLap
Dataset: 439,140 training samples with 15 base features
Years covered: 2022-2025
Evaluation metric: ROC-AUC

KEY FEATURES

Advanced Feature Engineering
- 80+ engineered features from 15 base features
- Tyre degradation modeling with compound-specific cliff detection
- Rolling window statistics for lap time trends
- Safety car detection proxies
- Race strategy indicators including undercut and overcut windows
- Position change momentum tracking

Robust Preprocessing Pipeline
- Feature-group-specific scaling strategies
- Winsorization for extreme outliers
- Yeo-Johnson power transformation for skewed features
- Handles binary flags, ordinal integers, bounded ratios, and continuous features separately

Consensus Feature Selection
- Multi-method importance scoring combining model-based, permutation, and SHAP
- Batch forward selection with cross-validation AUC gating
- Stability checks across inner folds to prevent overfitting

Ensemble Modeling
- LightGBM with GPU acceleration
- XGBoost with CUDA support
- Hyperparameter optimization using Optuna
- Stratified K-Fold cross-validation

TECHNICAL ARCHITECTURE

The project follows a modular pipeline design with strict separation of concerns:

Data Flow
1. Raw CSV data loading from Kaggle
2. Feature engineering with fold-wise statistics
3. Preprocessing with group-specific scaling
4. Feature selection using consensus ranking
5. Model training with GPU-accelerated gradient boosting
6. Ensemble prediction and submission generation

Core Modules

f1_feature_transformer.py
- Creates 80+ engineered features from raw telemetry
- Implements fold-safe feature engineering to prevent leakage
- Handles tyre compound metadata and degradation curves
- Computes rolling statistics and lag features
- Detects safety car periods and pit clusters

f1_preprocessor.py
- Applies feature-group-specific preprocessing
- Binary flags: mode imputation, optional standardization
- Ordinal integers: median imputation, standard scaling
- High-skew continuous: Winsorization, Yeo-Johnson, standard scaling
- Bounded ratios: MinMax scaling to preserve 0-1 range
- Standard continuous: robust scaling for outlier resistance

consensus_feature_selector.py
- Removes zero-variance features
- Computes three importance scores: model-based, permutation, SHAP
- Aggregates into consensus ranking
- Runs batch forward selection with AUC improvement gates
- Ensures stability across inner CV folds

DATASET STRUCTURE

Training Data
- 439,140 lap records
- 15 base features including Driver, Compound, Race, Year, LapNumber, TyreLife, Position, LapTime, etc.
- Target variable: PitNextLap (binary: 0 or 1)

Feature Categories
- Categorical: Driver, Compound, Race
- Temporal: Year, LapNumber, RaceProgress
- Tyre: TyreLife, Compound, Stint
- Performance: LapTime, LapTime_Delta, Cumulative_Degradation
- Position: Position, Position_Change
- Strategy: PitStop, Stint

INSTALLATION AND SETUP

System Requirements
- Python 3.8 or higher
- CUDA-capable GPU recommended for training (optional but significantly faster)
- 8GB RAM minimum, 16GB recommended
- 5GB disk space for data and models

Dependencies
- pandas, numpy for data manipulation
- scikit-learn for preprocessing and model evaluation
- lightgbm for gradient boosting (GPU build recommended)
- xgboost for additional gradient boosting
- optuna for hyperparameter optimization
- torch for CUDA detection and GPU management
- kagglehub for dataset download
- matplotlib, seaborn for visualization
- shap for model interpretability (optional)
- tqdm for progress bars

GPU Setup
The notebook includes automatic CUDA detection and PyTorch installation. It will:
- Detect your CUDA version via nvcc or nvidia-smi
- Install the correct PyTorch wheel for your CUDA version
- Fall back to CPU if no GPU is detected
- Configure LightGBM and XGBoost to use GPU when available

For LightGBM GPU support, you may need to install the GPU-enabled build separately from the official releases.

USAGE

Running the Main Notebook

1. Open main_final_V3.ipynb in Jupyter or JupyterLab

2. The notebook will automatically:
   - Log in to Kaggle and download the competition data
   - Detect GPU availability and install correct PyTorch version
   - Load and preprocess the data
   - Engineer features with fold-wise safety
   - Select features using consensus ranking
   - Train ensemble models with hyperparameter optimization
   - Generate predictions and submission file

3. Key notebook sections:
   - Data loading and exploration
   - GPU detection and dependency installation
   - Feature engineering pipeline
   - Preprocessing and scaling
   - Feature selection
   - Model training with cross-validation
   - Hyperparameter tuning with Optuna
   - Test set prediction and submission

Using the Modules Independently

Feature Engineering
```python
from f1_feature_transformer import F1FoldWiseWrapper

wrapper = F1FoldWiseWrapper(n_splits=5, random_state=42)
train_feats, val_feats, tr_idxs, val_idxs = wrapper.run_cv(X_train, y_train)
wrapper.fit_full(X_train, y_train)
X_test_feat = wrapper.transform_test(X_test)
```

Preprocessing
```python
from f1_preprocessor import F1Preprocessor

preprocessor = F1Preprocessor(passthrough_binary=True, winsor_q=(0.001, 0.999))
preprocessor.fit(X_train_fold, y_train_fold)
X_train_scaled = preprocessor.transform(X_train_fold)
X_val_scaled = preprocessor.transform(X_val_fold)
```

Feature Selection
```python
from consensus_feature_selector import ConsensusFeatureSelector

selector = ConsensusFeatureSelector(
    estimator=lgb.LGBMClassifier(),
    top_n_start=40,
    batch_size=5,
    min_auc_gain=0.0005
)
selector.fit(X_train_preprocessed, y_train)
X_train_selected = selector.transform(X_train_preprocessed)
print(selector.selection_report())
```

HYPERPARAMETER OPTIMIZATION

The project includes pre-tuned hyperparameters stored in JSON files:

optuna_params_lgb.json
- LightGBM standard boosting parameters
- Optimized for ROC-AUC on validation folds

optuna_params_lgb_dart.json
- LightGBM DART (Dropouts meet Multiple Additive Regression Trees) parameters
- More robust to overfitting through dropout regularization

optuna_params_xgb.json
- XGBoost parameters
- Complementary to LightGBM for ensemble diversity

These parameters were found using Optuna with 100+ trials and can be loaded directly or used as starting points for further tuning.

MODEL TRAINING WORKFLOW

The training process follows a strict fold-wise approach to prevent data leakage:

1. Stratified K-Fold split (5 folds) on raw data
2. For each fold:
   - Fit feature transformer on training split only
   - Transform both training and validation splits
   - Fit preprocessor on training features only
   - Scale both training and validation features
   - Fit feature selector on training features only
   - Select features from both splits
   - Train model on selected training features
   - Predict on validation split
3. Aggregate out-of-fold predictions for final validation score
4. Fit final models on full training set for test predictions

This ensures all statistics (means, quantiles, importance scores) are learned only from training data within each fold.

LEAKAGE PREVENTION

The codebase includes multiple safeguards against data leakage:

Target Column Protection
- Feature transformer explicitly drops PitNextLap before returning
- Assertions check that target never appears in feature matrices
- Validation splits never see target during fit operations

Fold-Wise Statistics
- Race total laps computed per fold
- Compound cliff laps calibrated from training pits only
- Pit cluster counts aggregated within training data
- Label encoders fit on training drivers and races only

Index Alignment
- All transformed DataFrames reset to default index before concatenation
- Prevents silent NaN-filling from mismatched indexes after StratifiedKFold slicing

Rolling Features
- All rolling windows use shift(1) to prevent current-row leakage
- Group-by operations respect race, year, and driver boundaries

KNOWN ISSUES AND FIXES

The modules include detailed documentation of bugs found and fixed:

FIX-T1: Target column leakage prevention in transform
FIX-T2: Lambda closure variable capture in rolling features
FIX-T3: Nested groupby context loss in consecutive_worse
FIX-T4: MultiIndex mapping failure on default RangeIndex
FIX-T5: Target column pre-attachment validation
FIX-T6: Transformer kwargs typo detection

FIX-P1: Index misalignment in preprocessing concat
FIX-P2: Redundant fit_transform override removal
FIX-P3: Empty column group handling

FIX-S1: Warm-start edge case when top_n exceeds available features
FIX-S2: Batch filtering breaking loop prematurely
FIX-S3: Estimator cloning from canonical source
FIX-S4: Series.get() returning Series on duplicate index
FIX-S5: Fold count validation in selection rate computation
FIX-S6: SHAP API version compatibility for binary classification

OUTPUT FILES

The notebook generates several output files:

submission.csv
- Final predictions for test set in Kaggle submission format
- Columns: id, PitNextLap (probability)

feature_selection_cache.json
- Cached feature selection results to speed up re-runs
- Contains selected feature lists per fold

optuna_params_lgb.json, optuna_params_xgb.json
- Optimized hyperparameters from Optuna trials
- Can be loaded to skip hyperparameter search

PERFORMANCE METRICS

The model performance is evaluated using:

Primary Metric: ROC-AUC
- Measures discrimination ability across all thresholds
- Robust to class imbalance
- Typical cross-validation scores: 0.85-0.88

Secondary Metrics
- Precision-Recall AUC for imbalanced classes
- F1 score at optimal threshold
- Confusion matrix analysis

Cross-Validation Strategy
- 5-fold stratified K-fold
- Stratification preserves PitNextLap class distribution
- Out-of-fold predictions aggregated for unbiased validation score

COMPUTATIONAL REQUIREMENTS

Training Time Estimates

With GPU (NVIDIA RTX 3050 or better):
- Feature engineering: 5-10 minutes
- Preprocessing: 2-3 minutes
- Feature selection: 10-15 minutes
- Model training (5 folds): 15-20 minutes
- Total: approximately 35-50 minutes

Without GPU (CPU only):
- Feature engineering: 10-15 minutes
- Preprocessing: 3-5 minutes
- Feature selection: 30-45 minutes
- Model training (5 folds): 60-90 minutes
- Total: approximately 2-3 hours

Memory Usage
- Peak RAM: 6-8 GB during feature engineering
- GPU VRAM: 2-4 GB for model training
- Disk space: 2-3 GB for data and intermediate files

REPRODUCIBILITY

All random operations are seeded for reproducibility:
- StratifiedKFold: random_state=42
- Feature transformer: random_state=42
- Feature selector: random_state=42
- LightGBM: random_state=42, seed=42
- XGBoost: random_state=42, seed=42
- Optuna: sampler seed=42

Running the notebook with the same random seeds should produce identical results within floating-point precision.

PROJECT STRUCTURE

Root Directory
- main_final_V3.ipynb: Main training and prediction notebook
- f1_feature_transformer.py: Feature engineering module
- f1_preprocessor.py: Preprocessing and scaling module
- consensus_feature_selector.py: Feature selection module
- submission.csv: Final predictions for submission
- feature_selection_cache.json: Cached selection results
- optuna_params_lgb.json: LightGBM hyperparameters
- optuna_params_lgb_dart.json: LightGBM DART hyperparameters
- optuna_params_xgb.json: XGBoost hyperparameters

data Directory
- train.csv: Training data (downloaded from Kaggle)
- test.csv: Test data (downloaded from Kaggle)
- sample_submission.csv: Submission format template

Backup Directory
- Previous versions of modules and notebooks
- Kept for reference and rollback if needed

Previous Notebook Versions Directory
- Earlier iterations of the main notebook
- Useful for tracking development history

params Directory
- Duplicate copies of hyperparameter JSON files
- Organized for easy access

CONTRIBUTING

This is a competition project, but the modular design allows for easy extension:

Adding New Features
- Add feature computation logic to F1FeatureTransformer.transform()
- Update get_feature_names_out() to include new feature names
- Assign new features to appropriate preprocessing groups in f1_preprocessor.py

Adding New Models
- Clone the existing model training cell in the notebook
- Configure your model with appropriate hyperparameters
- Ensure it follows sklearn API (fit, predict_proba)
- Add predictions to the ensemble averaging

Tuning Hyperparameters
- Use the Optuna cells in the notebook as templates
- Define your search space with appropriate distributions
- Run optimization with sufficient trials (100+ recommended)
- Save best parameters to JSON for reuse

TROUBLESHOOTING

Common Issues

CUDA not detected despite having GPU
- Verify CUDA toolkit is installed (not just drivers)
- Check CUDA version with: nvcc --version
- Ensure PyTorch CUDA version matches your toolkit
- Try reinstalling PyTorch with explicit CUDA version

LightGBM not using GPU
- Install GPU-enabled LightGBM build from official releases
- Check device parameter is set to 'gpu' in model config
- Verify GPU memory is sufficient (4GB+ recommended)

Out of memory errors
- Reduce batch_size in feature selection
- Reduce shap_sample size in feature selector
- Use fewer CV folds (3 instead of 5)
- Close other GPU applications

Feature selection taking too long
- Reduce top_n_start (try 30 instead of 40)
- Increase batch_size (try 10 instead of 5)
- Reduce n_inner_splits (try 2 instead of 3)
- Use cached results from feature_selection_cache.json

Kaggle authentication fails
- Ensure kaggle.json is in the correct location
- Check API credentials are valid and not expired
- Try manual download and place files in data directory

REFERENCES

Competition
- Kaggle Playground Series S6E5: https://www.kaggle.com/competitions/playground-series-s6e5

Libraries
- LightGBM: https://lightgbm.readthedocs.io/
- XGBoost: https://xgboost.readthedocs.io/
- Optuna: https://optuna.readthedocs.io/
- SHAP: https://shap.readthedocs.io/
- scikit-learn: https://scikit-learn.org/

Formula 1 Data
- FastF1: https://docs.fastf1.dev/ (for understanding F1 telemetry)
- Pirelli Tyre Compounds: Official F1 tyre supplier documentation

ACKNOWLEDGMENTS

This project uses data from the Kaggle Playground Series competition. The feature engineering approach is inspired by domain knowledge of Formula 1 racing strategy and tyre management.

The modular architecture with strict leakage prevention follows best practices from the machine learning competition community.

LICENSE

This project is for educational and competition purposes. The code is provided as-is for learning and reference.

CONTACT

For questions about the implementation or to report issues, please refer to the inline documentation in the source files. Each module includes detailed docstrings explaining the approach and known limitations.
