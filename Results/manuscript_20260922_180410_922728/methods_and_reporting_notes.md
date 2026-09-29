# Conductivity study: methods and reporting notes

## Target and feature sets

Target: log10 ionic conductivity, with conductivity in S/cm.
Set A: chemical descriptors plus operating temperature.
Set B: Set A plus sintering temperature and time.
Set C: Set B plus available physics descriptors.
Activation energy, grain size and relative density are not conductivity predictors in this workflow.

## Evaluation

The master split groups equivalent cation compositions.
Full-data A/B and common-subset A/B/C are distinct analyses.
Imputation, constant-feature removal, scaling and SHAP selection are fitted within training CV folds.
Hyperparameters are selected by mean grouped-CV RMSE.
Primary models are selected by validation RMSE.
Frozen pipelines are refitted on training plus validation.
Test results are reported for all benchmark models, with validation-selected primary models identified.
CV tuning scores, validation scores and test scores must be labelled separately.

## Interpretation

SHAP describes model associations, not causal mechanisms.
Report the explainer method recorded for each predictor.
Bootstrap quantiles describe model variability and are not calibrated prediction intervals.
Distance and feature-range checks are heuristic applicability indicators, not feasibility tests.

## Candidate screening

Report the assumed valences, cation-fraction grid, fixed operating temperature, processing schedules and applicability rules.
A composition absent from the dataset is not necessarily novel in the literature.
Candidate recommendations are not independently validated conductivity improvements.



## Reproducibility

The model archives contain fitted primary pipelines and, when enabled, bootstrap screening ensembles.
Load pickle archives only when they come from a trusted source.
Use the recorded software versions; cross-version model loading is not guaranteed.
The code snapshot is the saved src/sourcecode.py file.
Raw data are not copied by this block. Preserve the exact input workbooks and their source provenance alongside the study.