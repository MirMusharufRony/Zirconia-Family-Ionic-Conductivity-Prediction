# %% Block 1: Importing necessary libraries and modules

# Standard Python utilities
import os
import sys
import re
import json
import random
import platform
import time
import platform
import hashlib
import warnings
import gzip
import shutil
from datetime import datetime
from importlib.metadata import version, PackageNotFoundError
from joblib import memory


from importlib.metadata import version

# Progress bars and model saving
from tqdm.auto import tqdm
import joblib

# Numerical calculations and data handling
import numpy as np
import pandas as pd

# Statistical distributions, Arrhenius fitting and Bayesian optimization
from scipy.stats import (
    randint,
    uniform,
    loguniform,
    linregress,
    norm,
)

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Chemical formulas and ionic properties
from pymatgen.core import Composition, Element, Species

# Matminer chemical feature generation
import matminer

from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.conversions import StrToComposition

# Scikit-learn estimator utilities and pipelines
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline

# Preprocessing and feature filtering
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold


# Composition-grouped splitting and hyperparameter tuning
from sklearn.model_selection import (
    GroupShuffleSplit,
    GroupKFold,
    cross_val_score,
    cross_val_predict,
    cross_validate,
    GridSearchCV,
    RandomizedSearchCV,
    ParameterGrid,
    learning_curve,
)

# Baseline and linear regression models
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
)

# Support-vector and nearest-neighbour regression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors

# Tree-based regression models
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    AdaBoostRegressor,
)

# Gaussian-process regression and Bayesian-optimization surrogate
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    RBF,
    Matern,
    WhiteKernel,
)

# Scikit-learn neural-network regression
from sklearn.neural_network import MLPRegressor

# Regression evaluation
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

# Threshold-based classification evaluation
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    roc_auc_score,
    auc,
)

# XGBoost regression
import xgboost as xgb
from xgboost import XGBRegressor

# SHAP feature selection and interpretation
import shap
from sklearn.utils.validation import check_is_fitted

print("All libraries imported successfully.")
print("Python executable:", sys.executable)
print("Python version:", platform.python_version())
print("matminer version:", version("matminer"))
print("scikit-learn version:", version("scikit-learn"))











# %% Block 2: Loading and inspecting the dataset

# Project paths
PROJECT_FOLDER = (
    r"C:\Users\akter\OneDrive\Desktop"
    r"\Zirconia Family Ionic Conductivity Prediction"
)

DATASET_FOLDER = os.path.join(PROJECT_FOLDER, "Dataset")
FIGURES_FOLDER = os.path.join(PROJECT_FOLDER, "Figures")

DATASET_NAME = "book1 auto.xlsx"
DATASET_PATH = os.path.join(DATASET_FOLDER, DATASET_NAME)

# Checking the dataset folder
if not os.path.isdir(DATASET_FOLDER):
    raise FileNotFoundError(
        f"Dataset folder not found:\n{DATASET_FOLDER}"
    )

# Checking the file
if not os.path.isfile(DATASET_PATH):
    available_files = os.listdir(DATASET_FOLDER)

    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_PATH}\n\n"
        f"Files available in Dataset:\n{available_files}"
    )

# Inspecting worksheets and loading the first worksheet
with pd.ExcelFile(DATASET_PATH, engine="openpyxl") as workbook:
    print("Available worksheets:", workbook.sheet_names)

    SHEET_NAME = workbook.sheet_names[0]

    df_raw = pd.read_excel(
        workbook,
        sheet_name=SHEET_NAME,
    )

# Preserving the original dataframe for reference
df = df_raw.copy(deep=True)

print("\nLoaded file:", DATASET_PATH)
print("Selected worksheet:", SHEET_NAME)
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

if df.empty:
    raise ValueError(
        "The selected worksheet contains no data. "
        "Check the worksheet selection."
    )

# Seeing exact column names, including accidental spaces
print("\nCOLUMN NAMES")

for position, column in enumerate(df.columns, start=1):
    print(f"{position:02d}. {column!r}")

# Preview the data
print("\nFIRST FIVE ROWS")
print(df.head())

# Column types and missing values
column_summary = pd.DataFrame({
    "Data type": df.dtypes.astype(str),
    "Non-missing values": df.notna().sum(),
    "Missing values": df.isna().sum(),
    "Missing (%)": df.isna().mean().mul(100).round(2),
    "Unique non-missing values": df.nunique(dropna=True),
})

column_summary.index.name = "Column"

print("\nCOLUMN SUMMARY")
print(column_summary)

# Checking whitespace-only text cells separately from NaN values
blank_text_counts = df.apply(
    lambda column: column.map(
        lambda value: isinstance(value, str)
        and value.strip() == ""
    ).sum()
)

if blank_text_counts.gt(0).any():
    print("\nWHITESPACE-ONLY TEXT CELLS")
    print(
        blank_text_counts[blank_text_counts.gt(0)]
        .rename("Blank text cells")
        .to_frame()
    )

# Flaging exact duplicates for review; do not remove them
duplicate_mask = df.duplicated(keep=False)

print(
    "\nRows involved in exact duplicates:",
    int(duplicate_mask.sum()),
)
print(
    "Additional duplicate copies:",
    int(df.duplicated().sum()),
)

if duplicate_mask.any():
    print("\nDUPLICATE ROW PREVIEW")
    print(df.loc[duplicate_mask].head(10))

# Numeric ranges for preliminary inspection
numeric_columns = df.select_dtypes(include="number").columns

if len(numeric_columns) > 0:
    print("\nNUMERIC SUMMARY")
    print(df[numeric_columns].describe().T)

    infinite_counts = (
        df[numeric_columns]
        .isin([np.inf, -np.inf])
        .sum()
    )

    if infinite_counts.gt(0).any():
        print("\nINFINITE VALUES")
        print(
            infinite_counts[infinite_counts.gt(0)]
            .rename("Infinite values")
            .to_frame()
        )

print("\nDataset inspection complete. No rows or values were changed.")










# %% Block 3: Preparing columns and checking data types

# Working on a copy; preserving df_raw and the original workbook
data = df_raw.copy(deep=True)

# Preserving the original Excel row number
# headers occupy Excel row 1
data.insert(
    0,
    "excel_row",
    np.arange(2, len(data) + 2),
)

# Identifying the conductivity column without manually typing its backslash
conductivity_columns = [
    column
    for column in data.columns
    if str(column).strip().startswith("Ion Conductivity")
]

if len(conductivity_columns) != 1:
    raise ValueError(
        "Expected exactly one conductivity column, "
        f"but found: {conductivity_columns}"
    )

# Renaming columns only in our working dataframe
column_mapping = {
    "Chemical Formula": "formula",
    "Sintering Temperature(K)": "sintering_temperature_K",
    "Sintering Time(h)": "sintering_time_h",
    "Relative Density (%)": "relative_density_pct",
    "Grain Size(μm)": "grain_size_um",
    "Activation Energy(eV)": "activation_energy_eV",
    "Operating Temperature(K)": "temperature_K",
    conductivity_columns[0]: "log10_sigma_S_cm",
    "Paper Id(doi)": "paper_id",
    "Publisher": "publisher",
}

missing_columns = set(column_mapping) - set(data.columns)

if missing_columns:
    raise ValueError(
        f"Expected columns are missing: {sorted(missing_columns)}"
    )

data = data.rename(columns=column_mapping)

# Preserving original formulas before cleaning text
data["formula_original"] = data["formula"].copy()

# Removing whitespace and invisible characters.
# Do not normalize or change the chemical coefficients.
data["formula"] = (
    data["formula"]
    .astype("string")
    .str.replace(r"[\s\u200b\u200c\u200d\ufeff]", "", regex=True)
    .replace("", pd.NA)
)

# Cleaning metadata text while preserving missing values
for column in ["paper_id", "publisher"]:
    data[column] = (
        data[column]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
    )

# Columns that must contain numbers or missing values
numeric_columns = [
    "sintering_temperature_K",
    "sintering_time_h",
    "relative_density_pct",
    "grain_size_um",
    "activation_energy_eV",
    "temperature_K",
    "log10_sigma_S_cm",
]

# Detect invalid entries before accepting numeric conversion
invalid_entries = []

for column in numeric_columns:
    original = data[column].replace(r"^\s*$", np.nan, regex=True)
    converted = pd.to_numeric(original, errors="coerce")

    invalid_mask = (
        original.notna() & converted.isna()
    ) | converted.isin([np.inf, -np.inf])

    if invalid_mask.any():
        flagged = data.loc[invalid_mask, ["excel_row"]].copy()
        flagged["column"] = column
        flagged["original_value"] = original.loc[invalid_mask]
        invalid_entries.append(flagged)

    data[column] = converted

if invalid_entries:
    print("Entries requiring review:")
    print(pd.concat(invalid_entries, ignore_index=True))

    raise ValueError(
        "Invalid numeric entries were found. "
        "Review the listed Excel cells before continuing."
    )

# Essential fields for conductivity modelling
required_fields = [
    "formula",
    "temperature_K",
    "log10_sigma_S_cm",
]

missing_required = data[required_fields].isna().any(axis=1)

if missing_required.any():
    print("Rows with missing essential fields:")
    print(
        data.loc[
            missing_required,
            ["excel_row", *required_fields],
        ]
    )

    raise ValueError(
        "Formula, measurement temperature and conductivity "
        "must be present. No rows were automatically removed."
    )

if (data["temperature_K"] <= 0).any():
    raise ValueError(
        "Measurement temperatures must be positive Kelvin values."
    )

# Defining the experimental inputs for later feature construction
processing_columns = [
    "sintering_temperature_K",
    "sintering_time_h",
    "relative_density_pct",
    "grain_size_um",
]

conductivity_target = "log10_sigma_S_cm"

# Compacting summary that should fit in the output window
summary_columns = [
    "formula",
    *numeric_columns,
    "paper_id",
    "publisher",
]

summary = pd.DataFrame({
    "dtype": data[summary_columns].dtypes.astype(str),
    "missing": data[summary_columns].isna().sum(),
    "unique": data[summary_columns].nunique(dropna=True),
})

print("Rows retained:", len(data))
print("Unique cleaned formula strings:", data["formula"].nunique())
print("Unique source identifiers:", data["paper_id"].nunique())

print("\nCOLUMN CHECK")
print(summary.to_string())

print("\nConductivity target:", conductivity_target)
print("Activation energy excluded from conductivity inputs.")
print("No imputation, scaling or feature selection performed yet.")










# %% Block 4: Generating matminer chemical features and composition groups

# One calculation per unique cleaned formula
unique_formulas = data["formula"].dropna().unique()

composition_lookup = {}
group_lookup = {}
formula_errors = []

for formula in unique_formulas:
    try:
        composition = Composition(formula, strict=True)
        amounts = composition.get_el_amt_dict()

        # Basic formula checks
        if any(amount <= 0 for amount in amounts.values()):
            raise ValueError("Element amounts must be positive.")

        if amounts.get("Zr", 0) <= 0:
            raise ValueError("Formula contains no zirconium.")

        if amounts.get("O", 0) <= 0:
            raise ValueError("Formula contains no oxygen.")

        composition_lookup[formula] = composition

        # Group by cation proportions.
        # Oxygen variants of the same cation composition stay together.
        cation_amounts = {
            element: amount
            for element, amount in amounts.items()
            if element != "O"
        }

        total_cations = sum(cation_amounts.values())

        composition_group = "|".join(
            f"{element}:{cation_amounts[element] / total_cations:.7f}"
            for element in sorted(cation_amounts)
        )

        group_lookup[formula] = composition_group

    except Exception as error:
        formula_errors.append({
            "formula": formula,
            "error": str(error),
        })

# Stopping rather than silently discarding invalid formulas
if formula_errors:
    print(pd.DataFrame(formula_errors))

    raise ValueError(
        "Some formulas could not be processed. "
        "Review the formulas listed above."
    )

# Add grouping metadata, not a predictor
data["composition_group"] = data["formula"].map(group_lookup)

# Magpie elemental-property statistics
# Keep unavailable elemental-property values as missing.
chemical_featurizer = ElementProperty.from_preset(
    "magpie",
    impute_nan=False,
)

chemical_feature_names = [
    f"chem__{name}"
    for name in chemical_featurizer.feature_labels()
]

# Generating descriptors for each unique formula
feature_rows = []

for formula in tqdm(
    unique_formulas,
    desc="Generating chemical features",
):
    try:
        values = chemical_featurizer.featurize(
            composition_lookup[formula]
        )
        feature_rows.append(values)

    except Exception as error:
        raise RuntimeError(
            f"Chemical feature generation failed for: {formula}"
        ) from error

# Lookup table: one row per unique formula
chemical_feature_lookup = pd.DataFrame(
    feature_rows,
    index=pd.Index(unique_formulas, name="formula"),
    columns=chemical_feature_names,
    dtype=float,
)

# Expanding back to the original measurement rows
# Repeated formulas receive the same chemical descriptors.
X_chemical = chemical_feature_lookup.loc[
    data["formula"].tolist()
].copy()

X_chemical.index = data.index

# Confirming that feature rows align with the dataset
assert len(X_chemical) == len(data)
assert X_chemical.index.equals(data.index)

# Checking for infinite values
infinite_mask = np.isinf(X_chemical.to_numpy())

if infinite_mask.any():
    affected_columns = X_chemical.columns[
        infinite_mask.any(axis=0)
    ].tolist()

    raise ValueError(
        f"Infinite chemical descriptors found: {affected_columns}"
    )

# Summarizing missing descriptor values without imputing them
missing_feature_counts = X_chemical.isna().sum()
features_with_missing = missing_feature_counts[
    missing_feature_counts > 0
].sort_values(ascending=False)

print("\nCHEMICAL FEATURE GENERATION COMPLETE")
print("Measurement rows:", len(data))
print("Unique formula strings:", len(unique_formulas))
print("Composition groups:", data["composition_group"].nunique())
print("Chemical features:", X_chemical.shape[1])
print("Feature matrix shape:", X_chemical.shape)

print(
    "Features containing missing values:",
    len(features_with_missing),
)

if not features_with_missing.empty:
    print("\nMISSING VALUES — first 15 affected features")
    print(features_with_missing.head(15).to_string())

print("\nFORMULA AND GROUP PREVIEW")
print(
    data[
        ["excel_row", "formula", "composition_group"]
    ].head()
)

print("\nCHEMICAL FEATURE PREVIEW — first 8 features")
print(X_chemical.iloc[:5, :8])





# %% Block 5: Calculating physics descriptors and assembling feature sets

# ------------------------------------------------------------
# 5.1 Configuration
# ------------------------------------------------------------

PHYSICS_FILE = "physics_descriptor_inputs_all_39_papers_final.xlsx"
PHYSICS_PATH = os.path.join(DATASET_FOLDER, PHYSICS_FILE)

ALLOWED_RADIUS_COORDINATIONS = {5, 6, 8}
FRACTION_TOLERANCE = 1e-6

RADIUS_CONVENTION_LABEL = (
    "Documented coordination-specific radii: CN5, CN6 or CN8"
)

UNRESOLVED_RADIUS_SOURCES = {
    "",
    "nan",
    "none",
    "not reported",
    "unresolved",
    "reference basis not separately documented.",
}

# Optional source-identifier mappings.
# Values can be DOI strings or identifiers such as Scopus-54507495.
PAPER_ID_TO_DOI = {
    "Slilaty1996_transition_metal_YSZ": "Scopus-54507495",
}

# Optional explicit mapping for an ambiguous main-dataset row.
# Format: {original_excel_row_number: "composition_id"}
ROW_COMPOSITION_OVERRIDES = {}

# Optional reviewed vacancy-model decisions.
# Format: {"composition_id": True or False}
VACANCY_MODEL_OVERRIDES = {}

physics_columns = [
    "phys_dopant_cation_fraction",
    "phys_mean_dopant_valence",
    "phys_nominal_vacancies_per_cation",
    "phys_radius_mismatch_A",
    "phys_cation_radius_variance_A2",
]

processing_model_columns = [
    "sintering_temperature_K",
    "sintering_time_h",
]

if not os.path.isfile(PHYSICS_PATH):
    raise FileNotFoundError(
        f"Physics workbook not found:\n{PHYSICS_PATH}"
    )

if not data.index.is_unique:
    raise ValueError("The main dataset index must be unique.")

if not X_chemical.index.equals(data.index):
    raise ValueError(
        "X_chemical and data are misaligned. Rerun Blocks 2–4."
    )

# ------------------------------------------------------------
# 5.2 Loading the secondary workbook
# ------------------------------------------------------------

composition_columns = [
    "composition_id",
    "formula",
    "reported_phase",
    "substitution_basis",
    "source_reference",
    "notes",
    "paper_id",
    "sample_id",
]

# Spin state is deliberately not loaded.
ion_columns = [
    "composition_id",
    "element",
    "oxidation_state",
    "ionic_radius_A",
    "coordination_number",
    "valence_basis",
    "valence_source",
    "radius_source",
    "state_fraction",
]

with pd.ExcelFile(PHYSICS_PATH, engine="openpyxl") as workbook:

    compositions = pd.read_excel(
        workbook,
        sheet_name="Compositions",
        usecols=composition_columns,
    ).dropna(how="all")

    ions = pd.read_excel(
        workbook,
        sheet_name="Ion_Assignments",
        usecols=ion_columns,
    ).dropna(how="all")

    paper_review = pd.read_excel(
        workbook,
        sheet_name="Paper_Review",
    )

for table in [compositions, ions]:
    table["composition_id"] = (
        table["composition_id"]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
    )

for column in ["formula", "paper_id"]:
    compositions[column] = (
        compositions[column]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
    )

ions["element"] = (
    ions["element"]
    .astype("string")
    .str.strip()
    .replace("", pd.NA)
)

if compositions[
    ["composition_id", "formula", "paper_id"]
].isna().any().any():
    raise ValueError(
        "Some composition rows lack an ID, formula or paper identifier."
    )

if compositions["composition_id"].duplicated().any():
    raise ValueError("Duplicate composition IDs found.")

if ions[["composition_id", "element"]].isna().any().any():
    raise ValueError("Some ion rows lack an ID or element.")

unknown_ids = (
    set(ions["composition_id"])
    - set(compositions["composition_id"])
)

if unknown_ids:
    raise ValueError(
        f"Unknown composition IDs in Ion_Assignments: "
        f"{sorted(unknown_ids)}"
    )

numeric_ion_columns = [
    "oxidation_state",
    "ionic_radius_A",
    "coordination_number",
    "state_fraction",
]

for column in numeric_ion_columns:
    cleaned = ions[column].replace(r"^\s*$", np.nan, regex=True)
    ions[column] = pd.to_numeric(cleaned, errors="raise")

    if ions[column].isin([np.inf, -np.inf]).any():
        raise ValueError(f"Infinite values found in {column}.")

if (ions["ionic_radius_A"].dropna() <= 0).any():
    raise ValueError("Ionic radii must be positive.")

if (ions["coordination_number"].dropna() <= 0).any():
    raise ValueError("Coordination numbers must be positive.")

if not ions["state_fraction"].dropna().between(0, 1).all():
    raise ValueError("State fractions must lie between 0 and 1.")

# ------------------------------------------------------------
# 5.3 Formula and source-identifier helpers
# ------------------------------------------------------------

def clean_formula_key(value):
    if pd.isna(value):
        return ""

    return re.sub(
        r"[\s\u200b\u200c\u200d\ufeff]",
        "",
        str(value),
    )


def normalized_text(value):
    if pd.isna(value):
        return ""

    return str(value).strip().lower()


def extract_dois(value):
    if pd.isna(value):
        return set()

    matches = re.findall(
        r"10\.\d{4,9}/[^\s;<>\"]+",
        str(value),
        flags=re.IGNORECASE,
    )

    return {
        match.rstrip(".,)]}").lower()
        for match in matches
    }


def normalize_source_identifier(value):
    dois = extract_dois(value)

    if len(dois) == 1:
        return next(iter(dois))

    label = normalized_text(value)

    # Accept common formatting variations of Scopus identifiers.
    scopus_match = re.fullmatch(
        r"(?:doi\s*:\s*)?scopus[\s:_-]*(\d+)",
        label,
    )

    if scopus_match:
        return f"scopus-{scopus_match.group(1)}"

    return label


# Explicit mappings take precedence over automatic extraction.
explicit_aliases = {
    normalized_text(label): normalize_source_identifier(identifier)
    for label, identifier in PAPER_ID_TO_DOI.items()
}

if any(not value for value in explicit_aliases.values()):
    raise ValueError("An explicit paper mapping has an empty identifier.")

paper_aliases = {}
doi_mapping_issues = []

for label, group in compositions.groupby("paper_id"):
    label_key = normalized_text(label)
    direct_identifier = normalize_source_identifier(label)

    if label_key in explicit_aliases:
        paper_aliases[label_key] = explicit_aliases[label_key]
        continue

    # A paper_id already containing a DOI or Scopus ID needs no lookup.
    if (
        len(extract_dois(label)) == 1
        or re.fullmatch(r"scopus-\d+", direct_identifier)
    ):
        paper_aliases[label_key] = direct_identifier
        continue

    candidates = set()

    for reference in group["source_reference"]:
        candidates.update(extract_dois(reference))

    review_mask = (
        paper_review["paper_id"]
        .map(normalized_text)
        .eq(label_key)
        & paper_review["evidence_field"]
        .map(normalized_text)
        .eq("doi")
    )

    for reference in paper_review.loc[
        review_mask,
        "evidence / review note",
    ]:
        candidates.update(extract_dois(reference))

    if len(candidates) == 1:
        paper_aliases[label_key] = next(iter(candidates))
    else:
        doi_mapping_issues.append({
            "paper_id": label,
            "candidate_DOIs": "; ".join(sorted(candidates)),
            "issue": (
                "No DOI mapping found; exact source-label matching remains available"
                if not candidates
                else "Multiple DOI mappings found; review required"
            ),
        })

paper_aliases.update(explicit_aliases)


def paper_key(value):
    label = normalized_text(value)

    return paper_aliases.get(
        label,
        normalize_source_identifier(value),
    )


compositions["formula_key"] = compositions["formula"].map(
    clean_formula_key
)

compositions["paper_key"] = compositions["paper_id"].map(paper_key)

# ------------------------------------------------------------
# 5.4 Determining vacancy-model eligibility
# ------------------------------------------------------------

def vacancy_model_decision(record):
    cid = record["composition_id"]

    if cid in VACANCY_MODEL_OVERRIDES:
        decision = VACANCY_MODEL_OVERRIDES[cid]

        if not isinstance(decision, bool):
            raise ValueError(
                f"Vacancy override for {cid} must be True or False."
            )

        return decision, "explicit reviewed override"

    basis = normalized_text(record["substitution_basis"])

    # Conservative text-based classification.
    # Reviewing unresolved cases rather than assigning vacancies automatically.
    excluded_terms = [
        "separate",
        "heterogeneous",
        "ceria-rich",
        "grain bound",
        "grain-bound",
        "surface",
        "unresolved",
    ]

    if any(term in basis for term in excluded_terms):
        return False, "whole-composition vacancy model not established"

    supported_terms = [
        "zr-site substitution",
        "on zr sites",
        "on zr4+ sites",
        "zirconia cation sublattice",
        "zirconia cation",
    ]

    if any(term in basis for term in supported_terms):
        return True, "substitution model stated in input workbook"

    if basis == "host" or "host reference" in basis:
        return True, "host reference assignment"

    return False, "vacancy model requires review"


# ------------------------------------------------------------
# 5.5 Calculating descriptors for each composition record
# ------------------------------------------------------------

physics_records = []

for _, record in tqdm(
    compositions.iterrows(),
    total=len(compositions),
    desc="Calculating physics descriptors",
):
    cid = record["composition_id"]
    formula = record["formula_key"]

    composition = Composition(formula, strict=True)
    amounts = composition.get_el_amt_dict()

    if (
        amounts.get("Zr", 0) <= 0
        or amounts.get("O", 0) <= 0
        or any(amount <= 0 for amount in amounts.values())
    ):
        raise ValueError(f"Invalid zirconium oxide formula for {cid}.")

    cation_amounts = {
        element: amount
        for element, amount in amounts.items()
        if element != "O"
    }

    total_cations = sum(cation_amounts.values())

    cation_fractions = {
        element: amount / total_cations
        for element, amount in cation_amounts.items()
    }

    assignment = ions.loc[
        ions["composition_id"].eq(cid)
    ].copy()

    if set(assignment["element"]) != set(cation_amounts):
        raise ValueError(
            f"Ion coverage does not match the formula for {cid}."
        )

    issues = []
    element_charges = {}
    radius_states = []
    radius_inputs_valid = True

    for element, rows in assignment.groupby("element"):
        state_fractions = rows["state_fraction"]

        fractions_valid = (
            state_fractions.notna().all()
            and np.isclose(
                state_fractions.sum(),
                1.0,
                atol=FRACTION_TOLERANCE,
                rtol=0,
            )
        )

        if not fractions_valid:
            issues.append(
                f"{element}: unresolved or invalid state fractions"
            )
            radius_inputs_valid = False
            continue

        # States assigned zero fraction do not contribute.
        active = rows.loc[state_fractions.gt(0)]

        # Weighted charge across the assigned states of this element.
        if active["oxidation_state"].notna().all():
            element_charges[element] = float(
                (
                    active["state_fraction"]
                    * active["oxidation_state"]
                ).sum()
            )
        else:
            issues.append(f"{element}: missing oxidation state")

        numeric_inputs_complete = active[
            [
                "oxidation_state",
                "ionic_radius_A",
                "coordination_number",
            ]
        ].notna().all().all()

        coordination_allowed = (
            active["coordination_number"]
            .isin(ALLOWED_RADIUS_COORDINATIONS)
            .all()
        )

        radius_positive = active["ionic_radius_A"].gt(0).all()

        radius_sources = (
            active["radius_source"]
            .map(normalized_text)
        )

        source_documented = (
            ~radius_sources.isin(UNRESOLVED_RADIUS_SOURCES)
        ).all()

        valid_radii = (
            numeric_inputs_complete
            and coordination_allowed
            and radius_positive
            and source_documented
        )

        if valid_radii:
            for _, state in active.iterrows():
                radius_states.append({
                    "element": element,
                    "weight": (
                        cation_fractions[element]
                        * state["state_fraction"]
                    ),
                    "radius": float(state["ionic_radius_A"]),
                })

        else:
            radius_inputs_valid = False
            reasons = []

            if not numeric_inputs_complete:
                reasons.append(
                    "missing oxidation state, radius or coordination"
                )

            if not coordination_allowed:
                reasons.append(
                    "coordination missing or outside CN5/CN6/CN8"
                )

            if not radius_positive:
                reasons.append("radius missing or nonpositive")

            if not source_documented:
                reasons.append("radius source unresolved")

            issues.append(
                f"{element}: " + ", ".join(reasons)
            )

    dopants = [
        element
        for element in cation_fractions
        if element != "Zr"
    ]

    dopant_fraction = sum(
        cation_fractions[element]
        for element in dopants
    )

    result = {
        "composition_id": cid,
        "formula_key": formula,
        "paper_key": record["paper_key"],
        **{column: np.nan for column in physics_columns},
        "signed_charge_deficit": np.nan,
        "formula_charge_residual_per_cation": np.nan,
        "vacancy_model_usable": False,
        "radius_inputs_usable": False,
        "radius_convention": RADIUS_CONVENTION_LABEL,
    }

    # Descriptor 1: all non-Zr cations / all cations
    result["phys_dopant_cation_fraction"] = dopant_fraction

    # Descriptor 2: weighted mean dopant valence
    if dopant_fraction > 0 and all(
        element in element_charges
        for element in dopants
    ):
        result["phys_mean_dopant_valence"] = (
            sum(
                cation_fractions[element] * element_charges[element]
                for element in dopants
            )
            / dopant_fraction
        )

    elif dopant_fraction == 0:
        issues.append("mean dopant valence undefined: no dopants")

    # Descriptor 3: nominal vacancies under the approved model
    model_allowed, model_reason = vacancy_model_decision(record)
    result["vacancy_model_reason"] = model_reason

    if all(
        element in element_charges
        for element in cation_fractions
    ):
        mean_charge = sum(
            cation_fractions[element] * element_charges[element]
            for element in cation_fractions
        )

        deficit = 4.0 - mean_charge

        # Remove floating-point noise only, not physical negative deficits.
        if abs(deficit) < 1e-12:
            deficit = 0.0

        result["signed_charge_deficit"] = deficit

        result["formula_charge_residual_per_cation"] = (
            mean_charge
            - 2.0 * amounts["O"] / total_cations
        )

        if model_allowed and deficit >= 0:
            result["phys_nominal_vacancies_per_cation"] = deficit / 2.0
            result["vacancy_model_usable"] = True

        elif deficit < 0:
            issues.append(
                "negative charge deficit: vacancy model inapplicable"
            )

        else:
            issues.append(model_reason)

    else:
        issues.append(
            "charge descriptors require complete valence assignments"
        )

    # Descriptors 4 and 5: radius mismatch and radius variance
    if radius_inputs_valid:
        weights = np.array(
            [state["weight"] for state in radius_states],
            dtype=float,
        )

        radii = np.array(
            [state["radius"] for state in radius_states],
            dtype=float,
        )

        if not np.isclose(
            weights.sum(),
            1.0,
            atol=FRACTION_TOLERANCE,
            rtol=0,
        ):
            raise ValueError(
                f"Radius-state weights do not sum to 1 for {cid}."
            )

        # Normalize only numerical summation tolerance.
        weights = weights / weights.sum()

        mean_radius = np.sum(weights * radii)

        result["phys_cation_radius_variance_A2"] = float(
            np.sum(weights * (radii - mean_radius) ** 2)
        )

        # Weighted reference radius if Zr has multiple assigned states.
        zr_states = [
            state
            for state in radius_states
            if state["element"] == "Zr"
        ]

        zr_weight = sum(state["weight"] for state in zr_states)

        zr_radius = (
            sum(
                state["weight"] * state["radius"]
                for state in zr_states
            )
            / zr_weight
        )

        dopant_states = [
            state
            for state in radius_states
            if state["element"] != "Zr"
        ]

        dopant_weight = sum(
            state["weight"]
            for state in dopant_states
        )

        if dopant_weight > 0:
            result["phys_radius_mismatch_A"] = (
                sum(
                    state["weight"]
                    * abs(state["radius"] - zr_radius)
                    for state in dopant_states
                )
                / dopant_weight
            )
        else:
            result["phys_radius_mismatch_A"] = 0.0

        result["radius_inputs_usable"] = True

    # Recording CN combinations for inspection only.
    result["recorded_coordination_numbers"] = ", ".join(
        f"CN{value:g}"
        for value in sorted(
            assignment["coordination_number"].dropna().unique()
        )
    )

    # Used to distinguish conflicting repeated assignments.
   
    signature_columns = [
        "element",
        "oxidation_state",
        "ionic_radius_A",
        "coordination_number",
        "state_fraction",
    ]

    assignment_signature = (
        assignment[signature_columns]
        .sort_values(
            [
                "element",
                "oxidation_state",
                "coordination_number",
                "ionic_radius_A",
                "state_fraction",
            ],
            na_position="last",
        )
        .to_json(orient="records", double_precision=15)
    )

    result["assignment_signature"] = (
        assignment_signature
        + f"|vacancy_model={model_allowed}"
        + f"|radius_usable={radius_inputs_valid}"
    )

    result["issues"] = "; ".join(dict.fromkeys(issues))

    physics_records.append(result)

physics_by_composition = pd.DataFrame(physics_records)

# ------------------------------------------------------------
# 5.6 Aligning descriptor records with main measurement rows
# ------------------------------------------------------------

physics_by_id = physics_by_composition.set_index("composition_id")

grouped_candidates = {
    key: group
    for key, group in physics_by_composition.groupby(
        ["paper_key", "formula_key"]
    )
}

matched_records = []

for _, measurement in data.iterrows():
    excel_row = int(measurement["excel_row"])

    key = (
        paper_key(measurement["paper_id"]),
        clean_formula_key(measurement["formula"]),
    )

    output = {
        "excel_row": excel_row,
        "composition_ids": "",
        "match_status": "unmatched",
        **{column: np.nan for column in physics_columns},
        "issues": "",
    }

    if excel_row in ROW_COMPOSITION_OVERRIDES:
        cid = ROW_COMPOSITION_OVERRIDES[excel_row]

        if cid not in physics_by_id.index:
            raise ValueError(f"Unknown override composition ID: {cid}")

        selected = physics_by_id.loc[cid]

        if selected["formula_key"] != key[1]:
            raise ValueError(
                f"Formula mismatch in override for Excel row {excel_row}."
            )

        output["composition_ids"] = cid
        output["match_status"] = "explicit mapping"

    else:
        candidates = grouped_candidates.get(key)

        if candidates is None:
            output["issues"] = "No source/formula match found"
            matched_records.append(output)
            continue

        if candidates["assignment_signature"].nunique() != 1:
            output["composition_ids"] = "; ".join(
                candidates["composition_id"]
            )
            output["match_status"] = "ambiguous assignments"
            output["issues"] = (
                "Multiple non-equivalent assignments need explicit mapping"
            )
            matched_records.append(output)
            continue

        selected = candidates.iloc[0]

        output["composition_ids"] = "; ".join(
            candidates["composition_id"]
        )

        output["match_status"] = (
            "unique match"
            if len(candidates) == 1
            else "equivalent assignments"
        )

    for column in physics_columns:
        output[column] = selected[column]

    output["issues"] = selected["issues"]
    matched_records.append(output)

physics_alignment = pd.DataFrame(
    matched_records,
    index=data.index,
)

X_physics = physics_alignment[physics_columns].copy()

assert len(X_physics) == len(data)
assert X_physics.index.equals(data.index)
assert physics_alignment["excel_row"].tolist() == data["excel_row"].tolist()

# ------------------------------------------------------------
# 5.7 Assembling revised A/B/C feature sets
# ------------------------------------------------------------

# A: chemical descriptors and operating temperature, all rows
X_A = pd.concat(
    [
        X_chemical,
        data[["temperature_K"]],
    ],
    axis=1,
)

# B: A plus sintering temperature and time, all rows
X_B = pd.concat(
    [
        X_A,
        data[processing_model_columns],
    ],
    axis=1,
)

# Full C table for auditing missingness, before eligibility filtering
X_C_all = pd.concat(
    [
        X_B,
        X_physics,
    ],
    axis=1,
)

y_conductivity = data["log10_sigma_S_cm"].copy()
composition_groups = data["composition_group"].copy()

excluded_predictors = {
    "activation_energy_eV",
    "grain_size_um",
    "relative_density_pct",
    "paper_id",
    "publisher",
    "reported_phase",
    "spin_state",
}

for name, matrix in {
    "A": X_A,
    "B": X_B,
    "C_all": X_C_all,
}.items():

    if not matrix.index.equals(data.index):
        raise ValueError(f"Feature set {name} is misaligned.")

    if matrix.columns.duplicated().any():
        raise ValueError(f"Duplicate columns in feature set {name}.")

    unexpected_columns = set(matrix.columns) & excluded_predictors

    if unexpected_columns:
        raise ValueError(
            f"Unexpected predictors in {name}: "
            f"{sorted(unexpected_columns)}"
        )

# C requires resolved matching and all five physics descriptors.
physics_comparison_mask = (
    physics_alignment["match_status"].isin([
        "unique match",
        "equivalent assignments",
        "explicit mapping",
    ])
    & X_physics.notna().all(axis=1)
)

X_C = X_C_all.loc[physics_comparison_mask].copy()
y_C = y_conductivity.loc[physics_comparison_mask].copy()
groups_C = composition_groups.loc[physics_comparison_mask].copy()

# Full-dataset analysis: A versus B
feature_sets_full = {
    "A": X_A,
    "B": X_B,
}

y_full = y_conductivity.copy()
groups_full = composition_groups.copy()

# Controlled physics analysis: A/B/C on identical eligible rows
feature_sets_comparison = {
    "A": X_A.loc[physics_comparison_mask].copy(),
    "B": X_B.loc[physics_comparison_mask].copy(),
    "C": X_C.copy(),
}

y_comparison = y_C.copy()
groups_comparison = groups_C.copy()

for name, matrix in feature_sets_comparison.items():
    if not matrix.index.equals(y_comparison.index):
        raise ValueError(
            f"Comparison feature set {name} is misaligned."
        )

# ------------------------------------------------------------
# 5.8 Displaying updated coverage and remaining issues
# ------------------------------------------------------------

coverage = pd.DataFrame({
    "available_measurement_rows": X_physics.notna().sum(),
    "missing_measurement_rows": X_physics.isna().sum(),
    "missing_pct": (
        X_physics.isna().mean().mul(100).round(2)
    ),
})

print("\nPHYSICS DESCRIPTOR COVERAGE — ALL MEASUREMENT ROWS")
print(coverage.to_string())

print("\nMATCHING SUMMARY")
print(
    physics_alignment["match_status"]
    .value_counts()
    .to_string()
)

print("\nFULL-DATA FEATURE SETS")
print("A:", X_A.shape)
print("B:", X_B.shape)

print("\nPHYSICS-ELIGIBLE MODEL C")
print("C:", X_C.shape)

print("\nMATCHED A/B/C COMPARISON")
for name, matrix in feature_sets_comparison.items():
    print(f"{name}: {matrix.shape}")

print("\nEligible measurement rows:", len(X_C))
print("Eligible composition groups:", groups_C.nunique())
print(
    "Eligible source identifiers:",
    data.loc[physics_comparison_mask, "paper_id"].nunique(),
)

unresolved_matches = physics_alignment.loc[
    physics_alignment["match_status"].isin([
        "unmatched",
        "ambiguous assignments",
    ])
].copy()

if not unresolved_matches.empty:
    print("\nUNRESOLVED MATCHES — FIRST 20")
    print(
        unresolved_matches[
            [
                "excel_row",
                "composition_ids",
                "match_status",
                "issues",
            ]
        ].head(20)
    )

if doi_mapping_issues:
    print("\nSOURCE MAPPINGS TO REVIEW")
    print(pd.DataFrame(doi_mapping_issues))

print("\nCOMPOSITION CALCULATION PREVIEW")
print(
    physics_by_composition[
        [
            "composition_id",
            "recorded_coordination_numbers",
            *physics_columns,
            "issues",
        ]
    ].head(10)
)

print("\nSpin state was not read or used.")
print("CN5, CN6 and CN8 radius assignments are accepted.")
print("Grain size and relative density are excluded from A/B/C.")
print("Activation energy is excluded from conductivity inputs.")
print("Original formulas and assigned radii were not changed.")
print("No imputation, scaling, SHAP selection or training performed.")

if X_C.empty:
    print(
        "\nNo rows qualify for C. "
        "Review matching and descriptor-input issues."
    )







# %% Block 6: Checking missing-value percentages

def missing_value_report(frame):
    """Return missing-value counts and percentages for each column."""
    missing = frame.isna().sum()

    percentage = (
        missing.div(len(frame)).mul(100)
        if len(frame) > 0
        else pd.Series(np.nan, index=frame.columns)
    )

    return pd.DataFrame({
        "Missing count": missing,
        "Missing (%)": percentage.round(2),
        "Available count": frame.notna().sum(),
    }).sort_values(
        "Missing count",
        ascending=False,
        kind="stable",
    )


# 1. Physics descriptors across ALL measurement rows
physics_missing_report = missing_value_report(X_physics)

print("PHYSICS DESCRIPTORS — ALL MEASUREMENT ROWS")
print("Total rows:", len(X_physics))
print(physics_missing_report.to_string())


# 2. Overall missingness for each full feature set
full_feature_sets = {
    "A": X_A,
    "B": X_B,
    "C": X_C_all,
}

summary_rows = []
missing_reports = {}

for name, frame in full_feature_sets.items():
    missing_reports[name] = missing_value_report(frame)

    missing_cells = int(frame.isna().sum().sum())
    total_cells = frame.size
    rows_with_missing = int(frame.isna().any(axis=1).sum())

    summary_rows.append({
        "Feature set": name,
        "Rows": len(frame),
        "Features": frame.shape[1],
        "Features with missing values": int(
            frame.isna().any(axis=0).sum()
        ),
        "Missing cells (%)": (
            round(100 * missing_cells / total_cells, 2)
            if total_cells > 0 else np.nan
        ),
        "Rows with missing values (%)": (
            round(100 * rows_with_missing / len(frame), 2)
            if len(frame) > 0 else np.nan
        ),
    })

missing_summary = pd.DataFrame(summary_rows).set_index("Feature set")

print("\nFULL FEATURE-SET SUMMARY")
print(missing_summary.to_string())


# 3. Column-level missingness in feature set C
# C includes chemical, temperature, processing and physics inputs.
missing_columns_C = missing_reports["C"].loc[
    missing_reports["C"]["Missing count"] > 0
]

print("\nFEATURE SET C — COLUMNS WITH MISSING VALUES")

if missing_columns_C.empty:
    print("No missing values.")
else:
    print(missing_columns_C.to_string())


# 4. Check the common A/B/C comparison subset separately
print("\nCOMMON COMPARISON SUBSET")

for name, frame in feature_sets_comparison.items():
    print(f"\nSet {name}: {len(frame)} rows")

    if frame.empty:
        print("No eligible rows. Missing percentages are undefined.")
        continue

    report = missing_value_report(frame)
    affected = report.loc[report["Missing count"] > 0]

    if affected.empty:
        print("No missing values.")
    else:
        print(affected.to_string())

print("\nCheck complete. No data were changed.")







# %% BLOCK 7 — REPRODUCIBLE COMPOSITION-GROUPED DATA SPLITTING

#
# Full-data analysis:
#   A: Chemical descriptors + operating temperature
#   B: Set A + sintering temperature and time
#
# Controlled comparison:
#   A, B and C restricted to the same physics-complete rows
#
# IMPORTANT:
# Splitting occurs before fitting imputation, scaling, SHAP or models.

# ============================================================
# 7.1 SETTINGS
# ============================================================

RANDOM_STATE = 42

TEST_FRACTION = 0.10
VALIDATION_FRACTION = 0.10

N_CV_SPLITS = 5

# New filename: does not overwrite the old subset-only split.
SPLIT_MANIFEST_PATH = os.path.join(
    DATASET_FOLDER,
    "full_dataset_composition_split_manifest.csv",
)

if not 0 < TEST_FRACTION + VALIDATION_FRACTION < 1:
    raise ValueError("Training fraction must be greater than zero.")


# ============================================================
# 7.2 PREPARING AND CHECKING THE TWO ANALYSIS GROUPS
# ============================================================

full_features = {
    "A": X_A.copy(),
    "B": X_B.copy(),
}

comparison_features = {
    name: feature_sets_comparison[name].copy()
    for name in ("A", "B", "C")
}

target_all = y_conductivity.copy()
groups_all = data["composition_group"].copy()

full_index = data.index.copy()
comparison_index = comparison_features["C"].index.copy()

if not full_index.is_unique:
    raise ValueError("The main dataset index must be unique.")

if not target_all.index.equals(full_index):
    raise ValueError("The target is not aligned with the main dataset.")

if not groups_all.index.equals(full_index):
    raise ValueError("Composition groups are not aligned with the dataset.")

if target_all.isna().any() or groups_all.isna().any():
    raise ValueError("Targets and composition groups must not be missing.")

if not np.isfinite(target_all.to_numpy(dtype=float)).all():
    raise ValueError("The conductivity target contains non-finite values.")

if not comparison_index.is_unique:
    raise ValueError("The comparison subset index must be unique.")

if not comparison_index.isin(full_index).all():
    raise ValueError("Comparison rows must belong to the full dataset.")

for name, frame in full_features.items():
    if not frame.index.equals(full_index):
        raise ValueError(f"Full-data Set {name} is not aligned.")

for name, frame in comparison_features.items():
    if not frame.index.equals(comparison_index):
        raise ValueError(f"Comparison Set {name} is not aligned.")

    if frame.isna().any().any():
        raise ValueError(
            f"Comparison Set {name} contains missing values. "
            "Review Blocks 5 and 6."
        )

# A/B must contain identical feature values in both analyses.
for name in ("A", "B"):
    pd.testing.assert_frame_equal(
        full_features[name].loc[comparison_index],
        comparison_features[name],
    )

print("DATA AVAILABLE BEFORE SPLITTING")

print(
    f"Full dataset: {len(full_index)} rows; "
    f"{groups_all.nunique()} composition groups"
)

print(
    f"Physics-complete subset: {len(comparison_index)} rows; "
    f"{groups_all.loc[comparison_index].nunique()} composition groups"
)


# ============================================================
# 7.3 CREATING STABLE ROW IDENTIFIERS AND DATA FINGERPRINTS
# ============================================================

if data["excel_row"].isna().any():
    raise ValueError("Excel row references must not be missing.")

row_ids = data["excel_row"].astype(int).astype(str)

if not row_ids.is_unique:
    raise ValueError("Excel row references must be unique.")

# Fingerprints detect changed records when reusing a saved split.
# These hashes do not participate in splitting or model training.
fingerprint_columns = [
    "formula",
    "paper_id",
    "composition_group",
    "temperature_K",
    "sintering_temperature_K",
    "sintering_time_h",
]

fingerprint_inputs = data[fingerprint_columns].copy()
fingerprint_inputs["target"] = target_all

row_fingerprints = pd.util.hash_pandas_object(
    fingerprint_inputs,
    index=False,
).astype(str)

manifest_current = pd.DataFrame(
    {
        "row_id": row_ids.to_numpy(),
        "composition_group": groups_all.astype(str).to_numpy(),
        "row_fingerprint": row_fingerprints.to_numpy(),
    }
)

identity_columns = [
    "row_id",
    "composition_group",
    "row_fingerprint",
]

manifest_is_new = not os.path.isfile(SPLIT_MANIFEST_PATH)


# ============================================================
# 7.4 REUSING THE SAVED SPLIT OR CREATING A NEW MASTER SPLIT
# ============================================================

if not manifest_is_new:

    manifest = pd.read_csv(
        SPLIT_MANIFEST_PATH,
        dtype=str,
        keep_default_na=False,
    )

    required_columns = identity_columns + ["partition"]

    missing_columns = set(required_columns) - set(manifest.columns)

    if missing_columns:
        raise ValueError(
            f"The saved split is missing columns: {missing_columns}"
        )

    if manifest["row_id"].duplicated().any():
        raise ValueError("The saved split contains duplicate row IDs.")

    expected_identity = (
        manifest_current[identity_columns]
        .sort_values("row_id")
        .reset_index(drop=True)
    )

    saved_identity = (
        manifest[identity_columns]
        .sort_values("row_id")
        .reset_index(drop=True)
    )

    if not expected_identity.equals(saved_identity):
        raise ValueError(
            "The current dataset differs from the saved split manifest. "
            "Review the dataset changes before proceeding. "
            "Do not overwrite the existing split automatically. "
            "If a new dataset version is intentional, use a new "
            "versioned manifest filename."
        )

    print("\nReusing the saved master split.")

else:

    # First: hold out approximately 10% of composition groups.
    test_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_FRACTION,
        random_state=RANDOM_STATE,
    )

    train_val_positions, test_positions = next(
        test_splitter.split(
            np.zeros((len(full_index), 1)),
            groups=groups_all.to_numpy(),
        )
    )

    # Second: take validation groups from the remaining pool.
    # 0.10 / 0.90 gives approximately 10% of the original groups.
    relative_validation_fraction = (
        VALIDATION_FRACTION / (1.0 - TEST_FRACTION)
    )

    validation_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=relative_validation_fraction,
        random_state=RANDOM_STATE + 1,
    )

    train_local, validation_local = next(
        validation_splitter.split(
            np.zeros((len(train_val_positions), 1)),
            groups=groups_all.iloc[train_val_positions].to_numpy(),
        )
    )

    train_positions = train_val_positions[train_local]
    validation_positions = train_val_positions[validation_local]

    partition_values = np.full(
        len(full_index),
        "",
        dtype=object,
    )

    partition_values[train_positions] = "train"
    partition_values[validation_positions] = "validation"
    partition_values[test_positions] = "test"

    manifest = manifest_current.copy()
    manifest["partition"] = partition_values

    print("\nCreated a new master split.")


# Restore partition labels in the current dataset order.
partition_lookup = manifest.set_index("row_id")["partition"]

partition_all = pd.Series(
    row_ids.map(partition_lookup).to_numpy(),
    index=full_index,
    name="partition",
)

if partition_all.isna().any():
    raise ValueError("Some dataset rows have no assigned partition.")

if set(partition_all.unique()) != {"train", "validation", "test"}:
    raise ValueError("The split must contain train, validation and test.")


# ============================================================
# 7.5 VERIFYING COMPOSITION SEPARATION
# ============================================================

group_partition_check = pd.DataFrame(
    {
        "group": groups_all,
        "partition": partition_all,
    }
)

partitions_per_group = (
    group_partition_check.groupby("group")["partition"].nunique()
)

if partitions_per_group.gt(1).any():
    raise ValueError(
        "Composition leakage detected across dataset partitions."
    )


# ============================================================
# 7.6 INHERITING MASTER PARTITIONS FOR BOTH ANALYSES
# ============================================================

analysis_indices = {
    "full": full_index,
    "comparison": comparison_index,
}

split_indices = {}

for analysis_name, analysis_index in analysis_indices.items():

    labels = partition_all.loc[analysis_index]

    split_indices[analysis_name] = {
        partition: analysis_index[
            labels.eq(partition).to_numpy()
        ]
        for partition in ("train", "validation", "test")
    }

    for partition, indices in split_indices[analysis_name].items():

        if len(indices) < 2:
            raise ValueError(
                f"{analysis_name}/{partition} has fewer than two rows. "
                "This analysis cannot support the planned evaluation."
            )

    train_group_count = groups_all.loc[
        split_indices[analysis_name]["train"]
    ].nunique()

    if train_group_count < N_CV_SPLITS:
        raise ValueError(
            f"{analysis_name} has only {train_group_count} training "
            f"composition groups, fewer than {N_CV_SPLITS} CV folds."
        )


# ============================================================
# 7.7 BUILDING TRAINING AND VALIDATION OBJECTS
# ============================================================

analysis_features = {
    "full": full_features,
    "comparison": comparison_features,
}

# Access examples:
# X_train["full"]["A"]
# X_train["full"]["B"]
# X_train["comparison"]["A"]
# X_train["comparison"]["B"]
# X_train["comparison"]["C"]

X_train = {}
X_validation = {}

y_train = {}
y_validation = {}

groups_train = {}

# Store test indices only; do not evaluate test outcomes here.
locked_test_indices = {}

for analysis_name, feature_dictionary in analysis_features.items():

    train_idx = split_indices[analysis_name]["train"]
    validation_idx = split_indices[analysis_name]["validation"]

    X_train[analysis_name] = {
        name: frame.loc[train_idx].copy()
        for name, frame in feature_dictionary.items()
    }

    X_validation[analysis_name] = {
        name: frame.loc[validation_idx].copy()
        for name, frame in feature_dictionary.items()
    }

    y_train[analysis_name] = target_all.loc[train_idx].copy()
    y_validation[analysis_name] = target_all.loc[validation_idx].copy()

    groups_train[analysis_name] = groups_all.loc[train_idx].copy()

    locked_test_indices[analysis_name] = (
        split_indices[analysis_name]["test"].copy()
    )


# ============================================================
# 7.8 CREATING SHARED GROUPED CV FOLDS WITHIN MASTER TRAINING DATA
# ============================================================

master_train_idx = split_indices["full"]["train"]

master_cv = GroupKFold(n_splits=N_CV_SPLITS)

# Each training composition receives one held-out CV-fold number.
master_fold_assignment = pd.Series(
    -1,
    index=master_train_idx,
    dtype=int,
    name="cv_fold",
)

master_fold_positions = list(
    master_cv.split(
        X_train["full"]["A"],
        y_train["full"],
        groups=groups_train["full"],
    )
)

for fold_number, (_, holdout_positions) in enumerate(
    master_fold_positions,
    start=1,
):
    holdout_idx = master_train_idx.take(holdout_positions)
    master_fold_assignment.loc[holdout_idx] = fold_number

if master_fold_assignment.eq(-1).any():
    raise ValueError("Some training rows were not assigned a CV fold.")

# Restrict the same fold assignments to the comparison subset.
# All feature sets within an analysis use exactly the same folds.
cv_splits = {}
cv_summary_rows = []

for analysis_name in analysis_features:

    training_index = y_train[analysis_name].index

    fold_labels = master_fold_assignment.loc[
        training_index
    ].to_numpy()

    cv_splits[analysis_name] = []

    for fold_number in range(1, N_CV_SPLITS + 1):

        fit_positions = np.flatnonzero(
            fold_labels != fold_number
        )

        holdout_positions = np.flatnonzero(
            fold_labels == fold_number
        )

        if len(fit_positions) < 2 or len(holdout_positions) < 2:
            raise ValueError(
                f"{analysis_name}, CV fold {fold_number}: "
                "too few rows after restricting the master folds. "
                "Review the number of CV folds before modelling; "
                "do not search seeds based on model performance."
            )

        fit_groups = set(
            groups_train[analysis_name].iloc[fit_positions]
        )

        holdout_groups = set(
            groups_train[analysis_name].iloc[holdout_positions]
        )

        if fit_groups.intersection(holdout_groups):
            raise ValueError("Composition leakage within a CV fold.")

        cv_splits[analysis_name].append(
            (fit_positions, holdout_positions)
        )

        cv_summary_rows.append(
            {
                "Analysis": analysis_name,
                "Fold": fold_number,
                "Fit rows": len(fit_positions),
                "Held-out rows": len(holdout_positions),
                "Fit groups": len(fit_groups),
                "Held-out groups": len(holdout_groups),
            }
        )


# ============================================================
# 7.9 SAVING THE NEW MANIFEST ONLY AFTER CHECKS PASS
# ============================================================

if manifest_is_new:

    # Exclusive creation prevents overwriting an existing file.
    manifest.to_csv(
        SPLIT_MANIFEST_PATH,
        index=False,
        mode="x",
    )

    print("\nSaved master split:")
    print(SPLIT_MANIFEST_PATH)


# ============================================================
# 7.10 REPORTING PARTITION SIZES — WITHOUT TEST PERFORMANCE
# ============================================================

split_summary_rows = []

for analysis_name, partitions in split_indices.items():

    total_rows = len(analysis_indices[analysis_name])

    for partition, indices in partitions.items():

        split_summary_rows.append(
            {
                "Analysis": analysis_name,
                "Partition": partition,
                "Rows": len(indices),
                "Rows (%)": round(
                    100 * len(indices) / total_rows,
                    2,
                ),
                "Composition groups": (
                    groups_all.loc[indices].nunique()
                ),
                "Source identifiers": (
                    data.loc[indices, "paper_id"].nunique()
                ),
            }
        )

split_summary = pd.DataFrame(split_summary_rows)
cv_summary = pd.DataFrame(cv_summary_rows)

print("\nDATASET PARTITIONS")
print(split_summary.to_string(index=False))

print("\nTRAINING-ONLY GROUPED CROSS-VALIDATION")
print(cv_summary.to_string(index=False))

print("\nTRAINING FEATURE MATRICES")

for analysis_name, feature_dictionary in X_train.items():
    for name, frame in feature_dictionary.items():
        print(
            f"{analysis_name} — Set {name}: "
            f"{frame.shape[0]} rows, {frame.shape[1]} features"
        )


print("Composition groups do not cross train/validation/test.")
print("The comparison subset inherits the master partitions.")
print("All paired feature sets use identical CV folds.")
print("No preprocessing, SHAP selection or model fitting performed.")
print("Test data remain reserved for final evaluation.")







# %% BLOCK 8 — PREPROCESSING AND TRAINING-FOLD SHAP SELECTION


# ============================================================
# 8.1 SETTINGS
# ============================================================

TOP_N_FEATURES = 30

# Fixed selector settings, shared across feature sets and models.
SHAP_SELECTOR_TREES = 200
SHAP_SELECTOR_MAX_DEPTH = 12
SHAP_SELECTOR_MIN_LEAF = 2

# Limit explanation cost.
# The selector RF still trains on the entire fitting fold.
SHAP_MAX_EXPLANATION_ROWS = 300

# Using one thread inside the selector.
# Later, hyperparameter search can parallelize across fits.
SHAP_SELECTOR_N_JOBS = 1


# ============================================================
# 8.2 DEFINING A SCIKIT-LEARN-COMPATIBLE SHAP SELECTOR
# ============================================================

class TrainingFoldSHAPSelector(TransformerMixin, BaseEstimator):
    """
    Fitting an RF on the supplied training data and select features
    by their mean absolute Tree SHAP values.

    When placed inside a CV pipeline, fit() sees only that
    fold's fitting data.

    transform() applies the saved feature list without refitting.
    """

    def __init__(
        self,
        n_features=30,
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        max_explanation_rows=300,
        random_state=42,
        n_jobs=1,
    ):
        self.n_features = n_features
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_explanation_rows = max_explanation_rows
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):

        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "The SHAP selector requires a pandas DataFrame "
                "to preserve feature names."
            )

        if not X.columns.is_unique:
            raise ValueError("Feature names must be unique.")

        if self.n_features < 1:
            raise ValueError("n_features must be at least 1.")

        if self.max_explanation_rows < 1:
            raise ValueError(
                "max_explanation_rows must be at least 1."
            )

        if X.shape[1] == 0:
            raise ValueError("No features remain for selection.")

        values = X.to_numpy(dtype=float)
        target = np.asarray(y, dtype=float)

        if target.ndim != 1 or len(target) != len(X):
            raise ValueError(
                "A single regression target aligned with X is required."
            )

        if isinstance(y, pd.Series) and not X.index.equals(y.index):
            raise ValueError("X and y indices are not aligned.")

        if not np.isfinite(values).all():
            raise ValueError(
                "SHAP inputs contain missing or infinite values."
            )

        if not np.isfinite(target).all():
            raise ValueError("The target contains non-finite values.")

        self.feature_names_in_ = np.asarray(
            X.columns,
            dtype=object,
        )
        self.n_features_in_ = X.shape[1]

        # This RF is the feature selector, not necessarily
        # the final conductivity prediction model.
        self.selector_model_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

        self.selector_model_.fit(X, target)

        # Sample only from the data supplied to this fit().
        rng = np.random.default_rng(self.random_state)

        explanation_count = min(
            len(X),
            self.max_explanation_rows,
        )

        explanation_positions = np.sort(
            rng.choice(
                len(X),
                size=explanation_count,
                replace=False,
            )
        )

        X_explain = X.iloc[explanation_positions]

        explainer = shap.TreeExplainer(
            self.selector_model_,
            feature_perturbation="tree_path_dependent",
            model_output="raw",
        )

        shap_values = np.asarray(
            explainer.shap_values(
                X_explain,
                check_additivity=True,
            )
        )

        # Support a singleton output dimension.
        if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
            shap_values = shap_values[:, :, 0]

        if shap_values.shape != X_explain.shape:
            raise ValueError(
                f"Unexpected SHAP shape: {shap_values.shape}; "
                f"expected {X_explain.shape}."
            )

        if not np.isfinite(shap_values).all():
            raise ValueError("SHAP returned non-finite values.")

        importance = np.abs(shap_values).mean(axis=0)

        # Stable ordering makes tied rankings reproducible.
        ranked_positions = np.argsort(
            -importance,
            kind="stable",
        )

        self.n_features_selected_ = min(
            self.n_features,
            self.n_features_in_,
        )

        self.selected_features_ = self.feature_names_in_[
            ranked_positions[:self.n_features_selected_]
        ]

        self.feature_importances_ = pd.Series(
            importance,
            index=self.feature_names_in_,
            name="mean_absolute_SHAP",
        )

        self.ranking_ = pd.DataFrame(
            {
                "feature": self.feature_names_in_[ranked_positions],
                "mean_absolute_SHAP": importance[ranked_positions],
            }
        )

        self.ranking_.insert(
            0,
            "rank",
            np.arange(1, len(self.ranking_) + 1),
        )

        self.ranking_["selected"] = (
            self.ranking_["feature"].isin(self.selected_features_)
        )

        self.explanation_rows_ = explanation_count

        return self

    def transform(self, X):

        check_is_fitted(self, "selected_features_")

        if not isinstance(X, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame.")

        if not np.array_equal(
            np.asarray(X.columns, dtype=object),
            self.feature_names_in_,
        ):
            raise ValueError(
                "Input feature names or order differ from fitting."
            )

        return X.loc[:, self.selected_features_].copy()

    def get_feature_names_out(self, input_features=None):

        check_is_fitted(self, "selected_features_")

        if input_features is not None:
            if not np.array_equal(
                np.asarray(input_features, dtype=object),
                self.feature_names_in_,
            ):
                raise ValueError("Unexpected input feature names.")

        return self.selected_features_.copy()


# ============================================================
# 8.3 PIPELINE FACTORY
# ============================================================

def make_regression_pipeline(estimator, use_shap=True):
    """
    Return a NEW, unfitted pipeline.

    Always pass the whole pipeline to the hyperparameter search.
    Do not preprocess the full training set before CV.
    """

    imputer = SimpleImputer(
        strategy="median",
        keep_empty_features=True,
    ).set_output(transform="pandas")

    constant_filter = VarianceThreshold(
        threshold=0.0,
    ).set_output(transform="pandas")

    scaler = StandardScaler().set_output(
        transform="pandas"
    )

    if use_shap:
        selector = TrainingFoldSHAPSelector(
            n_features=TOP_N_FEATURES,
            n_estimators=SHAP_SELECTOR_TREES,
            max_depth=SHAP_SELECTOR_MAX_DEPTH,
            min_samples_leaf=SHAP_SELECTOR_MIN_LEAF,
            max_explanation_rows=SHAP_MAX_EXPLANATION_ROWS,
            random_state=RANDOM_STATE,
            n_jobs=SHAP_SELECTOR_N_JOBS,
        )
    else:
        # Optional all-nonconstant-feature baseline.
        selector = "passthrough"

    return Pipeline(
        steps=[
            ("imputer", imputer),
            ("constant_filter", constant_filter),
            ("scaler", scaler),
            ("shap_selector", selector),
            ("model", clone(estimator)),
        ]
    )


# ============================================================
# 8.4 CONFIRMING THE AVAILABLE TRAINING INPUTS
# ============================================================

training_summary_rows = []

for analysis_name, feature_dictionary in X_train.items():

    for feature_set, frame in feature_dictionary.items():

        training_summary_rows.append(
            {
                "Analysis": analysis_name,
                "Feature set": feature_set,
                "Training rows": len(frame),
                "Input features": frame.shape[1],
                "Maximum selected features": min(
                    TOP_N_FEATURES,
                    frame.shape[1],
                ),
                "CV folds": len(cv_splits[analysis_name]),
            }
        )

training_input_summary = pd.DataFrame(training_summary_rows)

print("TRAINING INPUTS")
print(training_input_summary.to_string(index=False))


#
# Creates five figures:
#   Full dataset: A, B
#   Common comparison subset: A, B, C
#
# Validation and test data are not used.

os.makedirs(FIGURES_FOLDER, exist_ok=True)

# Keeping diagnostic results separate from modelling pipelines.
shap_diagnostic_rankings = {}
shap_diagnostic_preprocessors = {}

for analysis_name, feature_dictionary in X_train.items():

    for feature_set, X_training in feature_dictionary.items():

        print(
            f"\nCalculating training SHAP importance: "
            f"{analysis_name}, Set {feature_set}"
        )

        # Construct a fresh pipeline and retain only preprocessing
        # and feature selection, excluding the final regressor.
        diagnostic_pipeline = make_regression_pipeline(
            RandomForestRegressor(random_state=RANDOM_STATE),
            use_shap=True,
        )[:-1]

        diagnostic_pipeline.fit(
            X_training,
            y_train[analysis_name],
        )

        selector = diagnostic_pipeline.named_steps["shap_selector"]

        ranking = selector.ranking_.copy()

        key = (analysis_name, feature_set)

        shap_diagnostic_rankings[key] = ranking
        shap_diagnostic_preprocessors[key] = diagnostic_pipeline

        # Plotting only the selected top features.
        plot_data = (
            ranking.loc[ranking["selected"]]
            .sort_values("mean_absolute_SHAP", ascending=True)
            .copy()
        )

        # Shortening labels for display only.
        # Original feature names remain unchanged.
        labels = (
            plot_data["feature"]
            .str.replace("chem__", "", regex=False)
            .str.replace("MagpieData ", "", regex=False)
        )

        # Distinguishing chemical, processing, temperature and physics.
        def feature_color(feature):
            if feature.startswith("phys_"):
                return "#D55E00"
            if feature in (
                "sintering_temperature_K",
                "sintering_time_h",
            ):
                return "#009E73"
            if feature == "temperature_K":
                return "#CC79A7"
            return "#0072B2"

        colors = [
            feature_color(feature)
            for feature in plot_data["feature"]
        ]

        figure_height = max(6, 0.30 * len(plot_data) + 2)

        fig, ax = plt.subplots(
            figsize=(12, figure_height),
            layout="constrained",
        )

        ax.barh(
            labels,
            plot_data["mean_absolute_SHAP"],
            color=colors,
        )

        analysis_label = (
            "Full-data analysis"
            if analysis_name == "full"
            else "Common-subset comparison"
        )

        ax.set_title(
            f"{analysis_label} — Set {feature_set}\n"
            f"Training RF selector: top {len(plot_data)} features",
            fontsize=13,
        )

        ax.set_xlabel(
            "Mean absolute SHAP value "
            "(log10 conductivity prediction units)"
        )

        ax.set_ylabel("")
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="x", alpha=0.20)
        ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        from matplotlib.patches import Patch

        legend_items = [
            Patch(color="#0072B2", label="Chemical"),
            Patch(color="#CC79A7", label="Operating temperature"),
            Patch(color="#009E73", label="Processing"),
            Patch(color="#D55E00", label="Physics"),
        ]

        ax.legend(
            handles=legend_items,
            loc="lower right",
            fontsize=8,
            frameon=True,
        )

        filename_base = (
            f"SHAP_training_{analysis_name}_"
            f"set_{feature_set}_top{len(plot_data)}"
        )

        png_path = os.path.join(
            FIGURES_FOLDER,
            filename_base + ".png",
        )

        pdf_path = os.path.join(
            FIGURES_FOLDER,
            filename_base + ".pdf",
        )

        fig.savefig(
            png_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )

        fig.savefig(
            pdf_path,
            bbox_inches="tight",
            facecolor="white",
        )

        plt.show()
        plt.close(fig)

        print(f"Training rows: {len(X_training)}")
        print(f"Rows explained: {selector.explanation_rows_}")
        print(f"Selected features: {selector.n_features_selected_}")
        print(f"Saved PNG: {png_path}")
        print(f"Saved PDF: {pdf_path}")

print("\nTraining SHAP figures completed.")
print("Original X_train inputs remain unchanged.")
print("Validation and test data were not used.")

print("\nBLOCK 8 COMPLETE")
print("Preprocessing and SHAP selector classes are ready.")
print("No preprocessing parameters have been fitted.")
print("No features have been selected yet.")
print("Validation and test data have not been used.")
print("Block 9 will fit these steps inside grouped CV.")




# %% OUTPUT FOLDERS


FIGURES_FOLDER = os.path.join(PROJECT_FOLDER, "Figures")
RESULTS_FOLDER = os.path.join(PROJECT_FOLDER, "Results")

os.makedirs(FIGURES_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

print("Figures:", FIGURES_FOLDER)
print("Results:", RESULTS_FOLDER)

# Saving tables already generated.
split_summary.to_csv(
    os.path.join(RESULTS_FOLDER, "dataset_split_summary.csv"),
    index=False,
)

cv_summary.to_csv(
    os.path.join(RESULTS_FOLDER, "training_cv_fold_summary.csv"),
    index=False,
)

# Saving training SHAP rankings.
if "shap_diagnostic_rankings" in globals():
    for (analysis, feature_set), ranking in (
        shap_diagnostic_rankings.items()
    ):
        ranking.to_csv(
            os.path.join(
                RESULTS_FOLDER,
                f"SHAP_training_{analysis}_set_{feature_set}.csv",
            ),
            index=False,
        )

print("Available result tables saved.")













# %% BLOCK 9 — GROUPED HYPERPARAMETER TUNING

#
# Analyses:
#   Full data: A, B
#   Common physics-complete subset: A, B, C
#
# Validation and test data are not used.
#





# ============================================================
# 9.1 SETTINGS AND OUTPUT LOCATIONS
# ============================================================

FIGURES_FOLDER = os.path.join(PROJECT_FOLDER, "Figures")
RESULTS_FOLDER = os.path.join(PROJECT_FOLDER, "Results")

os.makedirs(FIGURES_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Fixed budget per model and feature set.
N_SEARCH_ITERATIONS = 20

# Sequential fitting is easier to debug in VS Code and avoids
# nested parallelism. The models below also use one worker.
SEARCH_N_JOBS = 1

# A separate run folder preserves previous results.
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

TUNING_RESULTS_FOLDER = os.path.join(
    RESULTS_FOLDER,
    f"block9_{RUN_ID}",
)

os.makedirs(TUNING_RESULTS_FOLDER, exist_ok=False)

# Cache fitted preprocessing for identical training-fold inputs.
# Different folds have different cache keys.
# This avoids recalculating the same SHAP selector for every
# regressor hyperparameter configuration.
CACHE_FOLDER = os.path.join(
    TUNING_RESULTS_FOLDER,
    "_pipeline_cache",
)

pipeline_memory = memory(
    location=CACHE_FOLDER,
    verbose=0,
)

SCORING = {
    "RMSE": "neg_root_mean_squared_error",
    "MAE": "neg_mean_absolute_error",
    "R2": "r2",
}


# ============================================================
# 9.2 CHECKING TRAINING FOLDS AND DEFINING VALID KNN SETTINGS
# ============================================================

minimum_fold_training_size = min(
    len(fit_positions)
    for folds in cv_splits.values()
    for fit_positions, _ in folds
)

knn_neighbors = [
    k for k in (3, 5, 7, 11, 15, 21)
    if k <= minimum_fold_training_size
]

if not knn_neighbors:
    raise ValueError("Too few training rows for the KNN search.")

# R² requires variation in the held-out targets.
for analysis_name, folds in cv_splits.items():
    for fold_number, (_, heldout_positions) in enumerate(
        folds,
        start=1,
    ):
        heldout_target = y_train[analysis_name].iloc[
            heldout_positions
        ]

        if heldout_target.nunique() < 2:
            raise ValueError(
                f"{analysis_name}, fold {fold_number}: "
                "R² is undefined because the target is constant."
            )


# ============================================================
# 9.3 MODELS AND SEARCH SPACES
# ============================================================

model_specs = {
    "Ridge": {
        "estimator": Ridge(),
        "parameters": {
            "model__alpha": loguniform(1e-4, 1e4),
        },
    },

    "SVR": {
        "estimator": SVR(
            kernel="rbf",
            cache_size=512,
        ),
        "parameters": {
            "model__C": loguniform(1e-1, 1e3),
            "model__epsilon": loguniform(1e-3, 3e-1),
            "model__gamma": loguniform(1e-4, 1.0),
        },
    },

    "KNN": {
        "estimator": KNeighborsRegressor(n_jobs=1),
        "parameters": {
            "model__n_neighbors": knn_neighbors,
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
        },
    },

    "RandomForest": {
        "estimator": RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "parameters": {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 6, 12, 20],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__max_features": [0.5, 0.8, 1.0],
        },
    },

    "ExtraTrees": {
        "estimator": ExtraTreesRegressor(
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "parameters": {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 6, 12, 20],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__max_features": [0.5, 0.8, 1.0],
        },
    },

    "GradientBoosting": {
        "estimator": GradientBoostingRegressor(
            random_state=RANDOM_STATE,
        ),
        "parameters": {
            "model__n_estimators": [100, 200, 400],
            "model__learning_rate": loguniform(0.01, 0.2),
            "model__max_depth": [2, 3, 4],
            "model__min_samples_leaf": [2, 5, 10],
            "model__subsample": [0.7, 1.0],
        },
    },

    "XGBoost": {
        "estimator": XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbosity=0,
        ),
        "parameters": {
            "model__n_estimators": [200, 400, 600],
            "model__max_depth": [2, 3, 5, 7],
            "model__learning_rate": loguniform(0.01, 0.2),
            "model__min_child_weight": [1, 3, 7],
            "model__subsample": [0.7, 0.9, 1.0],
            "model__colsample_bytree": [0.7, 0.9, 1.0],
            "model__reg_lambda": loguniform(0.1, 100),
        },
    },

    "ANN_MLP": {
        "estimator": MLPRegressor(
            solver="lbfgs",
            max_iter=2000,
            max_fun=30000,
            early_stopping=False,
            random_state=RANDOM_STATE,
        ),
        "parameters": {
            "model__hidden_layer_sizes": [
                (32,),
                (64,),
                (64, 32),
                (64, 32, 16),
            ],
            "model__activation": ["relu", "tanh"],
            "model__alpha": loguniform(1e-5, 1.0),
        },
    },
}

# L-BFGS is used for the relatively small tabular ANN dataset.
# No internal random validation split is created.
# Architecture and regularization are chosen using grouped CV.


# ============================================================
# 9.4 SAVING RUN SETTINGS AND SOFTWARE VERSIONS
# ============================================================

software_versions = {
    "python": platform.python_version(),
}

for package_name in (
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "shap",
    "xgboost",
    "joblib",
):
    software_versions[package_name] = version(package_name)

run_settings = {
    "run_id": RUN_ID,
    "random_state": RANDOM_STATE,
    "search_iterations": N_SEARCH_ITERATIONS,
    "selection_metric": "Mean grouped-CV RMSE",
    "target": "log10_sigma_S_cm",
    "models": list(model_specs),
    "top_n_features": TOP_N_FEATURES,
    "selector_trees": SHAP_SELECTOR_TREES,
    "selector_max_depth": SHAP_SELECTOR_MAX_DEPTH,
    "selector_min_leaf": SHAP_SELECTOR_MIN_LEAF,
    "selector_explanation_rows": SHAP_MAX_EXPLANATION_ROWS,
    "software_versions": software_versions,
    "validation_used": False,
    "test_used": False,
}

with open(
    os.path.join(TUNING_RESULTS_FOLDER, "run_settings.json"),
    "w",
    encoding="utf-8",
) as file:
    json.dump(run_settings, file, indent=2)

search_space_table = pd.DataFrame(
    [
        {
            "Model": model_name,
            "Parameter": parameter,
            "Search space": (
                str(values)
                if not hasattr(values, "dist")
                else (
                    f"{values.dist.name}; "
                    f"args={values.args}; kwargs={values.kwds}"
                )
            ),
        }
        for model_name, specification in model_specs.items()
        for parameter, values in specification["parameters"].items()
    ]
)

search_space_table.to_csv(
    os.path.join(TUNING_RESULTS_FOLDER, "search_spaces.csv"),
    index=False,
)


# ============================================================
# 9.5 RUNNING GROUPED HYPERPARAMETER SEARCHES
# ============================================================

# Objects retained for validation in Block 10.
searches = {}
best_models = {}

tuning_summary_rows = []
best_fold_rows = []

total_searches = sum(
    len(feature_dictionary)
    for feature_dictionary in X_train.values()
) * len(model_specs)

search_number = 0

print(f"Total model searches: {total_searches}")
print(f"Maximum configurations per search: {N_SEARCH_ITERATIONS}")
print("All searches use training data only.")

for analysis_name, feature_dictionary in X_train.items():

    for feature_set, training_features in feature_dictionary.items():

        training_target = y_train[analysis_name]
        folds = cv_splits[analysis_name]

        for model_name, specification in model_specs.items():

            search_number += 1

            print(
                f"\n[{search_number}/{total_searches}] "
                f"{analysis_name} | Set {feature_set} | {model_name}",
                flush=True,
            )

            key = (analysis_name, feature_set, model_name)

            pipeline = make_regression_pipeline(
                specification["estimator"],
                use_shap=True,
            )

            pipeline.set_params(memory=pipeline_memory)

            parameters = specification["parameters"]

            # Avoid requesting more trials than a finite grid contains.
            if all(
                not hasattr(values, "rvs")
                for values in parameters.values()
            ):
                possible_configurations = int(
                    np.prod([
                        len(values)
                        for values in parameters.values()
                    ])
                )
                iterations = min(
                    N_SEARCH_ITERATIONS,
                    possible_configurations,
                )
            else:
                iterations = N_SEARCH_ITERATIONS

            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=parameters,
                n_iter=iterations,
                scoring=SCORING,
                refit="RMSE",
                cv=folds,
                random_state=RANDOM_STATE,
                n_jobs=SEARCH_N_JOBS,
                return_train_score=False,
                error_score="raise",
                verbose=1,
            )

            start_time = time.perf_counter()

            # Fit the entire pipeline inside every CV fold.
            search.fit(training_features, training_target)

            elapsed_seconds = time.perf_counter() - start_time

            searches[key] = search
            best_models[key] = search.best_estimator_

            best_index = search.best_index_
            results = pd.DataFrame(search.cv_results_)

            stem = (
                f"{analysis_name}_set_{feature_set}_{model_name}"
            )

            # sklearn names inner held-out CV scores "test".
            # These columns do NOT refer to the locked final test set.
            results.to_csv(
                os.path.join(
                    TUNING_RESULTS_FOLDER,
                    f"{stem}_search_results.csv",
                ),
                index=False,
            )

            selected_parameters = json.dumps(
                search.best_params_,
                sort_keys=True,
                default=lambda value: (
                    value.item()
                    if isinstance(value, np.generic)
                    else str(value)
                ),
            )

            summary_row = {
                "Analysis": analysis_name,
                "Feature_set": feature_set,
                "Model": model_name,
                "Training_rows": len(training_features),
                "Training_groups": (
                    groups_train[analysis_name].nunique()
                ),
                "Configurations": iterations,
                "CV_folds": len(folds),
                "CV_RMSE_mean": (
                    -results.loc[best_index, "mean_test_RMSE"]
                ),
                "CV_RMSE_std": (
                    results.loc[best_index, "std_test_RMSE"]
                ),
                "CV_MAE_mean": (
                    -results.loc[best_index, "mean_test_MAE"]
                ),
                "CV_R2_mean": (
                    results.loc[best_index, "mean_test_R2"]
                ),
                "Elapsed_seconds": elapsed_seconds,
                "Best_parameters": selected_parameters,
            }

            tuning_summary_rows.append(summary_row)

            for fold_number in range(len(folds)):
                best_fold_rows.append(
                    {
                        "Analysis": analysis_name,
                        "Feature_set": feature_set,
                        "Model": model_name,
                        "Fold": fold_number + 1,
                        "RMSE": -results.loc[
                            best_index,
                            f"split{fold_number}_test_RMSE",
                        ],
                        "MAE": -results.loc[
                            best_index,
                            f"split{fold_number}_test_MAE",
                        ],
                        "R2": results.loc[
                            best_index,
                            f"split{fold_number}_test_R2",
                        ],
                    }
                )

            # This ranking comes from refitting on ALL training rows.
            # It is not the ranking of every individual CV fold.
            refit_selector = search.best_estimator_.named_steps[
                "shap_selector"
            ]

            refit_selector.ranking_.to_csv(
                os.path.join(
                    TUNING_RESULTS_FOLDER,
                    f"{stem}_training_refit_SHAP_ranking.csv",
                ),
                index=False,
            )

            # Save progress after each completed search.
            pd.DataFrame(tuning_summary_rows).to_csv(
                os.path.join(
                    TUNING_RESULTS_FOLDER,
                    "tuning_summary.csv",
                ),
                index=False,
            )

            pd.DataFrame(best_fold_rows).to_csv(
                os.path.join(
                    TUNING_RESULTS_FOLDER,
                    "best_configuration_fold_scores.csv",
                ),
                index=False,
            )

            print(
                f"CV RMSE: {summary_row['CV_RMSE_mean']:.4f} | "
                f"CV MAE: {summary_row['CV_MAE_mean']:.4f} | "
                f"CV R²: {summary_row['CV_R2_mean']:.4f}",
                flush=True,
            )


# ============================================================
# 9.6 FINALIZING TUNING TABLES
# ============================================================

tuning_summary = pd.DataFrame(tuning_summary_rows)

tuning_summary = tuning_summary.sort_values(
    ["Analysis", "Feature_set", "CV_RMSE_mean"]
).reset_index(drop=True)

best_configuration_fold_scores = pd.DataFrame(best_fold_rows)

tuning_summary.to_csv(
    os.path.join(
        TUNING_RESULTS_FOLDER,
        "tuning_summary.csv",
    ),
    index=False,
)

print("\nGROUPED-CV TUNING SUMMARY")

print(
    tuning_summary[
        [
            "Analysis",
            "Feature_set",
            "Model",
            "CV_RMSE_mean",
            "CV_RMSE_std",
            "CV_MAE_mean",
            "CV_R2_mean",
        ]
    ].round(4).to_string(index=False)
)


# ============================================================
# 9.7 FIGURES: MODEL COMPARISON WITHIN EACH FEATURE SET
# ============================================================

for (analysis_name, feature_set), table in tuning_summary.groupby(
    ["Analysis", "Feature_set"],
    sort=False,
):

    table = table.sort_values("CV_RMSE_mean")
    positions = np.arange(len(table))

    fig, ax = plt.subplots(
        figsize=(9, 6),
        layout="constrained",
    )

    ax.errorbar(
        table["CV_RMSE_mean"],
        positions,
        xerr=table["CV_RMSE_std"],
        fmt="o",
        color="#0072B2",
        ecolor="#777777",
        capsize=4,
        markersize=6,
    )

    ax.set_yticks(positions)
    ax.set_yticklabels(table["Model"])
    ax.invert_yaxis()

    ax.set_xlabel(
        "Grouped-CV RMSE of log10 conductivity\n"
        "Mean ± fold standard deviation; lower is better"
    )

    ax.set_title(
        f"{analysis_name.capitalize()} analysis — Set {feature_set}\n"
        "Hyperparameter-tuning results"
    )

    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    figure_stem = (
        f"Block9_{RUN_ID}_{analysis_name}_"
        f"set_{feature_set}_CV_model_comparison"
    )

    for extension in ("png", "pdf"):
        fig.savefig(
            os.path.join(
                FIGURES_FOLDER,
                f"{figure_stem}.{extension}",
            ),
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()
    plt.close(fig)


# ============================================================
# 9.8 FIGURE: A/B/C ON THE COMMON COMPARISON SUBSET
# ============================================================

comparison_table = (
    tuning_summary.loc[
        tuning_summary["Analysis"].eq("comparison")
    ]
    .pivot(
        index="Model",
        columns="Feature_set",
        values="CV_RMSE_mean",
    )
    .reindex(
        index=list(model_specs),
        columns=["A", "B", "C"],
    )
)

comparison_table.to_csv(
    os.path.join(
        TUNING_RESULTS_FOLDER,
        "common_subset_ABC_CV_RMSE.csv",
    )
)

fig, ax = plt.subplots(
    figsize=(12, 6),
    layout="constrained",
)

x_positions = np.arange(len(comparison_table))
bar_width = 0.25

colors = {
    "A": "#0072B2",
    "B": "#009E73",
    "C": "#D55E00",
}

for offset, feature_set in enumerate(("A", "B", "C")):
    ax.bar(
        x_positions + (offset - 1) * bar_width,
        comparison_table[feature_set],
        width=bar_width,
        color=colors[feature_set],
        label=f"Set {feature_set}",
    )

ax.set_xticks(x_positions)
ax.set_xticklabels(
    comparison_table.index,
    rotation=35,
    ha="right",
)

ax.set_ylabel("Mean grouped-CV RMSE of log10 conductivity")
ax.set_title(
    "Common-subset A/B/C comparison\n"
    "Tuning scores using identical rows and folds"
)

ax.legend()
ax.grid(axis="y", alpha=0.25)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

for extension in ("png", "pdf"):
    fig.savefig(
        os.path.join(
            FIGURES_FOLDER,
            f"Block9_{RUN_ID}_common_subset_ABC_CV_RMSE.{extension}",
        ),
        dpi=300,
        bbox_inches="tight",
    )

plt.show()
plt.close(fig)

print("\nBLOCK 9 COMPLETE")
print("Tuning tables:", TUNING_RESULTS_FOLDER)
print("Figures:", FIGURES_FOLDER)
print("Best fitted pipelines are stored in best_models.")
print("Validation and final test data remain unused.")








# %% BLOCK 10 — VALIDATION EVALUATION AND MODEL SELECTION

#
# Uses the tuned pipelines fitted on TRAINING data in Block 9.
# No refitting, new hyperparameter searches or test evaluation.
#
# Selection rule:
# Lowest validation RMSE within each analysis and feature set.
# Exact ties are resolved alphabetically by model name.

# ============================================================
# 10.1 OUTPUT FOLDERS AND REQUIRED OBJECTS
# ============================================================

required_objects = [
    "best_models",
    "X_validation",
    "y_validation",
    "X_train",
    "data",
]

missing_objects = [
    name for name in required_objects
    if name not in globals()
]

if missing_objects:
    raise RuntimeError(
        f"Missing objects: {missing_objects}. "
        "Run this block in the kernel containing the completed "
        "Block 9 models."
    )

if not best_models:
    raise RuntimeError("No fitted models are available.")

FIGURES_FOLDER = os.path.join(PROJECT_FOLDER, "Figures")
RESULTS_FOLDER = os.path.join(PROJECT_FOLDER, "Results")

os.makedirs(FIGURES_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

VALIDATION_RUN_ID = datetime.now().strftime(
    "%Y%m%d_%H%M%S_%f"
)

VALIDATION_RESULTS_FOLDER = os.path.join(
    RESULTS_FOLDER,
    f"block10_{VALIDATION_RUN_ID}",
)

# Creating the results directory after the completeness checks below.


# ============================================================
# 10.2 VERIFYING THAT COMPLETED MODELS COVER ALL FIVE DATASETS
# ============================================================

expected_pairs = {
    (analysis, feature_set)
    for analysis, feature_dictionary in X_validation.items()
    for feature_set in feature_dictionary
}

available_pairs = {
    (analysis, feature_set)
    for analysis, feature_set, model_name in best_models
}

if available_pairs != expected_pairs:
    raise ValueError(
        "The fitted models do not cover exactly the expected "
        "analysis/feature-set combinations. "
        f"Missing: {expected_pairs - available_pairs}; "
        f"unexpected: {available_pairs - expected_pairs}"
    )

completed_model_names = sorted({
    model_name
    for _, _, model_name in best_models
})

# Every evaluated model must have completed all five datasets.
incomplete_models = {}

for model_name in completed_model_names:

    completed_pairs = {
        (analysis, feature_set)
        for analysis, feature_set, name in best_models
        if name == model_name
    }

    missing_pairs = expected_pairs - completed_pairs

    if missing_pairs:
        incomplete_models[model_name] = sorted(missing_pairs)

if incomplete_models:
    raise ValueError(
        "Some searches are only partly complete. Finish them "
        "before making paired comparisons. Missing combinations: "
        f"{incomplete_models}"
    )

print("Models available:", ", ".join(completed_model_names))
print("Number of models:", len(completed_model_names))
print("Number of fitted pipelines:", len(best_models))

# Report models specified but not present, without claiming
# they were evaluated.
if "model_specs" in globals():

    not_evaluated = sorted(
        set(model_specs) - set(completed_model_names)
    )

    if not_evaluated:
        print(
            "\nSpecified models absent from this evaluation:",
            ", ".join(not_evaluated),
        )

os.makedirs(VALIDATION_RESULTS_FOLDER, exist_ok=False)


# ============================================================
# 10.3 VERIFYING VALIDATION ALIGNMENT AND COMPOSITION SEPARATION
# ============================================================

for analysis, feature_dictionary in X_validation.items():

    target = y_validation[analysis]

    if len(target) < 2:
        raise ValueError(
            f"{analysis}: too few validation observations."
        )

    if not np.isfinite(target.to_numpy(dtype=float)).all():
        raise ValueError(
            f"{analysis}: invalid validation target values."
        )

    for feature_set, frame in feature_dictionary.items():

        if not frame.index.equals(target.index):
            raise ValueError(
                f"{analysis}/{feature_set}: X/y index mismatch."
            )

        train_frame = X_train[analysis][feature_set]

        if not frame.columns.equals(train_frame.columns):
            raise ValueError(
                f"{analysis}/{feature_set}: feature names or "
                "column order differ from training."
            )

        train_groups = set(
            data.loc[train_frame.index, "composition_group"]
        )

        validation_groups = set(
            data.loc[frame.index, "composition_group"]
        )

        if train_groups.intersection(validation_groups):
            raise ValueError(
                f"{analysis}/{feature_set}: composition leakage."
            )


# ============================================================
# 10.4 GENERATING VALIDATION PREDICTIONS — NO FIT CALLS
# ============================================================

validation_metric_rows = []
validation_prediction_frames = []

for key in sorted(best_models):

    analysis, feature_set, model_name = key

    fitted_pipeline = best_models[key]

    features = X_validation[analysis][feature_set]
    target = y_validation[analysis]

    predictions = np.asarray(
        fitted_pipeline.predict(features),
        dtype=float,
    ).reshape(-1)

    if len(predictions) != len(target):
        raise ValueError(f"Prediction length mismatch: {key}")

    if not np.isfinite(predictions).all():
        raise ValueError(f"Non-finite predictions: {key}")

    observed = target.to_numpy(dtype=float)

    # Residual convention used throughout this block:
    # predicted minus measured.
    residuals = predictions - observed

    rmse = np.sqrt(
        mean_squared_error(observed, predictions)
    )

    mae = mean_absolute_error(observed, predictions)

    # R² is undefined for a constant target.
    r2 = (
        r2_score(observed, predictions)
        if target.nunique() > 1
        else np.nan
    )

    validation_metric_rows.append({
        "Analysis": analysis,
        "Feature_set": feature_set,
        "Model": model_name,
        "Validation_rows": len(target),
        "Validation_groups": data.loc[
            target.index, "composition_group"
        ].nunique(),
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Mean_residual": residuals.mean(),
    })

    prediction_table = data.loc[
        target.index,
        [
            "excel_row",
            "formula",
            "paper_id",
            "composition_group",
            "temperature_K",
        ],
    ].copy()

    prediction_table.insert(0, "Analysis", analysis)
    prediction_table.insert(1, "Feature_set", feature_set)
    prediction_table.insert(2, "Model", model_name)

    prediction_table["Measured_log10_sigma"] = observed
    prediction_table["Predicted_log10_sigma"] = predictions
    prediction_table["Residual_pred_minus_measured"] = residuals
    prediction_table["Absolute_error"] = np.abs(residuals)

    validation_prediction_frames.append(
        prediction_table.reset_index(drop=True)
    )

    print(
        f"{analysis} | Set {feature_set} | {model_name}: "
        f"RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}"
    )

validation_metrics = pd.DataFrame(
    validation_metric_rows
).sort_values(
    ["Analysis", "Feature_set", "RMSE", "Model"]
).reset_index(drop=True)

validation_predictions = pd.concat(
    validation_prediction_frames,
    ignore_index=True,
)

validation_metrics.to_csv(
    os.path.join(
        VALIDATION_RESULTS_FOLDER,
        "validation_metrics.csv",
    ),
    index=False,
)

validation_predictions.to_csv(
    os.path.join(
        VALIDATION_RESULTS_FOLDER,
        "validation_predictions.csv",
    ),
    index=False,
)


# ============================================================
# 10.5 SELECTING ONE MODEL PER ANALYSIS / FEATURE SET
# ============================================================

selected_models_table = (
    validation_metrics
    .sort_values(["Analysis", "Feature_set", "RMSE", "Model"])
    .groupby(["Analysis", "Feature_set"], sort=False)
    .head(1)
    .copy()
    .reset_index(drop=True)
)

selected_models_table["Selection_rule"] = (
    "Minimum validation RMSE; alphabetical model-name tie break"
)

selected_model_keys = {}
selected_training_models = {}

for row in selected_models_table.itertuples(index=False):

    pair = (row.Analysis, row.Feature_set)
    key = (row.Analysis, row.Feature_set, row.Model)

    selected_model_keys[pair] = key

    # Still the TRAINING-FITTED model.
    # Train + validation refitting happens in a later block.
    selected_training_models[pair] = best_models[key]

selected_models_table.to_csv(
    os.path.join(
        VALIDATION_RESULTS_FOLDER,
        "selected_models.csv",
    ),
    index=False,
)

# Recording selected pipeline settings for later refitting.
selection_records = []

for pair, key in selected_model_keys.items():

    model = best_models[key]

    selection_records.append({
        "analysis": pair[0],
        "feature_set": pair[1],
        "model": key[2],
        "regressor_parameters": (
            model.named_steps["model"].get_params(deep=True)
        ),
        "selection_rule": "Minimum validation RMSE",
        "test_used": False,
    })

with open(
    os.path.join(
        VALIDATION_RESULTS_FOLDER,
        "selected_model_settings.json",
    ),
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        selection_records,
        file,
        indent=2,
        default=str,
    )


# ============================================================
# 10.6 HELPER FOR SAVING FIGURES
# ============================================================

def save_validation_figure(fig, filename):
    for extension in ("png", "pdf"):
        fig.savefig(
            os.path.join(
                FIGURES_FOLDER,
                f"Block10_{VALIDATION_RUN_ID}_"
                f"{filename}.{extension}",
            ),
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )


# ============================================================
# 10.7 FIGURES — VALIDATION MODEL COMPARISONS
# ============================================================

for (analysis, feature_set), table in validation_metrics.groupby(
    ["Analysis", "Feature_set"],
    sort=False,
):

    table = table.sort_values(["RMSE", "Model"])

    colors = ["#009E73"] + ["#0072B2"] * (len(table) - 1)

    fig, ax = plt.subplots(
        figsize=(9, max(5, 0.35 * len(table) + 2)),
        layout="constrained",
    )

    ax.barh(
        table["Model"],
        table["RMSE"],
        color=colors,
    )

    ax.invert_yaxis()

    ax.set_xlabel("Validation RMSE (log10 conductivity units)")
    ax.set_title(
        f"{analysis.capitalize()} analysis — Set {feature_set}\n"
        "Validation model comparison"
    )

    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_validation_figure(
        fig,
        f"{analysis}_set_{feature_set}_model_comparison",
    )

    plt.show()
    plt.close(fig)


# ============================================================
# 10.8 FIGURES — SELECTED-MODEL PARITY AND RESIDUALS
# ============================================================

for row in selected_models_table.itertuples(index=False):

    mask = (
        validation_predictions["Analysis"].eq(row.Analysis)
        & validation_predictions["Feature_set"].eq(row.Feature_set)
        & validation_predictions["Model"].eq(row.Model)
    )

    table = validation_predictions.loc[mask]

    observed = table["Measured_log10_sigma"].to_numpy()
    predicted = table["Predicted_log10_sigma"].to_numpy()
    residuals = table["Residual_pred_minus_measured"].to_numpy()

    lower = min(observed.min(), predicted.min())
    upper = max(observed.max(), predicted.max())

    margin = max(0.1, 0.05 * (upper - lower))
    limits = (lower - margin, upper + margin)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        layout="constrained",
    )

    axes[0].scatter(
        observed,
        predicted,
        color="#0072B2",
        alpha=0.65,
        s=30,
        edgecolors="none",
    )

    axes[0].plot(
        limits,
        limits,
        linestyle="--",
        color="black",
        linewidth=1,
    )

    axes[0].set_xlim(limits)
    axes[0].set_ylim(limits)
    axes[0].set_aspect("equal", adjustable="box")

    axes[0].set_xlabel("Measured log10(σ / [S cm⁻¹])")
    axes[0].set_ylabel("Predicted log10(σ / [S cm⁻¹])")
    axes[0].set_title("Validation parity")

    axes[0].text(
        0.04,
        0.96,
        (
            f"RMSE = {row.RMSE:.3f}\n"
            f"MAE = {row.MAE:.3f}\n"
            f"R² = {row.R2:.3f}\n"
            f"n = {row.Validation_rows}"
        ),
        transform=axes[0].transAxes,
        va="top",
        bbox={
            "facecolor": "white",
            "alpha": 0.85,
            "edgecolor": "none",
        },
    )

    axes[1].scatter(
        predicted,
        residuals,
        color="#D55E00",
        alpha=0.65,
        s=30,
        edgecolors="none",
    )

    axes[1].axhline(
        0,
        color="black",
        linestyle="--",
        linewidth=1,
    )

    axes[1].set_xlabel("Predicted log10(σ / [S cm⁻¹])")
    axes[1].set_ylabel("Residual: predicted − measured")
    axes[1].set_title("Validation residuals")

    for ax in axes:
        ax.grid(alpha=0.15)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"{row.Analysis.capitalize()} — Set {row.Feature_set} "
        f"— {row.Model}"
    )

    save_validation_figure(
        fig,
        f"{row.Analysis}_set_{row.Feature_set}_"
        f"{row.Model}_parity_residuals",
    )

    plt.show()
    plt.close(fig)


# ============================================================
# 10.9 PAIRED A/B/C COMPARISON ON IDENTICAL VALIDATION ROWS
# ============================================================

comparison_rmse = (
    validation_metrics.loc[
        validation_metrics["Analysis"].eq("comparison")
    ]
    .pivot(
        index="Model",
        columns="Feature_set",
        values="RMSE",
    )
    .reindex(
        index=completed_model_names,
        columns=["A", "B", "C"],
    )
)

if comparison_rmse.isna().any().any():
    raise ValueError("Incomplete paired A/B/C validation results.")

feature_contribution_table = comparison_rmse.copy()

# Positive differences mean reduced error after adding features.
feature_contribution_table["Processing_RMSE_reduction_A_minus_B"] = (
    comparison_rmse["A"] - comparison_rmse["B"]
)

feature_contribution_table["Physics_RMSE_reduction_B_minus_C"] = (
    comparison_rmse["B"] - comparison_rmse["C"]
)

feature_contribution_table.to_csv(
    os.path.join(
        VALIDATION_RESULTS_FOLDER,
        "common_subset_feature_contributions.csv",
    ),
    index=True,
)

fig, ax = plt.subplots(
    figsize=(max(10, 0.75 * len(comparison_rmse)), 6),
    layout="constrained",
)

positions = np.arange(len(comparison_rmse))
width = 0.25

for offset, (feature_set, color) in enumerate(
    [("A", "#0072B2"), ("B", "#009E73"), ("C", "#D55E00")]
):
    ax.bar(
        positions + (offset - 1) * width,
        comparison_rmse[feature_set],
        width=width,
        color=color,
        label=f"Set {feature_set}",
    )

ax.set_xticks(positions)
ax.set_xticklabels(
    comparison_rmse.index,
    rotation=45,
    ha="right",
)

ax.set_ylabel("Validation RMSE (log10 conductivity units)")
ax.set_title("Common-subset A/B/C validation comparison")
ax.legend()

ax.grid(axis="y", alpha=0.2)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

save_validation_figure(
    fig,
    "common_subset_ABC_RMSE",
)

plt.show()
plt.close(fig)


# ============================================================
# 10.10 REPORTING SELECTED MODELS
# ============================================================

print("\nMODELS SELECTED BY VALIDATION RMSE")

print(
    selected_models_table[
        [
            "Analysis",
            "Feature_set",
            "Model",
            "Validation_rows",
            "Validation_groups",
            "RMSE",
            "MAE",
            "R2",
        ]
    ].round(4).to_string(index=False)
)

print("\nCOMMON-SUBSET FEATURE CONTRIBUTIONS")
print(feature_contribution_table.round(4).to_string())

print("\nBLOCK 10 COMPLETE")
print("Results:", VALIDATION_RESULTS_FOLDER)
print("Figures:", FIGURES_FOLDER)
print("Selected model keys:", selected_model_keys)
print("No models were refitted.")
print("Final test data remain untouched.")












# %% BLOCK 11 — FINAL TEST BENCHMARK OF ALL TUNED MODELS

#
# Full-data analysis: A, B
# Common-subset comparison: A, B, C
#
# Each pipeline is cloned from Block 9 and refitted on
# training + validation data with its selected settings.
#
# Primary models remain those selected by validation in Block 10.
# No test-based model selection or hyperparameter tuning.
#



# ============================================================
# 11.1 REQUIRED OBJECTS AND OUTPUT LOCATIONS
# ============================================================

required_objects = [
    "best_models",
    "selected_model_keys",
    "analysis_features",
    "split_indices",
    "target_all",
    "data",
    "SPLIT_MANIFEST_PATH",
]

missing_objects = [
    name for name in required_objects
    if name not in globals()
]

if missing_objects:
    raise RuntimeError(
        f"Missing objects: {missing_objects}. "
        "Run this block in the kernel containing the completed "
        "tuning and validation results."
    )

if not best_models:
    raise RuntimeError("No tuned models are available.")

FIGURES_FOLDER = os.path.join(PROJECT_FOLDER, "Figures")
RESULTS_FOLDER = os.path.join(PROJECT_FOLDER, "Results")

os.makedirs(FIGURES_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

manifest_stem = os.path.splitext(
    os.path.basename(SPLIT_MANIFEST_PATH)
)[0]

# Earlier five-model evaluation record, if present.
PRIOR_TEST_RECORD = os.path.join(
    RESULTS_FOLDER,
    f"{manifest_stem}_test_evaluation_record.json",
)

# Separate record for this expanded benchmark.
BENCHMARK_RECORD_PATH = os.path.join(
    RESULTS_FOLDER,
    f"{manifest_stem}_all_models_test_record.json",
)

if os.path.exists(BENCHMARK_RECORD_PATH):
    raise RuntimeError(
        "The expanded test benchmark has already been started "
        "for this split. Inspect its record before continuing. "
        "If interrupted, recover the existing run rather than "
        "deleting the record and repeating test evaluation."
    )

prior_test_evaluation = os.path.exists(PRIOR_TEST_RECORD)

if prior_test_evaluation:
    print(
        "An earlier test-evaluation record exists. "
        "This run will be recorded as an expanded benchmark "
        "after prior test evaluation."
    )


# ============================================================
# 11.2 CHECKING THAT EACH MODEL COVERS ALL FIVE DATASETS
# ============================================================

expected_pairs = {
    (analysis, feature_set)
    for analysis, feature_dictionary in analysis_features.items()
    for feature_set in feature_dictionary
}

model_names = sorted({
    model_name
    for analysis, feature_set, model_name in best_models
})

expected_keys = {
    (analysis, feature_set, model_name)
    for analysis, feature_set in expected_pairs
    for model_name in model_names
}

actual_keys = set(best_models)

if actual_keys != expected_keys:
    raise ValueError(
        "Some models have not completed every analysis/feature set. "
        f"Missing: {sorted(expected_keys - actual_keys)}; "
        f"unexpected: {sorted(actual_keys - expected_keys)}"
    )

if set(selected_model_keys) != expected_pairs:
    raise ValueError(
        "Validation-selected models do not cover all five datasets."
    )

primary_keys = set()

for pair, key in selected_model_keys.items():

    if key[:2] != pair or key not in best_models:
        raise ValueError(
            f"Invalid validation-selected model mapping: {pair}: {key}"
        )

    primary_keys.add(key)

print("\nFrozen model families:", ", ".join(model_names))
print("Model families:", len(model_names))
print("Pipelines to evaluate:", len(best_models))

if "model_specs" in globals():
    absent_models = sorted(set(model_specs) - set(model_names))
    if absent_models:
        print(
            "Specified but not included in this benchmark:",
            ", ".join(absent_models),
        )


# ============================================================
# 11.3 VERIFYING DEVELOPMENT / TEST SEPARATION
# ============================================================

development_indices = {}
final_test_indices = {}

for analysis, partitions in split_indices.items():

    development_idx = partitions["train"].append(
        partitions["validation"]
    )
    test_idx = partitions["test"]

    if not development_idx.is_unique or not test_idx.is_unique:
        raise ValueError(f"{analysis}: duplicate row indices.")

    if len(development_idx.intersection(test_idx)) > 0:
        raise ValueError(f"{analysis}: development/test row overlap.")

    development_groups = set(
        data.loc[development_idx, "composition_group"]
    )
    test_groups = set(
        data.loc[test_idx, "composition_group"]
    )

    if development_groups.intersection(test_groups):
        raise ValueError(
            f"{analysis}: development/test composition overlap."
        )

    if len(test_idx) < 2:
        raise ValueError(f"{analysis}: fewer than two test rows.")

    for feature_set, frame in analysis_features[analysis].items():

        if not development_idx.isin(frame.index).all():
            raise ValueError(
                f"{analysis}/{feature_set}: development rows missing."
            )

        if not test_idx.isin(frame.index).all():
            raise ValueError(
                f"{analysis}/{feature_set}: test rows missing."
            )

    development_indices[analysis] = development_idx
    final_test_indices[analysis] = test_idx


# ============================================================
# 11.4 FREEZING SETTINGS AND RECORDING THE RUN BEFORE TESTING
# ============================================================

FINAL_RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

FINAL_RESULTS_FOLDER = os.path.join(
    RESULTS_FOLDER,
    f"block11_all_models_{FINAL_RUN_ID}",
)

os.makedirs(FINAL_RESULTS_FOLDER, exist_ok=False)


def json_compatible(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


with open(SPLIT_MANIFEST_PATH, "rb") as file:
    split_manifest_sha256 = hashlib.sha256(file.read()).hexdigest()

frozen_settings = []

for key in sorted(best_models):

    analysis, feature_set, model_name = key
    pipeline = best_models[key]

    # Cache location affects execution, not the modelling choices.
    pipeline_parameters = {
        name: value
        for name, value in pipeline.get_params(deep=True).items()
        if name != "memory"
    }

    frozen_settings.append({
        "analysis": analysis,
        "feature_set": feature_set,
        "model": model_name,
        "primary_validation_selected": key in primary_keys,
        "pipeline_parameters": pipeline_parameters,
    })

with open(
    os.path.join(FINAL_RESULTS_FOLDER, "frozen_pipeline_settings.json"),
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        frozen_settings,
        file,
        indent=2,
        default=json_compatible,
    )

benchmark_record = {
    "run_id": FINAL_RUN_ID,
    "status": "started",
    "results_folder": FINAL_RESULTS_FOLDER,
    "split_manifest": SPLIT_MANIFEST_PATH,
    "split_manifest_sha256": split_manifest_sha256,
    "prior_test_evaluation_record_exists": prior_test_evaluation,
    "model_families": model_names,
    "pipeline_count": len(best_models),
    "primary_model_keys": [
        list(key) for key in sorted(primary_keys)
    ],
    "selection_policy": (
        "Primary models fixed by validation RMSE in Block 10. "
        "Test results do not select or retune models."
    ),
    "completed_pipeline_keys": [],
    "software_versions": {
        "python": platform.python_version(),
        "scikit-learn": version("scikit-learn"),
        "numpy": version("numpy"),
        "pandas": version("pandas"),
        "shap": version("shap"),
        "xgboost": version("xgboost"),
    },
}

with open(
    BENCHMARK_RECORD_PATH,
    "x",
    encoding="utf-8",
) as file:
    json.dump(benchmark_record, file, indent=2)


# ============================================================
# 11.5 REFITTING EVERY FROZEN PIPELINE AND EVALUATING ITS TEST SET
# ============================================================

# Triple keys:
# final_models[("full", "A", "SVR")]
# final_models[("comparison", "C", "XGBoost")]

final_models = {}

final_test_metric_rows = []
final_test_prediction_frames = []
final_group_metric_frames = []

for number, key in enumerate(sorted(best_models), start=1):

    analysis, feature_set, model_name = key

    development_idx = development_indices[analysis]
    test_idx = final_test_indices[analysis]

    feature_frame = analysis_features[analysis][feature_set]

    X_development = feature_frame.loc[development_idx]
    y_development = target_all.loc[development_idx]

    X_test = feature_frame.loc[test_idx]
    y_test = target_all.loc[test_idx]

    if not X_development.index.equals(y_development.index):
        raise ValueError(f"{key}: development X/y mismatch.")

    if not X_test.index.equals(y_test.index):
        raise ValueError(f"{key}: test X/y mismatch.")

    print(
        f"\n[{number}/{len(best_models)}] "
        f"{analysis} | Set {feature_set} | {model_name}",
        flush=True,
    )

    # Clone preserves tuned settings but removes fitted state.
    final_pipeline = clone(best_models[key])

    # No cache is needed for this one-time refit.
    final_pipeline.set_params(memory=None)

    # Imputation, scaling and SHAP are refitted on development only.
    final_pipeline.fit(X_development, y_development)

    final_models[key] = final_pipeline

    selector = final_pipeline.named_steps["shap_selector"]

    stem = f"{analysis}_set_{feature_set}_{model_name}"

    if hasattr(selector, "ranking_"):
        selector.ranking_.to_csv(
            os.path.join(
                FINAL_RESULTS_FOLDER,
                f"{stem}_development_SHAP_ranking.csv",
            ),
            index=False,
        )

    selected_feature_count = (
        selector.n_features_selected_
        if hasattr(selector, "n_features_selected_")
        else np.nan
    )

    observed = y_test.to_numpy(dtype=float)

    predicted = np.asarray(
        final_pipeline.predict(X_test),
        dtype=float,
    ).reshape(-1)

    if len(predicted) != len(observed):
        raise ValueError(f"{key}: prediction length mismatch.")

    if not (
        np.isfinite(observed).all()
        and np.isfinite(predicted).all()
    ):
        raise ValueError(f"{key}: non-finite test values.")

    residuals = predicted - observed

    rmse = np.sqrt(mean_squared_error(observed, predicted))
    mae = mean_absolute_error(observed, predicted)

    r2 = (
        r2_score(observed, predicted)
        if np.unique(observed).size > 1
        else np.nan
    )

    # Equal-composition-weighted supplementary metrics.
    errors = pd.DataFrame({
        "composition_group": data.loc[
            test_idx, "composition_group"
        ].to_numpy(),
        "absolute_error": np.abs(residuals),
        "squared_error": residuals ** 2,
    })

    group_metrics = (
        errors.groupby("composition_group")
        .agg(
            Rows=("absolute_error", "size"),
            MAE=("absolute_error", "mean"),
            MSE=("squared_error", "mean"),
        )
        .reset_index()
    )

    group_metrics["RMSE"] = np.sqrt(group_metrics["MSE"])

    balanced_mae = group_metrics["MAE"].mean()
    balanced_rmse = np.sqrt(group_metrics["MSE"].mean())

    group_metrics.insert(0, "Analysis", analysis)
    group_metrics.insert(1, "Feature_set", feature_set)
    group_metrics.insert(2, "Model", model_name)

    final_group_metric_frames.append(group_metrics)

    final_test_metric_rows.append({
        "Analysis": analysis,
        "Feature_set": feature_set,
        "Model": model_name,
        "Primary_validation_selected": key in primary_keys,
        "Development_rows": len(development_idx),
        "Test_rows": len(test_idx),
        "Test_groups": len(group_metrics),
        "Selected_feature_count": selected_feature_count,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Mean_residual": residuals.mean(),
        "Composition_balanced_MAE": balanced_mae,
        "Composition_balanced_RMSE": balanced_rmse,
    })

    prediction_table = data.loc[
        test_idx,
        [
            "excel_row",
            "formula",
            "paper_id",
            "composition_group",
            "temperature_K",
        ],
    ].copy()

    prediction_table.insert(0, "Analysis", analysis)
    prediction_table.insert(1, "Feature_set", feature_set)
    prediction_table.insert(2, "Model", model_name)

    prediction_table["Primary_validation_selected"] = (
        key in primary_keys
    )

    prediction_table["Measured_log10_sigma"] = observed
    prediction_table["Predicted_log10_sigma"] = predicted
    prediction_table["Residual_pred_minus_measured"] = residuals
    prediction_table["Absolute_error"] = np.abs(residuals)

    final_test_prediction_frames.append(
        prediction_table.reset_index(drop=True)
    )

    # Save accumulated results after every completed pipeline.
    pd.DataFrame(final_test_metric_rows).to_csv(
        os.path.join(
            FINAL_RESULTS_FOLDER,
            "final_test_metrics.csv",
        ),
        index=False,
    )

    pd.concat(
        final_test_prediction_frames,
        ignore_index=True,
    ).to_csv(
        os.path.join(
            FINAL_RESULTS_FOLDER,
            "final_test_predictions.csv",
        ),
        index=False,
    )

    pd.concat(
        final_group_metric_frames,
        ignore_index=True,
    ).to_csv(
        os.path.join(
            FINAL_RESULTS_FOLDER,
            "final_test_metrics_by_composition.csv",
        ),
        index=False,
    )

    benchmark_record["completed_pipeline_keys"].append(list(key))

    with open(
        BENCHMARK_RECORD_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(benchmark_record, file, indent=2)

    print(
        f"RMSE={rmse:.4f} | MAE={mae:.4f} | R²={r2:.4f}",
        flush=True,
    )


# ============================================================
# 11.6 FINALIZING METRIC AND PREDICTION TABLES
# ============================================================

final_test_metrics = pd.DataFrame(final_test_metric_rows)

final_test_predictions = pd.concat(
    final_test_prediction_frames,
    ignore_index=True,
)

final_test_metrics_by_composition = pd.concat(
    final_group_metric_frames,
    ignore_index=True,
)

primary_test_metrics = final_test_metrics.loc[
    final_test_metrics["Primary_validation_selected"]
].copy()

primary_test_metrics.to_csv(
    os.path.join(
        FINAL_RESULTS_FOLDER,
        "primary_validation_selected_test_metrics.csv",
    ),
    index=False,
)

# Convenient access for later primary-model interpretation.
primary_final_models = {
    pair: final_models[key]
    for pair, key in selected_model_keys.items()
}

benchmark_record["status"] = "numerical_evaluation_completed"
benchmark_record["evaluation_completed_at"] = (
    datetime.now().isoformat()
)

with open(
    BENCHMARK_RECORD_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(benchmark_record, file, indent=2)


# ============================================================
# 11.7 SAME-ALGORITHM FEATURE-SET COMPARISONS
# ============================================================

feature_effect_rows = []

for analysis in ("full", "comparison"):

    analysis_table = final_test_metrics.loc[
        final_test_metrics["Analysis"].eq(analysis)
    ]

    transitions = (
        [("A", "B"), ("B", "C")]
        if analysis == "comparison"
        else [("A", "B")]
    )

    for model_name in model_names:

        model_table = (
            analysis_table.loc[
                analysis_table["Model"].eq(model_name)
            ]
            .set_index("Feature_set")
        )

        for before, after in transitions:

            feature_effect_rows.append({
                "Analysis": analysis,
                "Model": model_name,
                "Comparison": f"{before} to {after}",
                "RMSE_before": model_table.loc[before, "RMSE"],
                "RMSE_after": model_table.loc[after, "RMSE"],
                "RMSE_reduction": (
                    model_table.loc[before, "RMSE"]
                    - model_table.loc[after, "RMSE"]
                ),
                "MAE_reduction": (
                    model_table.loc[before, "MAE"]
                    - model_table.loc[after, "MAE"]
                ),
                "R2_increase": (
                    model_table.loc[after, "R2"]
                    - model_table.loc[before, "R2"]
                ),
            })

same_algorithm_feature_effects = pd.DataFrame(feature_effect_rows)

same_algorithm_feature_effects.to_csv(
    os.path.join(
        FINAL_RESULTS_FOLDER,
        "same_algorithm_feature_set_comparisons.csv",
    ),
    index=False,
)


# ============================================================
# 11.8 FIGURE-SAVING HELPER
# ============================================================

def save_benchmark_figure(fig, filename):
    for extension in ("png", "pdf"):
        fig.savefig(
            os.path.join(
                FIGURES_FOLDER,
                f"Block11_{FINAL_RUN_ID}_{filename}.{extension}",
            ),
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )


# ============================================================
# 11.9 ALL-MODEL TEST METRICS — PRIMARY MODELS HIGHLIGHTED
# ============================================================

for (analysis, feature_set), table in final_test_metrics.groupby(
    ["Analysis", "Feature_set"],
    sort=False,
):

    # Alphabetical order avoids implying test-based model selection.
    table = table.sort_values("Model")
    positions = np.arange(len(table))

    labels = [
        model + (" *" if primary else "")
        for model, primary in zip(
            table["Model"],
            table["Primary_validation_selected"],
        )
    ]

    colors = [
        "#009E73" if primary else "#0072B2"
        for primary in table["Primary_validation_selected"]
    ]

    fig, axes = plt.subplots(
        1, 3,
        figsize=(15, max(5, 0.35 * len(table) + 2)),
        layout="constrained",
    )

    for ax, metric in zip(axes, ("RMSE", "MAE", "R2")):

        ax.barh(positions, table[metric], color=colors)

        ax.set_yticks(positions)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()

        ax.set_xlabel(
            f"{metric} (log10 conductivity units)"
            if metric != "R2"
            else "R²"
        )

        ax.axvline(0, color="black", linewidth=0.7)
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"{analysis.capitalize()} — Set {feature_set}: final test\n"
        "* Green = model selected by validation"
    )

    save_benchmark_figure(
        fig,
        f"{analysis}_set_{feature_set}_all_model_metrics",
    )

    plt.show()
    plt.close(fig)


# ============================================================
# 11.10 PARITY AND RESIDUAL FIGURES FOR EVERY PIPELINE
# ============================================================

# Save all figures; display only the primary models in VS Code.
for row in final_test_metrics.itertuples(index=False):

    mask = (
        final_test_predictions["Analysis"].eq(row.Analysis)
        & final_test_predictions["Feature_set"].eq(row.Feature_set)
        & final_test_predictions["Model"].eq(row.Model)
    )

    table = final_test_predictions.loc[mask]

    observed = table["Measured_log10_sigma"].to_numpy()
    predicted = table["Predicted_log10_sigma"].to_numpy()
    residuals = table["Residual_pred_minus_measured"].to_numpy()

    lower = min(observed.min(), predicted.min())
    upper = max(observed.max(), predicted.max())

    margin = max(0.1, 0.05 * (upper - lower))
    limits = [lower - margin, upper + margin]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(12, 5),
        layout="constrained",
    )

    axes[0].scatter(
        observed, predicted,
        s=30, alpha=0.7,
        color="#0072B2",
        edgecolors="none",
    )

    axes[0].plot(
        limits, limits,
        "--", color="black", linewidth=1,
    )

    axes[0].set_xlim(limits)
    axes[0].set_ylim(limits)
    axes[0].set_aspect("equal", adjustable="box")

    axes[0].set_xlabel("Measured log10(σ / [S cm⁻¹])")
    axes[0].set_ylabel("Predicted log10(σ / [S cm⁻¹])")
    axes[0].set_title("Final test parity")

    axes[0].text(
        0.04, 0.96,
        (
            f"RMSE = {row.RMSE:.3f}\n"
            f"MAE = {row.MAE:.3f}\n"
            f"R² = {row.R2:.3f}\n"
            f"Rows = {row.Test_rows}\n"
            f"Compositions = {row.Test_groups}"
        ),
        transform=axes[0].transAxes,
        va="top",
        bbox={
            "facecolor": "white",
            "alpha": 0.85,
            "edgecolor": "none",
        },
    )

    axes[1].scatter(
        predicted, residuals,
        s=30, alpha=0.7,
        color="#D55E00",
        edgecolors="none",
    )

    axes[1].axhline(
        0, color="black",
        linestyle="--", linewidth=1,
    )

    axes[1].set_xlabel("Predicted log10(σ / [S cm⁻¹])")
    axes[1].set_ylabel("Residual: predicted − measured")
    axes[1].set_title("Final test residuals")

    for ax in axes:
        ax.grid(alpha=0.15)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    designation = (
        "Primary: validation-selected"
        if row.Primary_validation_selected
        else "Benchmark model"
    )

    fig.suptitle(
        f"{row.Analysis.capitalize()} — Set {row.Feature_set}"
        f" — {row.Model}\n{designation}"
    )

    save_benchmark_figure(
        fig,
        f"{row.Analysis}_set_{row.Feature_set}_"
        f"{row.Model}_parity_residuals",
    )

    if row.Primary_validation_selected:
        plt.show()

    plt.close(fig)


# ============================================================
# 11.11 PAIRED FEATURE-SET RMSE FIGURES
# ============================================================

feature_colors = {
    "A": "#0072B2",
    "B": "#009E73",
    "C": "#D55E00",
}

for analysis in ("full", "comparison"):

    feature_sets = (
        ["A", "B", "C"]
        if analysis == "comparison"
        else ["A", "B"]
    )

    table = (
        final_test_metrics.loc[
            final_test_metrics["Analysis"].eq(analysis)
        ]
        .pivot(
            index="Model",
            columns="Feature_set",
            values="RMSE",
        )
        .reindex(index=model_names, columns=feature_sets)
    )

    table.to_csv(
        os.path.join(
            FINAL_RESULTS_FOLDER,
            f"{analysis}_paired_test_RMSE.csv",
        )
    )

    fig, ax = plt.subplots(
        figsize=(max(10, 0.8 * len(model_names)), 6),
        layout="constrained",
    )

    positions = np.arange(len(table))
    width = 0.8 / len(feature_sets)

    for number, feature_set in enumerate(feature_sets):

        offset = (
            number - (len(feature_sets) - 1) / 2
        ) * width

        ax.bar(
            positions + offset,
            table[feature_set],
            width=width,
            color=feature_colors[feature_set],
            label=f"Set {feature_set}",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        table.index,
        rotation=45,
        ha="right",
    )

    ax.set_ylabel("Final test RMSE (log10 conductivity units)")
    ax.set_title(
        f"{analysis.capitalize()}: same-algorithm feature comparison\n"
        "Identical test observations within this analysis"
    )

    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_benchmark_figure(
        fig,
        f"{analysis}_paired_feature_set_RMSE",
    )

    plt.show()
    plt.close(fig)


# ============================================================
# 11.12 FINAL REPORT
# ============================================================

benchmark_record["status"] = "completed"
benchmark_record["completed_at"] = datetime.now().isoformat()

with open(
    BENCHMARK_RECORD_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(benchmark_record, file, indent=2)

print("\nPRIMARY RESULTS — MODELS SELECTED BY VALIDATION")

print(
    primary_test_metrics[
        [
            "Analysis",
            "Feature_set",
            "Model",
            "Test_rows",
            "Test_groups",
            "RMSE",
            "MAE",
            "R2",
        ]
    ].round(4).to_string(index=False)
)

print("\nALL-MODEL BENCHMARK")

print(
    final_test_metrics[
        [
            "Analysis",
            "Feature_set",
            "Model",
            "Primary_validation_selected",
            "RMSE",
            "MAE",
            "R2",
        ]
    ].round(4).to_string(index=False)
)

print("\nSAME-ALGORITHM FEATURE COMPARISONS")
print(same_algorithm_feature_effects.round(4).to_string(index=False))

print("\nBLOCK 11 COMPLETE")
print("Results:", FINAL_RESULTS_FOLDER)
print("Figures:", FIGURES_FOLDER)
print("All fitted pipelines: final_models[(analysis, set, model)]")
print("Primary pipelines: primary_final_models[(analysis, set)]")
print("Test results did not change the validation-selected models.")







# %% BLOCK 11B — THRESHOLD-BASED CLASSIFICATION ASSESSMENT

#
# Converts final regression predictions into high/low labels.
# Uses final_test_predictions from the expanded Block 11.
#
# Positive class:
# Measured conductivity >= the specified conductivity threshold.
#
# ROC and PR curves use continuous predicted log10 conductivity.
# No classifier is trained and no threshold is optimized.



# ============================================================
# 11B.1 SCIENTIFIC SETTINGS
# ============================================================

# Provisional cutoff: 0.01 S/cm.
CONDUCTIVITY_THRESHOLD_S_CM = 1e-2

# None includes all available test temperatures.
# Example of a temperature window:
# TEMPERATURE_WINDOW_K = (948.15, 998.15)
#
# A window is not an exact fixed-temperature evaluation.
TEMPERATURE_WINDOW_K = None

if CONDUCTIVITY_THRESHOLD_S_CM <= 0:
    raise ValueError("The conductivity threshold must be positive.")

LOG10_CONDUCTIVITY_THRESHOLD = np.log10(
    CONDUCTIVITY_THRESHOLD_S_CM
)

if TEMPERATURE_WINDOW_K is not None:
    lower_temperature, upper_temperature = TEMPERATURE_WINDOW_K

    if not 0 < lower_temperature <= upper_temperature:
        raise ValueError("Invalid temperature window.")


# ============================================================
# 11B.2 CHECKING INPUTS AND PREPARING OUTPUT FOLDER
# ============================================================

required_objects = [
    "final_test_predictions",
    "FINAL_RESULTS_FOLDER",
    "FINAL_RUN_ID",
    "FIGURES_FOLDER",
]

missing_objects = [
    name for name in required_objects
    if name not in globals()
]

if missing_objects:
    raise RuntimeError(
        f"Missing objects: {missing_objects}. "
        "Complete the expanded Block 11 first."
    )

required_columns = {
    "Analysis",
    "Feature_set",
    "Model",
    "Primary_validation_selected",
    "excel_row",
    "composition_group",
    "temperature_K",
    "Measured_log10_sigma",
    "Predicted_log10_sigma",
}

missing_columns = (
    required_columns - set(final_test_predictions.columns)
)

if missing_columns:
    raise ValueError(
        f"Missing prediction columns: {missing_columns}"
    )

classification_data = final_test_predictions.copy()

numeric_columns = [
    "temperature_K",
    "Measured_log10_sigma",
    "Predicted_log10_sigma",
]

if not np.isfinite(
    classification_data[numeric_columns].to_numpy(dtype=float)
).all():
    raise ValueError("Predictions contain missing or infinite values.")

if classification_data.duplicated(
    ["Analysis", "Feature_set", "Model", "excel_row"]
).any():
    raise ValueError("Duplicate model–measurement predictions found.")

if TEMPERATURE_WINDOW_K is not None:
    classification_data = classification_data.loc[
        classification_data["temperature_K"].between(
            lower_temperature,
            upper_temperature,
            inclusive="both",
        )
    ].copy()

if classification_data.empty:
    raise ValueError("No test observations meet the temperature scope.")

CLASSIFICATION_RESULTS_FOLDER = os.path.join(
    FINAL_RESULTS_FOLDER,
    "classification",
)

# Preventing accidental replacement of a previous classification run.
os.makedirs(CLASSIFICATION_RESULTS_FOLDER, exist_ok=False)
os.makedirs(FIGURES_FOLDER, exist_ok=True)

temperature_description = (
    "All available test temperatures"
    if TEMPERATURE_WINDOW_K is None
    else (
        f"{lower_temperature:g}–{upper_temperature:g} K"
    )
)

classification_settings = {
    "conductivity_threshold_S_cm": CONDUCTIVITY_THRESHOLD_S_CM,
    "log10_conductivity_threshold": LOG10_CONDUCTIVITY_THRESHOLD,
    "positive_class": "High conductivity",
    "positive_rule": "Conductivity >= threshold",
    "temperature_scope": temperature_description,
    "score_used_for_curves": "Predicted log10 conductivity",
    "threshold_optimized": False,
    "models_retrained": False,
    "zero_division_policy": (
        "Precision, recall and F1 use zero_division=0; "
        "undefined cases are flagged in the metrics table."
    ),
}

with open(
    os.path.join(
        CLASSIFICATION_RESULTS_FOLDER,
        "classification_settings.json",
    ),
    "w",
    encoding="utf-8",
) as file:
    json.dump(classification_settings, file, indent=2)

classification_data["Actual_high"] = (
    classification_data["Measured_log10_sigma"]
    >= LOG10_CONDUCTIVITY_THRESHOLD
).astype(int)

classification_data["Predicted_high"] = (
    classification_data["Predicted_log10_sigma"]
    >= LOG10_CONDUCTIVITY_THRESHOLD
).astype(int)

classification_data.to_csv(
    os.path.join(
        CLASSIFICATION_RESULTS_FOLDER,
        "threshold_classification_predictions.csv",
    ),
    index=False,
)


# ============================================================
# 11B.3 FIGURE-SAVING HELPER
# ============================================================

def save_classification_figure(fig, filename):
    for extension in ("png", "pdf"):
        fig.savefig(
            os.path.join(
                FIGURES_FOLDER,
                f"Block11B_{FINAL_RUN_ID}_{filename}.{extension}",
            ),
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )


# ============================================================
# 11B.4 METRICS, REPORTS AND CURVES FOR EVERY PIPELINE
# ============================================================

classification_metric_rows = []
classification_report_frames = []
classification_curve_frames = []

group_columns = ["Analysis", "Feature_set", "Model"]

for key, table in classification_data.groupby(
    group_columns,
    sort=True,
):

    analysis, feature_set, model_name = key
    stem = f"{analysis}_set_{feature_set}_{model_name}"

    if table["Primary_validation_selected"].nunique() != 1:
        raise ValueError(f"Inconsistent primary-model flag: {key}")

    is_primary = bool(
        table["Primary_validation_selected"].iloc[0]
    )

    actual = table["Actual_high"].to_numpy(dtype=int)
    predicted = table["Predicted_high"].to_numpy(dtype=int)

    # Ranking score, not a probability.
    scores = table["Predicted_log10_sigma"].to_numpy(dtype=float)

    cm = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    actual_positive_count = int(actual.sum())
    actual_negative_count = int(len(actual) - actual_positive_count)
    predicted_positive_count = int(predicted.sum())

    prevalence = actual.mean()
    both_classes_present = np.unique(actual).size == 2

    precision_value = precision_score(
        actual, predicted, zero_division=0
    )
    recall_value = recall_score(
        actual, predicted, zero_division=0
    )
    f1_value = f1_score(
        actual, predicted, zero_division=0
    )

    report = classification_report(
        actual,
        predicted,
        labels=[0, 1],
        target_names=["Low conductivity", "High conductivity"],
        output_dict=True,
        zero_division=0,
    )

    # Keep class-specific and macro/weighted report rows.
    report_table = pd.DataFrame({
        label: values
        for label, values in report.items()
        if isinstance(values, dict)
    }).T

    report_table.index.name = "Class_or_average"
    report_table = report_table.reset_index()

    report_table.insert(0, "Analysis", analysis)
    report_table.insert(1, "Feature_set", feature_set)
    report_table.insert(2, "Model", model_name)

    classification_report_frames.append(report_table)

    pd.DataFrame(
        cm,
        index=["Actual_low", "Actual_high"],
        columns=["Predicted_low", "Predicted_high"],
    ).to_csv(
        os.path.join(
            CLASSIFICATION_RESULTS_FOLDER,
            f"{stem}_confusion_matrix.csv",
        )
    )

    # Ranking curves are omitted when the actual test labels
    # contain only one class.
    roc_auc_value = np.nan
    pr_auc_value = np.nan
    average_precision_value = np.nan

    if both_classes_present:

        fpr, tpr, roc_thresholds = roc_curve(actual, scores)

        roc_auc_value = roc_auc_score(actual, scores)

        pr_precision, pr_recall, pr_thresholds = (
            precision_recall_curve(actual, scores)
        )

        # Two distinct PR summaries:
        # 1. Trapezoidal area under the PR curve
        # 2. Average precision
        pr_auc_value = auc(pr_recall, pr_precision)

        average_precision_value = average_precision_score(
            actual,
            scores,
        )

        roc_table = pd.DataFrame({
            "Curve": "ROC",
            "X": fpr,
            "Y": tpr,
            "Score_threshold": roc_thresholds,
        })

        pr_table = pd.DataFrame({
            "Curve": "PR",
            "X": pr_recall,
            "Y": pr_precision,
            # Final PR point has no corresponding threshold.
            "Score_threshold": np.append(
                pr_thresholds,
                np.nan,
            ),
        })

        curve_table = pd.concat(
            [roc_table, pr_table],
            ignore_index=True,
        )

        curve_table.insert(0, "Analysis", analysis)
        curve_table.insert(1, "Feature_set", feature_set)
        curve_table.insert(2, "Model", model_name)

        classification_curve_frames.append(curve_table)

    classification_metric_rows.append({
        "Analysis": analysis,
        "Feature_set": feature_set,
        "Model": model_name,
        "Primary_validation_selected": is_primary,
        "Rows": len(table),
        "Composition_groups": table["composition_group"].nunique(),
        "Actual_low_count": actual_negative_count,
        "Actual_high_count": actual_positive_count,
        "Predicted_high_count": predicted_positive_count,
        "High_class_prevalence": prevalence,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "Accuracy": np.mean(actual == predicted),
        "Precision_high": precision_value,
        "Recall_high": recall_value,
        "F1_high": f1_value,
        "Precision_defined": predicted_positive_count > 0,
        "Recall_defined": actual_positive_count > 0,
        "F1_defined": (2 * tp + fp + fn) > 0,
        "ROC_AUC": roc_auc_value,
        "PR_AUC_trapezoidal": pr_auc_value,
        "Average_precision": average_precision_value,
        "Curve_status": (
            "Both actual classes present"
            if both_classes_present
            else "Unavailable: only one actual class present"
        ),
    })

    # --------------------------------------------------------
    # FIGURE: CONFUSION MATRIX + ROC + PRECISION–RECALL
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1, 3,
        figsize=(16, 4.8),
        layout="constrained",
    )

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Low", "High"],
    ).plot(
        ax=axes[0],
        cmap="Blues",
        colorbar=False,
        values_format="d",
    )

    axes[0].set_title(
        "Confusion matrix\n"
        f"Precision={precision_value:.3f}, "
        f"Recall={recall_value:.3f}, F1={f1_value:.3f}"
    )

    if both_classes_present:

        axes[1].plot(
            fpr,
            tpr,
            color="#0072B2",
            linewidth=2,
            label=f"ROC AUC = {roc_auc_value:.3f}",
        )

        axes[1].plot(
            [0, 1], [0, 1],
            "--",
            color="gray",
            label="Random-ranking reference",
        )

        # Mark the operating point at the specified cutoff.
        axes[1].scatter(
            fp / (fp + tn),
            tp / (tp + fn),
            color="#D55E00",
            s=45,
            zorder=5,
            label="Specified conductivity cutoff",
        )

        axes[2].step(
            pr_recall,
            pr_precision,
            where="post",
            color="#009E73",
            linewidth=2,
            label=f"Average precision = {average_precision_value:.3f}",
        )

        axes[2].axhline(
            prevalence,
            linestyle="--",
            color="gray",
            label=f"Class prevalence = {prevalence:.3f}",
        )

        if predicted_positive_count > 0:
            axes[2].scatter(
                recall_value,
                precision_value,
                color="#D55E00",
                s=45,
                zorder=5,
                label="Specified conductivity cutoff",
            )

        axes[1].legend(fontsize=8)
        axes[2].legend(fontsize=8)

    else:
        for ax in axes[1:]:
            ax.text(
                0.5, 0.5,
                "Curve unavailable:\nonly one actual class",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    axes[1].set_title("ROC curve")
    axes[1].set_xlabel("False-positive rate")
    axes[1].set_ylabel("True-positive rate")

    axes[2].set_title("Precision–recall curve")
    axes[2].set_xlabel("Recall")
    axes[2].set_ylabel("Precision")

    for ax in axes[1:]:
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    designation = (
        "Validation-selected primary model"
        if is_primary
        else "Benchmark model"
    )

    fig.suptitle(
        f"{analysis.capitalize()} — Set {feature_set} — {model_name}\n"
        f"{designation}; high ≥ "
        f"{CONDUCTIVITY_THRESHOLD_S_CM:g} S/cm; "
        f"{temperature_description}"
    )

    save_classification_figure(fig, f"{stem}_classification")

    # Save every figure, display only primary models.
    if is_primary:
        plt.show()

    plt.close(fig)


# ============================================================
# 11B.5 SAVING COMBINED RESULTS
# ============================================================

classification_metrics = pd.DataFrame(
    classification_metric_rows
)

classification_reports = pd.concat(
    classification_report_frames,
    ignore_index=True,
)

classification_metrics.to_csv(
    os.path.join(
        CLASSIFICATION_RESULTS_FOLDER,
        "classification_metrics.csv",
    ),
    index=False,
)

classification_reports.to_csv(
    os.path.join(
        CLASSIFICATION_RESULTS_FOLDER,
        "classification_reports.csv",
    ),
    index=False,
)

curve_columns = [
    "Analysis",
    "Feature_set",
    "Model",
    "Curve",
    "X",
    "Y",
    "Score_threshold",
]

classification_curves = (
    pd.concat(classification_curve_frames, ignore_index=True)
    if classification_curve_frames
    else pd.DataFrame(columns=curve_columns)
)

classification_curves.to_csv(
    os.path.join(
        CLASSIFICATION_RESULTS_FOLDER,
        "ROC_PR_curve_coordinates.csv",
    ),
    index=False,
)

primary_classification_metrics = classification_metrics.loc[
    classification_metrics["Primary_validation_selected"]
].copy()

primary_classification_metrics.to_csv(
    os.path.join(
        CLASSIFICATION_RESULTS_FOLDER,
        "primary_classification_metrics.csv",
    ),
    index=False,
)

print("\nPRIMARY MODEL CLASSIFICATION RESULTS")

print(
    primary_classification_metrics[
        [
            "Analysis",
            "Feature_set",
            "Model",
            "Rows",
            "Actual_high_count",
            "Precision_high",
            "Recall_high",
            "F1_high",
            "ROC_AUC",
            "Average_precision",
        ]
    ].round(4).to_string(index=False)
)

print("\nBLOCK 11B COMPLETE")
print("Conductivity cutoff:", CONDUCTIVITY_THRESHOLD_S_CM, "S/cm")
print("Temperature scope:", temperature_description)
print("Tables:", CLASSIFICATION_RESULTS_FOLDER)
print("Figures:", FIGURES_FOLDER)
print("No models were trained or changed.")









# %% BLOCK 12 — FINAL-PREDICTOR SHAP INTERPRETATION
#
# Explains the validation-selected final models.
# Uses development data = training + validation.
#
# Does not use test observations or retrain models.
#
# Distinction:
# Block 8 explained the RF feature selector.
# This block explains the actual final conductivity predictor.





# ============================================================
# 12.1 SETTINGS
# ============================================================

SHAP_BACKGROUND_ROWS = 50
SHAP_EXPLAIN_ROWS = 100

# Number of random permutation cycles for model-agnostic SHAP.
# These are approximation settings, not tuning parameters.
SHAP_PERMUTATION_CYCLES = 10

SHAP_DISPLAY_FEATURES = 30
SHAP_MAX_DEPENDENCE_PLOTS = 5

SHAP_RANDOM_STATE = RANDOM_STATE

required_objects = [
    "primary_final_models",
    "selected_model_keys",
    "analysis_features",
    "development_indices",
    "data",
]

missing_objects = [
    name for name in required_objects
    if name not in globals()
]

if missing_objects:
    raise RuntimeError(
        f"Missing objects: {missing_objects}. "
        "Complete the expanded Block 11 first."
    )

FIGURES_FOLDER = os.path.join(PROJECT_FOLDER, "Figures")
RESULTS_FOLDER = os.path.join(PROJECT_FOLDER, "Results")

os.makedirs(FIGURES_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

SHAP_RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

SHAP_RESULTS_FOLDER = os.path.join(
    RESULTS_FOLDER,
    f"block12_{SHAP_RUN_ID}",
)

os.makedirs(SHAP_RESULTS_FOLDER, exist_ok=False)

# These are explicitly marked as physics descriptors in Block 5.
physics_feature_names = [
    "phys_dopant_cation_fraction",
    "phys_mean_dopant_valence",
    "phys_nominal_vacancies_per_cation",
    "phys_radius_mismatch_A",
    "phys_cation_radius_variance_A2",
]


# ============================================================
# 12.2 FIXING SHARED DEVELOPMENT-DATA SAMPLES
# ============================================================

# Row-weighted sampling: each measurement has equal probability.
# Save the identities so the explanations are reproducible.
shap_sample_indices = {}
sample_manifest_frames = []

for analysis_number, analysis in enumerate(
    sorted(development_indices)
):

    indices = development_indices[analysis]

    rng = np.random.default_rng(
        SHAP_RANDOM_STATE + analysis_number
    )

    background_positions = np.sort(
        rng.choice(
            len(indices),
            size=min(SHAP_BACKGROUND_ROWS, len(indices)),
            replace=False,
        )
    )

    explain_positions = np.sort(
        rng.choice(
            len(indices),
            size=min(SHAP_EXPLAIN_ROWS, len(indices)),
            replace=False,
        )
    )

    shap_sample_indices[analysis] = {
        "background": indices.take(background_positions),
        "explain": indices.take(explain_positions),
    }

    for role, selected_indices in shap_sample_indices[analysis].items():

        manifest = data.loc[
            selected_indices,
            [
                "excel_row",
                "formula",
                "paper_id",
                "composition_group",
                "temperature_K",
            ],
        ].copy()

        manifest.insert(0, "Analysis", analysis)
        manifest.insert(1, "Role", role)

        sample_manifest_frames.append(
            manifest.reset_index(drop=True)
        )

shap_sample_manifest = pd.concat(
    sample_manifest_frames,
    ignore_index=True,
)

shap_sample_manifest.to_csv(
    os.path.join(
        SHAP_RESULTS_FOLDER,
        "SHAP_development_sample_manifest.csv",
    ),
    index=False,
)


# ============================================================
# 12.3 HELPERS
# ============================================================

def save_interpretation_figure(fig, filename):
    for extension in ("png", "pdf"):
        fig.savefig(
            os.path.join(
                FIGURES_FOLDER,
                f"Block12_{SHAP_RUN_ID}_{filename}.{extension}",
            ),
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )


def readable_feature_name(name):
    """Display labels only; original feature names are preserved."""
    labels = {
        "temperature_K": "Operating temperature (K)",
        "sintering_temperature_K": "Sintering temperature (K)",
        "sintering_time_h": "Sintering time (h)",
        "phys_dopant_cation_fraction": "Dopant cation fraction",
        "phys_mean_dopant_valence": "Mean dopant valence",
        "phys_nominal_vacancies_per_cation": (
            "Nominal oxygen vacancies per cation"
        ),
        "phys_radius_mismatch_A": "Ionic-radius mismatch (Å)",
        "phys_cation_radius_variance_A2": (
            "Cation-radius variance (Å²)"
        ),
    }

    return labels.get(
        name,
        name.replace("chem__", "").replace("MagpieData ", ""),
    )


# These selected final predictors have dedicated SHAP explainers.
tree_model_names = {
    "RandomForest",
    "ExtraTrees",
    "GradientBoosting",
    "HistGradientBoosting",
    "DecisionTree",
    "XGBoost",
}

linear_model_names = {
    "Ridge",
    "LinearRegression",
    "Lasso",
    "ElasticNet",
}


# ============================================================
# 12.4 EXPLAINING EACH VALIDATION-SELECTED FINAL PREDICTOR
# ============================================================

final_shap_explanations = {}
final_shap_rankings = {}

importance_frames = []
retention_rows = []
explanation_diagnostics = []

for pair in sorted(primary_final_models):

    analysis, feature_set = pair
    model_name = selected_model_keys[pair][2]

    pipeline = primary_final_models[pair]
    predictor = pipeline.named_steps["model"]

    feature_frame = analysis_features[analysis][feature_set]

    background_idx = shap_sample_indices[analysis]["background"]
    explain_idx = shap_sample_indices[analysis]["explain"]

    X_background_raw = feature_frame.loc[background_idx]
    X_explain_raw = feature_frame.loc[explain_idx]

    # Transform with already fitted preprocessing.
    # No fit() or fit_transform() calls are made.
    preprocessing = pipeline[:-1]

    X_background_model = preprocessing.transform(X_background_raw)
    X_explain_model = preprocessing.transform(X_explain_raw)

    if not isinstance(X_explain_model, pd.DataFrame):
        raise TypeError(
            "Expected named DataFrame output from the fitted pipeline."
        )

    selected_features = X_explain_model.columns.tolist()

    if not X_background_model.columns.equals(
        X_explain_model.columns
    ):
        raise ValueError("Background/explanation feature mismatch.")

    if not np.isfinite(
        X_explain_model.to_numpy(dtype=float)
    ).all():
        raise ValueError(f"{pair}: invalid transformed inputs.")

    if not np.isfinite(
        X_background_model.to_numpy(dtype=float)
    ).all():
        raise ValueError(f"{pair}: invalid background inputs.")

    stem = f"{analysis}_set_{feature_set}_{model_name}"

    print(
        f"\nExplaining {analysis} | Set {feature_set} | {model_name}",
        flush=True,
    )

    # --------------------------------------------------------
    # CHOOSING AN EXPLAINER FOR THE ACTUAL PREDICTOR
    # --------------------------------------------------------

    if model_name in tree_model_names:

        explainer = shap.TreeExplainer(
            predictor,
            feature_perturbation="tree_path_dependent",
            model_output="raw",
        )

        explanation = explainer(
            X_explain_model,
            check_additivity=True,
        )

        explanation_method = ("Tree SHAP, tree-path-dependent;"
                              "background represented by fitting tree-path statistics")

    elif model_name in linear_model_names:

        masker = shap.maskers.Independent(
            X_background_model,
            max_samples=len(X_background_model),
        )

        explainer = shap.LinearExplainer(
            predictor,
            masker,
        )

        explanation = explainer(X_explain_model)

        explanation_method = "Linear SHAP, independent masker"

    else:

        # SVR, ANN, KNN, GPR and other supported regressors.
        # Restore feature names when SHAP passes numpy arrays.
        def predict_for_shap(values):
            frame = pd.DataFrame(
                values,
                columns=selected_features,
            )
            return np.asarray(
                predictor.predict(frame),
                dtype=float,
            ).reshape(-1)

        masker = shap.maskers.Independent(
            X_background_model,
            max_samples=len(X_background_model),
        )

        explainer = shap.PermutationExplainer(
            predict_for_shap,
            masker,
            seed=SHAP_RANDOM_STATE,
        )

        explanation = explainer(
            X_explain_model,
            max_evals=(
                SHAP_PERMUTATION_CYCLES
                * (2 * len(selected_features) + 1)
            ),
        )

        explanation_method = "Permutation SHAP, independent masker"

    # --------------------------------------------------------
    # VERIFYING EXPLANATION SHAPE AND ADDITIVITY
    # --------------------------------------------------------

    shap_values = np.asarray(explanation.values, dtype=float)

    if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
        shap_values = shap_values[:, :, 0]

    if shap_values.shape != X_explain_model.shape:
        raise ValueError(
            f"{pair}: unexpected SHAP shape {shap_values.shape}."
        )

    base_values = np.asarray(
        explanation.base_values,
        dtype=float,
    ).reshape(-1)

    if base_values.size == 1:
        base_values = np.repeat(base_values, len(X_explain_model))

    if base_values.size != len(X_explain_model):
        raise ValueError(f"{pair}: unexpected base-value shape.")

    model_predictions = np.asarray(
        predictor.predict(X_explain_model),
        dtype=float,
    ).reshape(-1)

    reconstructed_predictions = (
        base_values + shap_values.sum(axis=1)
    )

    maximum_additivity_error = np.max(
        np.abs(model_predictions - reconstructed_predictions)
    )

    if not (
        np.isfinite(shap_values).all()
        and np.isfinite(base_values).all()
    ):
        raise ValueError(f"{pair}: non-finite SHAP explanation.")

    if not np.allclose(
        model_predictions,
        reconstructed_predictions,
        rtol=1e-4,
        atol=1e-4,
    ):
        raise ValueError(
            f"{pair}: SHAP values do not reconstruct predictions. "
            f"Maximum error: {maximum_additivity_error}"
        )

    # --------------------------------------------------------
    # RESTORING ORIGINAL UNITS FOR DISPLAY
    # --------------------------------------------------------

    # Reverse standardization, retaining fitted imputed values.
    scaler = pipeline.named_steps["scaler"]

    positions = scaler.feature_names_in_.tolist()

    display_values = pd.DataFrame(
        {
            feature: (
                X_explain_model[feature].to_numpy()
                * scaler.scale_[positions.index(feature)]
                + scaler.mean_[positions.index(feature)]
            )
            for feature in selected_features
        },
        index=explain_idx,
    )

    display_names = [
        readable_feature_name(feature)
        for feature in selected_features
    ]

    display_explanation = shap.Explanation(
        values=shap_values,
        base_values=base_values,
        data=display_values.to_numpy(),
        feature_names=display_names,
    )

    final_shap_explanations[pair] = display_explanation

    # --------------------------------------------------------
    # SAVING VALUES, RANKINGS AND DIAGNOSTICS
    # --------------------------------------------------------

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    ranking = pd.DataFrame({
        "Feature": selected_features,
        "Display_name": display_names,
        "Mean_absolute_SHAP": mean_abs_shap,
    }).sort_values(
        "Mean_absolute_SHAP",
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)

    ranking.insert(0, "Rank", np.arange(1, len(ranking) + 1))
    ranking.insert(0, "Model", model_name)
    ranking.insert(0, "Feature_set", feature_set)
    ranking.insert(0, "Analysis", analysis)

    final_shap_rankings[pair] = ranking
    importance_frames.append(ranking)

    shap_value_table = pd.DataFrame(
        shap_values,
        columns=selected_features,
    )

    shap_value_table.insert(
        0,
        "excel_row",
        data.loc[explain_idx, "excel_row"].to_numpy(),
    )

    shap_value_table.insert(1, "Base_value", base_values)
    shap_value_table.insert(2, "Prediction", model_predictions)

    shap_value_table.to_csv(
        os.path.join(
            SHAP_RESULTS_FOLDER,
            f"{stem}_SHAP_values.csv",
        ),
        index=False,
    )

    display_table = display_values.reset_index(drop=True)
    display_table.insert(
        0,
        "excel_row",
        data.loc[explain_idx, "excel_row"].to_numpy(),
    )

    display_table.to_csv(
        os.path.join(
            SHAP_RESULTS_FOLDER,
            f"{stem}_explained_feature_values_original_units.csv",
        ),
        index=False,
    )

    explanation_diagnostics.append({
        "Analysis": analysis,
        "Feature_set": feature_set,
        "Model": model_name,
        "Method": explanation_method,
        "Background_rows": (
            0
            if model_name in tree_model_names
            else len(X_background_model)
        ),
        "Explained_rows": len(X_explain_model),
        "Explained_compositions": data.loc[
            explain_idx, "composition_group"
        ].nunique(),
        "Selected_features": len(selected_features),
        "Maximum_additivity_error": maximum_additivity_error,
    })

    # --------------------------------------------------------
    # PHYSICS-FEATURE RETENTION FOR SET C
    # --------------------------------------------------------

    if feature_set == "C":

        constant_filter = pipeline.named_steps["constant_filter"]

        nonconstant_features = set(
            constant_filter.get_feature_names_out()
        )

        importance_lookup = dict(
            zip(selected_features, mean_abs_shap)
        )

        for feature in physics_feature_names:

            if feature not in feature_frame.columns:
                status = "Not supplied"
            elif feature not in nonconstant_features:
                status = "Removed as constant during development fit"
            elif feature not in selected_features:
                status = "Not retained by SHAP selector"
            else:
                status = "Retained"

            retention_rows.append({
                "Analysis": analysis,
                "Feature_set": feature_set,
                "Model": model_name,
                "Physics_feature": feature,
                "Status": status,
                # Unselected features have no final-model SHAP value.
                "Mean_absolute_SHAP_if_retained": (
                    importance_lookup.get(feature, np.nan)
                ),
            })

    # --------------------------------------------------------
    # FIGURE 1: BEESWARM
    # --------------------------------------------------------

    shap.plots.beeswarm(
        display_explanation,
        max_display=min(
            SHAP_DISPLAY_FEATURES,
            len(selected_features),
        ),
        show=False,
    )

    fig = plt.gcf()
    plt.title(
        f"{analysis.capitalize()} — Set {feature_set} — {model_name}\n"
        "Final-predictor SHAP: development observations"
    )

    save_interpretation_figure(fig, f"{stem}_SHAP_beeswarm")

    plt.show()
    plt.close(fig)

    # --------------------------------------------------------
    # FIGURE 2: MEAN ABSOLUTE SHAP BAR PLOT
    # --------------------------------------------------------

    shap.plots.bar(
        display_explanation,
        max_display=min(
            SHAP_DISPLAY_FEATURES,
            len(selected_features),
        ),
        show=False,
    )

    fig = plt.gcf()
    plt.title(
        f"{analysis.capitalize()} — Set {feature_set} — {model_name}\n"
        "Final-predictor mean absolute SHAP"
    )

    save_interpretation_figure(fig, f"{stem}_SHAP_bar")

    plt.show()
    plt.close(fig)

    # --------------------------------------------------------
    # FIGURE 3: SELECTED DEPENDENCE PLOTS
    # --------------------------------------------------------

    # Prioritize physically relevant retained features.
    # Within that category, order by final-model SHAP importance.
    scientifically_relevant = {
        "temperature_K",
        "sintering_temperature_K",
        "sintering_time_h",
        *physics_feature_names,
    }

    ranked_features = ranking["Feature"].tolist()

    priority_features = [
        feature for feature in ranked_features
        if feature in scientifically_relevant
    ]

    other_features = [
        feature for feature in ranked_features
        if feature not in scientifically_relevant
    ]

    dependence_features = (
        priority_features + other_features
    )[:SHAP_MAX_DEPENDENCE_PLOTS]

    for plot_number, feature in enumerate(
        dependence_features,
        start=1,
    ):

        feature_position = selected_features.index(feature)

        fig, ax = plt.subplots(
            figsize=(7, 5),
            layout="constrained",
        )

        ax.scatter(
            display_values[feature],
            shap_values[:, feature_position],
            s=28,
            alpha=0.7,
            color="#0072B2",
            edgecolors="none",
        )

        ax.axhline(
            0,
            linestyle="--",
            color="gray",
            linewidth=1,
        )

        ax.set_xlabel(readable_feature_name(feature))
        ax.set_ylabel(
            "SHAP contribution to predicted log10 conductivity"
        )

        ax.set_title(
            f"{analysis.capitalize()} — Set {feature_set} — {model_name}"
        )

        ax.grid(alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        save_interpretation_figure(
            fig,
            f"{stem}_dependence_{plot_number:02d}",
        )

        # Saved but not displayed to avoid excessive output.
        plt.close(fig)

    print(
        f"Completed using {explanation_method}; "
        f"maximum additivity error = "
        f"{maximum_additivity_error:.3g}",
        flush=True,
    )


# ============================================================
# 12.5 SAVING COMBINED TABLES AND SETTINGS
# ============================================================

final_shap_importance_table = pd.concat(
    importance_frames,
    ignore_index=True,
)

physics_retention_table = pd.DataFrame(retention_rows)

shap_diagnostics_table = pd.DataFrame(explanation_diagnostics)

final_shap_importance_table.to_csv(
    os.path.join(
        SHAP_RESULTS_FOLDER,
        "final_predictor_SHAP_importance.csv",
    ),
    index=False,
)

physics_retention_table.to_csv(
    os.path.join(
        SHAP_RESULTS_FOLDER,
        "physics_feature_retention.csv",
    ),
    index=False,
)

shap_diagnostics_table.to_csv(
    os.path.join(
        SHAP_RESULTS_FOLDER,
        "SHAP_explanation_diagnostics.csv",
    ),
    index=False,
)

with open(
    os.path.join(
        SHAP_RESULTS_FOLDER,
        "SHAP_settings.json",
    ),
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        {
            "background_rows_max": SHAP_BACKGROUND_ROWS,
            "explained_rows_max": SHAP_EXPLAIN_ROWS,
            "permutation_cycles": SHAP_PERMUTATION_CYCLES,
            "random_state": SHAP_RANDOM_STATE,
            "sampling": "Uniform measurement-row sampling",
            "data_scope": "Training plus validation only",
            "test_data_used": False,
            "models_retrained": False,
            "display_units": (
                "Original feature units; fitted imputation retained"
            ),
            "interpretation": (
                "Tree predictors: tree-path-dependent SHAP using"
                "fitted development-data tree-path statistics."
                "Model associations under interventional/independent "
                "feature masking; not causal effects."
            ),
        },
        file,
        indent=2,
    )

print("\nEXPLANATION DIAGNOSTICS")
print(shap_diagnostics_table.to_string(index=False))

print("\nPHYSICS-FEATURE RETENTION")
print(physics_retention_table.to_string(index=False))

print("\nBLOCK 12 COMPLETE")
print("Tables:", SHAP_RESULTS_FOLDER)
print("Figures:", FIGURES_FOLDER)
print("Final predictors explained; no models retrained.")
print("Test observations were not used.")






# %% BLOCK 13 — BOOTSTRAP VARIABILITY AND APPLICABILITY
#
# Development data = training + validation.
# No test observations or test outcomes are used.
#
# Original final models remain unchanged.
#
# Outputs:
#   Results: bootstrap membership and applicability references
#   Figures: reference-distance distributions
#
# Retained in memory:
#   bootstrap_models
#   applicability_models
#   predict_with_reliability()





# ============================================================
# 13.1 SETTINGS AND REQUIRED OBJECTS
# ============================================================

N_BOOTSTRAP_MODELS = 30
BOOTSTRAP_RANDOM_STATE = RANDOM_STATE

# Heuristic reference threshold, not a confidence level.
APPLICABILITY_QUANTILE = 0.95

required_objects = [
    "primary_final_models",
    "selected_model_keys",
    "analysis_features",
    "development_indices",
    "target_all",
    "data",
]

missing_objects = [
    name for name in required_objects
    if name not in globals()
]

if missing_objects:
    raise RuntimeError(
        f"Missing objects: {missing_objects}. "
        "Complete the expanded Block 11 first."
    )

if N_BOOTSTRAP_MODELS < 2:
    raise ValueError("At least two bootstrap models are required.")

FIGURES_FOLDER = os.path.join(PROJECT_FOLDER, "Figures")
RESULTS_FOLDER = os.path.join(PROJECT_FOLDER, "Results")

os.makedirs(FIGURES_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

RELIABILITY_RUN_ID = datetime.now().strftime(
    "%Y%m%d_%H%M%S_%f"
)

RELIABILITY_RESULTS_FOLDER = os.path.join(
    RESULTS_FOLDER,
    f"block14_{RELIABILITY_RUN_ID}",
)

os.makedirs(RELIABILITY_RESULTS_FOLDER, exist_ok=False)


# ============================================================
# 13.2 DRAWING COMPOSITION-GROUP BOOTSTRAP SAMPLES
# ============================================================

# Draw whole composition groups with replacement.
# Each selected group contributes all its measurement rows.
#
# The same draws are reused across A/B/C within an analysis.
# This preserves the grouped data structure and paired design.

bootstrap_draws = {}
membership_rows = []

for analysis_number, analysis in enumerate(
    sorted(development_indices)
):

    development_idx = development_indices[analysis]

    groups = data.loc[
        development_idx, "composition_group"
    ].astype(str)

    unique_groups = np.sort(groups.unique())

    if len(unique_groups) < 2:
        raise ValueError(
            f"{analysis}: insufficient development compositions."
        )

    positions_by_group = {
        group: np.flatnonzero(groups.to_numpy() == group)
        for group in unique_groups
    }

    rng = np.random.default_rng(
        BOOTSTRAP_RANDOM_STATE + analysis_number
    )

    bootstrap_draws[analysis] = []

    for bootstrap_number in range(N_BOOTSTRAP_MODELS):

        sampled_groups = rng.choice(
            unique_groups,
            size=len(unique_groups),
            replace=True,
        )

        sampled_positions = np.concatenate([
            positions_by_group[group]
            for group in sampled_groups
        ])

        bootstrap_draws[analysis].append(sampled_positions)

        multiplicities = pd.Series(
            sampled_groups
        ).value_counts()

        for group in unique_groups:
            membership_rows.append({
                "Analysis": analysis,
                "Bootstrap": bootstrap_number + 1,
                "Composition_group": group,
                "Times_sampled": int(
                    multiplicities.get(group, 0)
                ),
            })

bootstrap_membership = pd.DataFrame(membership_rows)

bootstrap_membership.to_csv(
    os.path.join(
        RELIABILITY_RESULTS_FOLDER,
        "bootstrap_composition_membership.csv",
    ),
    index=False,
)


# ============================================================
# 13.3 REFITTING BOOTSTRAP COPIES OF THE PRIMARY PIPELINES
# ============================================================

bootstrap_models = {}

for pair in sorted(primary_final_models):

    analysis, feature_set = pair
    model_name = selected_model_keys[pair][2]

    development_idx = development_indices[analysis]

    X_development = analysis_features[analysis][feature_set].loc[
        development_idx
    ]

    y_development = target_all.loc[development_idx]

    bootstrap_models[pair] = []

    for bootstrap_number, positions in enumerate(
        bootstrap_draws[analysis],
        start=1,
    ):

        print(
            f"{analysis} | Set {feature_set} | {model_name} | "
            f"bootstrap {bootstrap_number}/{N_BOOTSTRAP_MODELS}",
            flush=True,
        )

        # Reset indices identically because groups can be repeated.
        X_bootstrap = X_development.iloc[
            positions
        ].reset_index(drop=True)

        y_bootstrap = y_development.iloc[
            positions
        ].reset_index(drop=True)

        estimator = clone(primary_final_models[pair])
        estimator.set_params(memory=None)

        # Refit preprocessing and SHAP selection on this resample.
        # Keep the previously selected hyperparameters and seeds.
        estimator.fit(X_bootstrap, y_bootstrap)

        bootstrap_models[pair].append(estimator)


# ============================================================
# 13.4 BUILDING APPLICABILITY REFERENCE MODELS
# ============================================================

applicability_models = {}
distance_reference_frames = []
applicability_summary_rows = []

for pair in sorted(primary_final_models):

    analysis, feature_set = pair

    development_idx = development_indices[analysis]
    X_development = analysis_features[analysis][feature_set].loc[
        development_idx
    ]

    pipeline = primary_final_models[pair]

    # Use all nonconstant standardized input features BEFORE
    # SHAP selection, so an unselected descriptor is not invisible
    # to the applicability check.
    distance_preprocessor = pipeline[:3]

    transformed = distance_preprocessor.transform(X_development)
    values = np.asarray(transformed, dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(f"{pair}: invalid applicability inputs.")

    groups = data.loc[
        development_idx, "composition_group"
    ].astype(str).to_numpy()

    # For each development observation, find the nearest
    # observation belonging to a DIFFERENT composition.
    # This avoids trivially small distances to another temperature
    # point from the same composition.
    reference_distances = np.full(len(values), np.nan)

    for group in np.unique(groups):

        query_mask = groups == group
        other_mask = ~query_mask

        neighbor_model = NearestNeighbors(
            n_neighbors=1,
            metric="euclidean",
        ).fit(values[other_mask])

        distances, _ = neighbor_model.kneighbors(
            values[query_mask]
        )

        reference_distances[query_mask] = distances[:, 0]

    if not np.isfinite(reference_distances).all():
        raise ValueError(f"{pair}: invalid reference distances.")

    distance_threshold = float(
        np.quantile(
            reference_distances,
            APPLICABILITY_QUANTILE,
        )
    )

    # New candidates are compared against all development rows.
    all_neighbors = NearestNeighbors(
        n_neighbors=1,
        metric="euclidean",
    ).fit(values)

    applicability_models[pair] = {
        "preprocessor": distance_preprocessor,
        "neighbors": all_neighbors,
        "distance_threshold": distance_threshold,
        "input_columns": X_development.columns.tolist(),
        "raw_min": X_development.min(),
        "raw_max": X_development.max(),
    }

    reference_table = data.loc[
        development_idx,
        ["excel_row", "formula", "composition_group"],
    ].copy()

    reference_table.insert(0, "Analysis", analysis)
    reference_table.insert(1, "Feature_set", feature_set)

    reference_table["Nearest_other_composition_distance"] = (
        reference_distances
    )

    distance_reference_frames.append(
        reference_table.reset_index(drop=True)
    )

    applicability_summary_rows.append({
        "Analysis": analysis,
        "Feature_set": feature_set,
        "Development_rows": len(values),
        "Distance_features": values.shape[1],
        "Reference_quantile": APPLICABILITY_QUANTILE,
        "Heuristic_distance_threshold": distance_threshold,
    })

    fig, ax = plt.subplots(
        figsize=(8, 5),
        layout="constrained",
    )

    ax.hist(
        reference_distances,
        bins=25,
        color="#0072B2",
        alpha=0.8,
        edgecolor="white",
    )

    ax.axvline(
        distance_threshold,
        color="#D55E00",
        linestyle="--",
        label=f"95th-percentile reference = {distance_threshold:.2f}",
    )

    ax.set_xlabel(
        "Distance to nearest different development composition"
    )
    ax.set_ylabel("Measurement rows")
    ax.set_title(
        f"{analysis.capitalize()} — Set {feature_set}\n"
        "Standardized descriptor-distance reference"
    )

    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for extension in ("png", "pdf"):
        fig.savefig(
            os.path.join(
                FIGURES_FOLDER,
                f"Block13_{RELIABILITY_RUN_ID}_"
                f"{analysis}_set_{feature_set}_"
                f"distance_reference.{extension}",
            ),
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()
    plt.close(fig)


# ============================================================
# 13.5 FUNCTION FOR FUTURE CANDIDATE SCREENING
# ============================================================

def predict_with_reliability(candidate_features, analysis, feature_set):
    """
    Predict using the frozen primary model and bootstrap copies.

    Input:
        DataFrame with the same named input features as the
        corresponding A/B/C dataset.

    Returns:
        Original-model prediction, bootstrap variability,
        descriptor distance and raw-feature range flags.

    Bootstrap quantiles are NOT calibrated prediction intervals.
    """

    pair = (analysis, feature_set)

    if pair not in bootstrap_models:
        raise KeyError(f"No bootstrap ensemble for {pair}.")

    if not isinstance(candidate_features, pd.DataFrame):
        raise TypeError("candidate_features must be a DataFrame.")

    if candidate_features.empty:
        raise ValueError("No candidate rows supplied.")

    if not candidate_features.columns.is_unique:
        raise ValueError("Candidate feature names must be unique.")

    reference = applicability_models[pair]
    required = reference["input_columns"]

    missing = set(required) - set(candidate_features.columns)

    if missing:
        raise ValueError(f"Missing candidate features: {sorted(missing)}")

    X_candidate = candidate_features.loc[:, required].copy()

    # Missing features could conceal applicability problems.
    # Require complete candidate descriptors for this screening.
    if not np.isfinite(
        X_candidate.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Candidate descriptors must be complete and finite."
        )

    original_prediction = primary_final_models[pair].predict(
        X_candidate
    )

    ensemble_predictions = np.column_stack([
        estimator.predict(X_candidate)
        for estimator in bootstrap_models[pair]
    ])

    if not np.isfinite(ensemble_predictions).all():
        raise ValueError("A bootstrap model returned invalid predictions.")

    transformed_candidate = reference["preprocessor"].transform(
        X_candidate
    )

    distances, _ = reference["neighbors"].kneighbors(
        np.asarray(transformed_candidate, dtype=float)
    )

    distances = distances[:, 0]

    outside_range = (
        X_candidate.lt(reference["raw_min"], axis="columns")
        | X_candidate.gt(reference["raw_max"], axis="columns")
    )

    output = pd.DataFrame(
        {
            "Predicted_log10_sigma": original_prediction,
            "Bootstrap_mean_log10_sigma": (
                ensemble_predictions.mean(axis=1)
            ),
            "Bootstrap_SD_log10_sigma": (
                ensemble_predictions.std(axis=1, ddof=1)
            ),
            "Bootstrap_prediction_q05": np.quantile(
                ensemble_predictions, 0.05, axis=1
            ),
            "Bootstrap_prediction_q95": np.quantile(
                ensemble_predictions, 0.95, axis=1
            ),
            "Nearest_development_distance": distances,
            "Beyond_distance_reference": (
                distances > reference["distance_threshold"]
            ),
            "Features_outside_development_range": (
                outside_range.sum(axis=1).to_numpy()
            ),
        },
        index=X_candidate.index,
    )

    output["Outside_range_feature_names"] = [
        "; ".join(X_candidate.columns[row_flags])
        for row_flags in outside_range.to_numpy()
    ]

    return output


# ============================================================
# 13.6 SAVE REFERENCE TABLES AND METHOD SETTINGS
# ============================================================

applicability_summary = pd.DataFrame(
    applicability_summary_rows
)

applicability_reference = pd.concat(
    distance_reference_frames,
    ignore_index=True,
)

applicability_summary.to_csv(
    os.path.join(
        RELIABILITY_RESULTS_FOLDER,
        "applicability_summary.csv",
    ),
    index=False,
)

applicability_reference.to_csv(
    os.path.join(
        RELIABILITY_RESULTS_FOLDER,
        "development_distance_reference.csv",
    ),
    index=False,
)

with open(
    os.path.join(
        RELIABILITY_RESULTS_FOLDER,
        "reliability_settings.json",
    ),
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        {
            "bootstrap_models_per_pipeline": N_BOOTSTRAP_MODELS,
            "bootstrap_unit": "Composition group",
            "bootstrap_seed": BOOTSTRAP_RANDOM_STATE,
            "preprocessing_and_selection_refitted": True,
            "hyperparameters_retuned": False,
            "data_used": "Training plus validation",
            "test_used": False,
            "distance_space": (
                "All nonconstant standardized inputs before SHAP selection"
            ),
            "distance_reference": (
                "Nearest other composition for each development row"
            ),
            "distance_quantile": APPLICABILITY_QUANTILE,
            "uncertainty_interpretation": (
                "Bootstrap model variability, conditional on selected "
                "model family and hyperparameters; not calibrated "
                "prediction intervals."
            ),
        },
        file,
        indent=2,
    )

print("\nAPPLICABILITY REFERENCES")
print(applicability_summary.to_string(index=False))

print("\nBLOCK 13 COMPLETE")
print("Results:", RELIABILITY_RESULTS_FOLDER)
print("Figures:", FIGURES_FOLDER)
print("Bootstrap models remain in bootstrap_models.")
print("Candidate helper: predict_with_reliability(...)")
print("Original final predictors and test results are unchanged.")








# %% BLOCK 14 — CANDIDATE GENERATION AND MATMINER FEATURES

#
# No model fitting, prediction or ranking.
#
# Candidate formula convention:
# Zr_(1-y-sc) Y_y Sc_sc O_(2 - (y+sc)/2)
#
# Assumptions:
# Zr4+, Y3+, Sc3+, O2-
# All candidate dopants occupy the nominal cation sublattice.
#
# Fractions below are CATION fractions, not oxide mole fractions.
# Original dataset formulas are not modified.




# ============================================================
# 14.1 EDITABLE SEARCH-SPACE SETTINGS
# ============================================================

CANDIDATE_TEMPERATURE_K = 973.15

# Illustrative grid, not experimentally established bounds.
CANDIDATE_DOPANT_FRACTIONS = [0.04, 0.08, 0.12, 0.16]

MAX_TOTAL_DOPANT_CATION_FRACTION = 0.20

INCLUDE_Y_ONLY = True
INCLUDE_SC_ONLY = True
INCLUDE_Y_SC_MIXTURES = True

# Retain known compositions for reference, but label them clearly.
# Set False to generate only compositions absent from the dataset.
INCLUDE_KNOWN_COMPOSITIONS = True

# Guard against unexpectedly large candidate libraries.
MAX_CANDIDATE_ROWS = 20000

required_objects = [
    "data",
    "development_indices",
    "analysis_features",
    "chemical_featurizer",
    "X_chemical",
]

missing_objects = [
    name for name in required_objects
    if name not in globals()
]

if missing_objects:
    raise RuntimeError(
        f"Missing objects: {missing_objects}. "
        "Run the preceding feature-generation and modelling blocks."
    )

if CANDIDATE_TEMPERATURE_K <= 0:
    raise ValueError("Operating temperature must be positive.")

if not 0 < MAX_TOTAL_DOPANT_CATION_FRACTION < 1:
    raise ValueError("Invalid maximum dopant cation fraction.")

if any(
    not 0 < fraction < 1
    for fraction in CANDIDATE_DOPANT_FRACTIONS
):
    raise ValueError("Dopant fractions must lie between zero and one.")

FIGURES_FOLDER = os.path.join(PROJECT_FOLDER, "Figures")
RESULTS_FOLDER = os.path.join(PROJECT_FOLDER, "Results")

os.makedirs(FIGURES_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

CANDIDATE_RUN_ID = datetime.now().strftime(
    "%Y%m%d_%H%M%S_%f"
)

CANDIDATE_RESULTS_FOLDER = os.path.join(
    RESULTS_FOLDER,
    f"block15_{CANDIDATE_RUN_ID}",
)

os.makedirs(CANDIDATE_RESULTS_FOLDER, exist_ok=False)


# ============================================================
# 14.2 HELPERS FOR FORMULAS AND COMPOSITION IDENTITIES
# ============================================================

def candidate_formula(zr_fraction, y_fraction, sc_fraction):
    """
    Express candidates on a one-total-cation basis.
    This does not normalize or change the original dataset.
    """
    oxygen_coefficient = 2.0 - 0.5 * (
        y_fraction + sc_fraction
    )

    coefficients = [
        ("Zr", zr_fraction),
        ("Y", y_fraction),
        ("Sc", sc_fraction),
        ("O", oxygen_coefficient),
    ]

    formula = "".join(
        f"{element}{amount:.10g}"
        for element, amount in coefficients
        if amount > 0
    )

    return formula, oxygen_coefficient


def candidate_composition_key(formula):
    """
    Cation-fraction identity using the same seven-decimal
    convention as the composition grouping in Block 4.
    """
    composition = Composition(formula)
    amounts = composition.get_el_amt_dict()

    cations = {
        element: amount
        for element, amount in amounts.items()
        if element != "O"
    }

    total = sum(cations.values())

    if total <= 0:
        raise ValueError(f"No cations found in {formula}")

    return "|".join(
        f"{element}:{cations[element] / total:.7f}"
        for element in sorted(cations)
    )


# ============================================================
# 14.3 GENERATE THE COMPOSITION GRID
# ============================================================

fraction_pairs = []

if INCLUDE_Y_ONLY:
    fraction_pairs.extend(
        (fraction, 0.0)
        for fraction in CANDIDATE_DOPANT_FRACTIONS
    )

if INCLUDE_SC_ONLY:
    fraction_pairs.extend(
        (0.0, fraction)
        for fraction in CANDIDATE_DOPANT_FRACTIONS
    )

if INCLUDE_Y_SC_MIXTURES:
    fraction_pairs.extend(
        (y_fraction, sc_fraction)
        for y_fraction in CANDIDATE_DOPANT_FRACTIONS
        for sc_fraction in CANDIDATE_DOPANT_FRACTIONS
    )

fraction_pairs = sorted(set(fraction_pairs))

# Metadata-only comparison. No conductivity targets are accessed.
known_composition_keys = {
    candidate_composition_key(formula)
    for formula in data["formula"].dropna().unique()
}

development_idx = development_indices["full"]

development_composition_keys = {
    candidate_composition_key(formula)
    for formula in data.loc[
        development_idx, "formula"
    ].dropna().unique()
}

composition_records = []

for y_fraction, sc_fraction in fraction_pairs:

    total_dopant = y_fraction + sc_fraction

    if total_dopant > (
        MAX_TOTAL_DOPANT_CATION_FRACTION + 1e-12
    ):
        continue

    zr_fraction = 1.0 - total_dopant

    if zr_fraction <= 0:
        continue

    formula, oxygen_coefficient = candidate_formula(
        zr_fraction,
        y_fraction,
        sc_fraction,
    )

    composition_key = candidate_composition_key(formula)

    known_in_dataset = composition_key in known_composition_keys

    if not INCLUDE_KNOWN_COMPOSITIONS and known_in_dataset:
        continue

    if y_fraction > 0 and sc_fraction > 0:
        family = "Y-Sc co-stabilized"
    elif y_fraction > 0:
        family = "Y-stabilized"
    else:
        family = "Sc-stabilized"

    # Formal charge-balance check under the declared assumptions.
    formal_positive_charge = (
        4 * zr_fraction
        + 3 * y_fraction
        + 3 * sc_fraction
    )

    residual = formal_positive_charge - 2 * oxygen_coefficient

    if not np.isclose(residual, 0.0, atol=1e-10):
        raise ValueError(f"Charge-balance failure: {formula}")

    composition_records.append({
        "formula": formula,
        "composition_group": composition_key,
        "stabilizer_family": family,
        "Zr_cation_fraction": zr_fraction,
        "Y_cation_fraction": y_fraction,
        "Sc_cation_fraction": sc_fraction,
        "total_dopant_cation_fraction": total_dopant,
        "nominal_oxygen_per_cation": oxygen_coefficient,
        "known_composition_in_dataset": known_in_dataset,
        "seen_in_full_development": (
            composition_key in development_composition_keys
        ),
        "phase_stability_verified": False,
        "synthesis_feasibility_verified": False,
    })

candidate_compositions = pd.DataFrame(composition_records)

if candidate_compositions.empty:
    raise ValueError("No candidate compositions survived the settings.")

candidate_compositions = (
    candidate_compositions
    .drop_duplicates("composition_group")
    .reset_index(drop=True)
)

candidate_compositions.insert(
    0,
    "candidate_composition_id",
    [
        f"ZC{number:04d}"
        for number in range(1, len(candidate_compositions) + 1)
    ],
)


# ============================================================
# 14.4 USE OBSERVED DEVELOPMENT PROCESSING SCHEDULES
# ============================================================

schedule_columns = [
    "sintering_temperature_K",
    "sintering_time_h",
]

schedule_source = data.loc[
    development_idx,
    ["excel_row", "paper_id"] + schedule_columns,
].copy()

valid_schedule = (
    schedule_source[schedule_columns].notna().all(axis=1)
    & np.isfinite(
        schedule_source[schedule_columns].to_numpy(dtype=float)
    ).all(axis=1)
    & schedule_source["sintering_temperature_K"].gt(0)
    & schedule_source["sintering_time_h"].gt(0)
)

schedule_source = schedule_source.loc[valid_schedule]

candidate_schedules = (
    schedule_source.groupby(
        schedule_columns,
        as_index=False,
        sort=True,
    )
    .agg(
        source_excel_rows=(
            "excel_row",
            lambda values: "; ".join(
                values.astype(int).astype(str)
            ),
        ),
        source_papers=(
            "paper_id",
            lambda values: "; ".join(
                sorted(values.dropna().astype(str).unique())
            ),
        ),
    )
)

if candidate_schedules.empty:
    raise ValueError("No complete positive processing schedules found.")

candidate_schedules.insert(
    0,
    "schedule_id",
    [
        f"PS{number:04d}"
        for number in range(1, len(candidate_schedules) + 1)
    ],
)

# A schedule observed for one composition is not automatically
# validated for every other candidate composition.
expected_rows = (
    len(candidate_compositions) * len(candidate_schedules)
)

if expected_rows > MAX_CANDIDATE_ROWS:
    raise ValueError(
        f"The grid would contain {expected_rows} rows. "
        "Narrow the composition grid or processing schedules "
        "before continuing."
    )

candidate_library = candidate_compositions.merge(
    candidate_schedules,
    how="cross",
)

candidate_library["temperature_K"] = CANDIDATE_TEMPERATURE_K

candidate_library.insert(
    0,
    "candidate_id",
    [
        f"ZCAND{number:06d}"
        for number in range(1, len(candidate_library) + 1)
    ],
)

candidate_library = candidate_library.set_index(
    "candidate_id",
    drop=False,
)

observed_temperature_min = data.loc[
    development_idx, "temperature_K"
].min()

observed_temperature_max = data.loc[
    development_idx, "temperature_K"
].max()

candidate_library["temperature_outside_development_range"] = not (
    observed_temperature_min
    <= CANDIDATE_TEMPERATURE_K
    <= observed_temperature_max
)


# ============================================================
# 14.5 GENERATE CHEMICAL DESCRIPTORS ONCE PER COMPOSITION
# ============================================================

# Reuse the exact featurizer from Block 4.
chemical_columns = [
    "chem__" + label
    for label in chemical_featurizer.feature_labels()
]

if chemical_columns != X_chemical.columns.tolist():
    raise ValueError(
        "The candidate featurizer does not match Block 4. "
        "Use the same preset and feature names."
    )

candidate_descriptor_lookup = {}

for number, row in enumerate(
    candidate_compositions.itertuples(index=False),
    start=1,
):

    print(
        f"Featurizing composition {number}/"
        f"{len(candidate_compositions)}: {row.formula}",
        flush=True,
    )

    candidate_descriptor_lookup[row.formula] = (
        chemical_featurizer.featurize(
            Composition(row.formula)
        )
    )

candidate_chemical_features = pd.DataFrame(
    [
        candidate_descriptor_lookup[formula]
        for formula in candidate_library["formula"]
    ],
    columns=chemical_columns,
    index=candidate_library.index,
)

if not np.isfinite(
    candidate_chemical_features.to_numpy(dtype=float)
).all():
    raise ValueError(
        "Candidate chemical descriptors contain missing or "
        "infinite values. Review before screening."
    )


# ============================================================
# 14.6 BUILD EXACT MODEL INPUTS FOR A AND B
# ============================================================

candidate_X_A = pd.concat(
    [
        candidate_chemical_features,
        candidate_library[["temperature_K"]],
    ],
    axis=1,
)

candidate_X_B = pd.concat(
    [
        candidate_X_A,
        candidate_library[
            ["sintering_temperature_K", "sintering_time_h"]
        ],
    ],
    axis=1,
)

candidate_feature_sets = {
    "A": candidate_X_A,
    "B": candidate_X_B,
}

for feature_set, frame in candidate_feature_sets.items():

    expected_columns = analysis_features["full"][
        feature_set
    ].columns.tolist()

    if set(frame.columns) != set(expected_columns):
        raise ValueError(
            f"Candidate Set {feature_set} does not match "
            "the trained model inputs."
        )

    # Match the fitted model's feature order exactly.
    candidate_feature_sets[feature_set] = frame.loc[
        :, expected_columns
    ].copy()

candidate_X_A = candidate_feature_sets["A"]
candidate_X_B = candidate_feature_sets["B"]

# Set A ignores processing. Consequently, its input rows repeat
# across schedules for the same composition.
# Keep these rows aligned with B for paired candidate reporting.


# ============================================================
# 14.7 SAVE CANDIDATE TABLES AND SEARCH-SPACE SETTINGS
# ============================================================

candidate_compositions.to_csv(
    os.path.join(
        CANDIDATE_RESULTS_FOLDER,
        "candidate_compositions.csv",
    ),
    index=False,
)

candidate_schedules.to_csv(
    os.path.join(
        CANDIDATE_RESULTS_FOLDER,
        "candidate_processing_schedules.csv",
    ),
    index=False,
)

candidate_library.to_csv(
    os.path.join(
        CANDIDATE_RESULTS_FOLDER,
        "candidate_library.csv",
    ),
    index=False,
)

for feature_set, frame in candidate_feature_sets.items():
    frame.to_csv(
        os.path.join(
            CANDIDATE_RESULTS_FOLDER,
            f"candidate_features_set_{feature_set}.csv",
        ),
        index=True,
        index_label="candidate_id",
    )

with open(
    os.path.join(
        CANDIDATE_RESULTS_FOLDER,
        "candidate_generation_settings.json",
    ),
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        {
            "operating_temperature_K": CANDIDATE_TEMPERATURE_K,
            "dopant_fraction_basis": "Fraction of total cations",
            "dopant_fraction_grid": CANDIDATE_DOPANT_FRACTIONS,
            "maximum_total_dopant_fraction": (
                MAX_TOTAL_DOPANT_CATION_FRACTION
            ),
            "include_Y_only": INCLUDE_Y_ONLY,
            "include_Sc_only": INCLUDE_SC_ONLY,
            "include_Y_Sc_mixtures": INCLUDE_Y_SC_MIXTURES,
            "include_known_compositions": INCLUDE_KNOWN_COMPOSITIONS,
            "assumed_valences": {
                "Zr": 4,
                "Y": 3,
                "Sc": 3,
                "O": -2,
            },
            "formula_basis": "One total cation",
            "processing_source": (
                "Distinct complete schedules from full development data"
            ),
            "candidate_phase_stability_verified": False,
            "candidate_synthesis_feasibility_verified": False,
            "physics_features_generated": False,
            "predictions_generated": False,
        },
        file,
        indent=2,
    )


# ============================================================
# 14.8 FIGURE — COMPOSITION SEARCH SPACE
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 6),
    layout="constrained",
)

for known_status, marker, label, color in [
    (False, "o", "Composition absent from dataset", "#0072B2"),
    (True, "s", "Composition present in dataset", "#D55E00"),
]:

    subset = candidate_compositions.loc[
        candidate_compositions[
            "known_composition_in_dataset"
        ].eq(known_status)
    ]

    if subset.empty:
        continue

    ax.scatter(
        100 * subset["Y_cation_fraction"],
        100 * subset["Sc_cation_fraction"],
        s=90,
        marker=marker,
        color=color,
        label=label,
        edgecolors="white",
    )

ax.set_xlabel("Y fraction of total cations (%)")
ax.set_ylabel("Sc fraction of total cations (%)")
ax.set_title(
    "Candidate composition grid\n"
    "Nominal charge-balanced formulas; phase stability unverified"
)

ax.grid(alpha=0.2)
ax.set_axisbelow(True)
ax.legend(fontsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

for extension in ("png", "pdf"):
    fig.savefig(
        os.path.join(
            FIGURES_FOLDER,
            f"Block15_{CANDIDATE_RUN_ID}_composition_grid.{extension}",
        ),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

plt.show()
plt.close(fig)


# ============================================================
# 14.9 SUMMARY
# ============================================================

print("\nCANDIDATE LIBRARY")
print("Unique compositions:", len(candidate_compositions))
print("Processing schedules:", len(candidate_schedules))
print("Composition–schedule combinations:", len(candidate_library))
print("Operating temperature:", CANDIDATE_TEMPERATURE_K, "K")

print(
    "Compositions absent from dataset:",
    (~candidate_compositions["known_composition_in_dataset"]).sum(),
)

print("Set A input shape:", candidate_X_A.shape)
print("Set B input shape:", candidate_X_B.shape)

print("\nBLOCK 14 COMPLETE")
print("Results:", CANDIDATE_RESULTS_FOLDER)
print("Figures:", FIGURES_FOLDER)
print("No models were fitted and no candidates were ranked.")
print("Set C awaits justified candidate physics assignments.")







# %% BLOCK 15 — CANDIDATE SCREENING AND SHORTLISTING


#
# Main screening model: full-data Set B
# Reference model: full-data Set A
#
# No model fitting, test evaluation or Bayesian optimization.
# Bootstrap spread is NOT a calibrated prediction interval.




# ============================================================
# 15.1 SETTINGS AND REQUIRED OBJECTS
# ============================================================

SCREENING_ANALYSIS = "full"
SHORTLIST_SIZE = 20
PREDICTION_BATCH_SIZE = 250

required_objects = [
    "candidate_library",
    "candidate_feature_sets",
    "candidate_compositions",
    "predict_with_reliability",
    "bootstrap_models",
    "applicability_models",
    "selected_model_keys",
    "CANDIDATE_RUN_ID",
    "RELIABILITY_RUN_ID",
]

missing_objects = [
    name for name in required_objects
    if name not in globals()
]

if missing_objects:
    raise RuntimeError(
        f"Missing objects: {missing_objects}. "
        "Complete Blocks 14 and 15 in the current kernel."
    )

for feature_set in ("A", "B"):
    pair = (SCREENING_ANALYSIS, feature_set)

    if pair not in bootstrap_models:
        raise ValueError(f"Missing bootstrap ensemble: {pair}")

    if feature_set not in candidate_feature_sets:
        raise ValueError(f"Missing candidate features: {feature_set}")

    if not candidate_feature_sets[feature_set].index.equals(
        candidate_library.index
    ):
        raise ValueError(
            f"Set {feature_set}: candidate indices are not aligned."
        )

if not candidate_library.index.is_unique:
    raise ValueError("Candidate IDs must be unique.")

if candidate_library["temperature_K"].nunique() != 1:
    raise ValueError(
        "This screening block requires one fixed operating temperature."
    )

screening_temperature = float(
    candidate_library["temperature_K"].iloc[0]
)

FIGURES_FOLDER = os.path.join(PROJECT_FOLDER, "Figures")
RESULTS_FOLDER = os.path.join(PROJECT_FOLDER, "Results")

os.makedirs(FIGURES_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

SCREENING_RUN_ID = datetime.now().strftime(
    "%Y%m%d_%H%M%S_%f"
)

SCREENING_RESULTS_FOLDER = os.path.join(
    RESULTS_FOLDER,
    f"block16_{SCREENING_RUN_ID}",
)

os.makedirs(SCREENING_RESULTS_FOLDER, exist_ok=False)


# ============================================================
# 15.2 PREDICT IN BATCHES
# ============================================================

def screen_in_batches(features, feature_set):

    batches = []

    for start in range(0, len(features), PREDICTION_BATCH_SIZE):

        stop = min(start + PREDICTION_BATCH_SIZE, len(features))

        print(
            f"Set {feature_set}: candidates {start + 1}–{stop} "
            f"of {len(features)}",
            flush=True,
        )

        batch_result = predict_with_reliability(
            features.iloc[start:stop],
            analysis=SCREENING_ANALYSIS,
            feature_set=feature_set,
        )

        batches.append(batch_result)

    result = pd.concat(batches)

    if not result.index.equals(features.index):
        raise ValueError("Prediction output lost candidate alignment.")

    return result


screening_outputs = {}

# Set A has identical inputs across processing schedules.
# Predict it once per candidate composition, then map back.
first_composition_row = ~candidate_library[
    "candidate_composition_id"
].duplicated()

unique_A_features = candidate_feature_sets["A"].loc[
    first_composition_row
]

unique_A_results = screen_in_batches(
    unique_A_features,
    feature_set="A",
)

unique_A_results = unique_A_results.copy()
unique_A_results.index = candidate_library.loc[
    unique_A_results.index,
    "candidate_composition_id",
].to_numpy()

mapped_A_results = unique_A_results.reindex(
    candidate_library["candidate_composition_id"].to_numpy()
).copy()

mapped_A_results.index = candidate_library.index

screening_outputs["A"] = mapped_A_results

# Set B includes each composition–schedule combination.
screening_outputs["B"] = screen_in_batches(
    candidate_feature_sets["B"],
    feature_set="B",
)


# ============================================================
# 15.3 ASSEMBLE PREDICTIONS AND EXTRAPOLATION FLAGS
# ============================================================

candidate_screening = candidate_library.copy()

for feature_set, results in screening_outputs.items():

    candidate_screening = candidate_screening.join(
        results.add_prefix(f"{feature_set}__"),
        validate="one_to_one",
    )

    prediction_column = (
        f"{feature_set}__Predicted_log10_sigma"
    )

    if not np.isfinite(
        candidate_screening[prediction_column].to_numpy()
    ).all():
        raise ValueError(
            f"Set {feature_set}: non-finite conductivity predictions."
        )

    # Transform the point prediction only.
    # This is NOT an estimate of mean conductivity in linear space.
    with np.errstate(over="ignore", under="ignore"):
        linear_prediction = np.power(
            10.0,
            candidate_screening[prediction_column].to_numpy(),
        )

    valid_conversion = (
        np.isfinite(linear_prediction)
        & (linear_prediction > 0)
    )

    candidate_screening[
        f"{feature_set}__Predicted_sigma_S_cm"
    ] = np.where(valid_conversion, linear_prediction, np.nan)

    candidate_screening[
        f"{feature_set}__Linear_conversion_valid"
    ] = valid_conversion

    candidate_screening[
        f"{feature_set}__Passes_heuristic_checks"
    ] = (
        ~candidate_screening[
            f"{feature_set}__Beyond_distance_reference"
        ].astype(bool)
        & candidate_screening[
            f"{feature_set}__Features_outside_development_range"
        ].eq(0)
        & valid_conversion
    )

# Difference between two distinct trained pipelines.
# Do not interpret it as the causal effect of processing.
candidate_screening[
    "B_minus_A_predicted_log10_sigma"
] = (
    candidate_screening["B__Predicted_log10_sigma"]
    - candidate_screening["A__Predicted_log10_sigma"]
)

candidate_screening["Screening_status"] = np.where(
    candidate_screening["B__Passes_heuristic_checks"],
    "Passes B heuristic checks; feasibility unverified",
    "Flagged by B applicability/range checks",
)

candidate_screening.to_csv(
    os.path.join(
        SCREENING_RESULTS_FOLDER,
        "all_candidate_predictions.csv",
    ),
    index=False,
)


# ============================================================
# 15.4 RANK ELIGIBLE COMPOSITION–SCHEDULE COMBINATIONS
# ============================================================

eligible_candidates = candidate_screening.loc[
    candidate_screening["B__Passes_heuristic_checks"]
].copy()

# Rank by the original primary B model's point prediction.
# Use lower bootstrap spread only as a tie-breaker.
# No uncertainty penalty is tuned on test data.
ranking_columns = [
    "B__Predicted_log10_sigma",
    "B__Bootstrap_SD_log10_sigma",
    "B__Nearest_development_distance",
    "candidate_composition_id",
    "schedule_id",
]

ranked_candidates = (
    eligible_candidates
    .sort_values(
        ranking_columns,
        ascending=[False, True, True, True, True],
        kind="stable",
    )
    .copy()
)

ranked_candidates.insert(
    1,
    "Screening_rank",
    np.arange(1, len(ranked_candidates) + 1),
)

ranked_candidates.to_csv(
    os.path.join(
        SCREENING_RESULTS_FOLDER,
        "ranked_eligible_composition_schedule_pairs.csv",
    ),
    index=False,
)


# ============================================================
# 15.5 ONE BEST ELIGIBLE SCHEDULE PER COMPOSITION
# ============================================================

best_schedule_per_composition = (
    ranked_candidates
    .drop_duplicates("candidate_composition_id", keep="first")
    .copy()
)

# Rank among distinct compositions, not repeated schedules.
best_schedule_per_composition["Composition_rank"] = np.arange(
    1,
    len(best_schedule_per_composition) + 1,
)

candidate_shortlist = best_schedule_per_composition.head(
    SHORTLIST_SIZE
).copy()

# Separate dataset-absent candidates from known references.
# "Absent from dataset" does not mean novel in the literature.
new_composition_shortlist = (
    best_schedule_per_composition.loc[
        ~best_schedule_per_composition[
            "known_composition_in_dataset"
        ].astype(bool)
    ]
    .head(SHORTLIST_SIZE)
    .copy()
)

best_schedule_per_composition.to_csv(
    os.path.join(
        SCREENING_RESULTS_FOLDER,
        "best_eligible_schedule_per_composition.csv",
    ),
    index=False,
)

candidate_shortlist.to_csv(
    os.path.join(
        SCREENING_RESULTS_FOLDER,
        "candidate_shortlist.csv",
    ),
    index=False,
)

new_composition_shortlist.to_csv(
    os.path.join(
        SCREENING_RESULTS_FOLDER,
        "dataset_absent_composition_shortlist.csv",
    ),
    index=False,
)


# ============================================================
# 15.6 FIGURE-SAVING HELPER
# ============================================================

def save_screening_figure(fig, filename):
    for extension in ("png", "pdf"):
        fig.savefig(
            os.path.join(
                FIGURES_FOLDER,
                f"Block15_{SCREENING_RUN_ID}_{filename}.{extension}",
            ),
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )


# ============================================================
# 15.7 FIGURE — PREDICTION VERSUS BOOTSTRAP VARIABILITY
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 6),
    layout="constrained",
)

for passed, color, marker, label in [
    (False, "#999999", "x", "Flagged by heuristic checks"),
    (True, "#0072B2", "o", "Passes heuristic checks"),
]:

    subset = candidate_screening.loc[
        candidate_screening[
            "B__Passes_heuristic_checks"
        ].eq(passed)
    ]

    if subset.empty:
        continue

    ax.scatter(
        subset["B__Predicted_log10_sigma"],
        subset["B__Bootstrap_SD_log10_sigma"],
        s=25,
        alpha=0.6,
        color=color,
        marker=marker,
        label=label,
    )

ax.set_xlabel("Set B predicted log10(σ / [S cm⁻¹])")
ax.set_ylabel("Bootstrap prediction SD (log10 units)")
ax.set_title(
    f"Candidate screening at {screening_temperature:g} K\n"
    "Bootstrap spread is not a calibrated prediction interval"
)

ax.legend()
ax.grid(alpha=0.2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

save_screening_figure(fig, "prediction_vs_bootstrap_spread")

plt.show()
plt.close(fig)


# ============================================================
# 15.8 FIGURE — DISTINCT-COMPOSITION SHORTLIST
# ============================================================

if not candidate_shortlist.empty:

    plot_table = candidate_shortlist.iloc[::-1].copy()
    positions = np.arange(len(plot_table))

    labels = [
        (
            f"{row.formula} | "
            f"{row.sintering_temperature_K:g} K, "
            f"{row.sintering_time_h:g} h"
        )
        for row in plot_table.itertuples(index=False)
    ]

    fig, ax = plt.subplots(
        figsize=(12, max(5, 0.4 * len(plot_table) + 2)),
        layout="constrained",
    )

    # Show ensemble quantiles separately from original-model points.
    # The original prediction need not lie within these quantiles.
    ax.hlines(
        positions,
        plot_table["B__Bootstrap_prediction_q05"],
        plot_table["B__Bootstrap_prediction_q95"],
        color="#777777",
        linewidth=2,
        label="Bootstrap 5th–95th percentiles (not a prediction interval)",
    )

    ax.scatter(
        plot_table["B__Predicted_log10_sigma"],
        positions,
        color="#0072B2",
        s=40,
        zorder=3,
        label="Original Set B model prediction",
    )

    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=9)

    ax.set_xlabel("Predicted log10(σ / [S cm⁻¹])")
    ax.set_title(
        f"Candidate shortlist at {screening_temperature:g} K\n"
        "Best eligible schedule per composition"
    )

    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_screening_figure(fig, "composition_shortlist")

    plt.show()
    plt.close(fig)

else:
    print(
        "\nNo candidates pass the Set B heuristic checks. "
        "No shortlist was created. Do not relax the checks merely "
        "to obtain high predictions; inspect the flagged features."
    )


# ============================================================
# 15.9 FIGURE — COMPOSITION MAP
# ============================================================

if not best_schedule_per_composition.empty:

    plot_table = best_schedule_per_composition

    fig, ax = plt.subplots(
        figsize=(8, 6),
        layout="constrained",
    )

    scatter = ax.scatter(
        100 * plot_table["Y_cation_fraction"],
        100 * plot_table["Sc_cation_fraction"],
        c=plot_table["B__Predicted_log10_sigma"],
        cmap="viridis",
        s=120,
        edgecolors="black",
        linewidths=0.5,
    )

    fig.colorbar(
        scatter,
        ax=ax,
        label="Set B predicted log10 conductivity",
    )

    ax.set_xlabel("Y fraction of total cations (%)")
    ax.set_ylabel("Sc fraction of total cations (%)")
    ax.set_title(
        f"Eligible candidates at {screening_temperature:g} K\n"
        "Each point uses its best eligible processing schedule"
    )

    ax.grid(alpha=0.2)
    ax.set_axisbelow(True)

    save_screening_figure(fig, "composition_prediction_map")

    plt.show()
    plt.close(fig)


# ============================================================
# 15.10 SAVE SCREENING SETTINGS AND COVERAGE
# ============================================================

screening_summary = pd.DataFrame([
    {
        "Total_candidate_pairs": len(candidate_screening),
        "Total_compositions": (
            candidate_screening["candidate_composition_id"].nunique()
        ),
        "Pairs_passing_B_checks": len(eligible_candidates),
        "Compositions_with_eligible_schedule": (
            len(best_schedule_per_composition)
        ),
        "Shortlisted_compositions": len(candidate_shortlist),
        "Dataset_absent_shortlisted_compositions": (
            len(new_composition_shortlist)
        ),
        "Operating_temperature_K": screening_temperature,
    }
])

screening_summary.to_csv(
    os.path.join(
        SCREENING_RESULTS_FOLDER,
        "screening_summary.csv",
    ),
    index=False,
)

with open(
    os.path.join(
        SCREENING_RESULTS_FOLDER,
        "screening_settings.json",
    ),
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        {
            "candidate_run_id": CANDIDATE_RUN_ID,
            "reliability_run_id": RELIABILITY_RUN_ID,
            "analysis": SCREENING_ANALYSIS,
            "A_model": selected_model_keys[
                (SCREENING_ANALYSIS, "A")
            ][2],
            "B_model": selected_model_keys[
                (SCREENING_ANALYSIS, "B")
            ][2],
            "operating_temperature_K": screening_temperature,
            "main_ranking": (
                "Set B point prediction descending; bootstrap SD "
                "and descriptor distance used as tie-breakers"
            ),
            "eligibility": (
                "Passes Set B distance threshold, no raw input "
                "outside development min/max, finite linear conversion"
            ),
            "shortlist_size_max": SHORTLIST_SIZE,
            "one_schedule_per_shortlisted_composition": True,
            "bootstrap_intervals_calibrated": False,
            "phase_stability_verified": False,
            "synthesis_feasibility_verified": False,
            "test_data_used": False,
            "models_retrained": False,
        },
        file,
        indent=2,
    )

print("\nSCREENING SUMMARY")
print(screening_summary.to_string(index=False))

print("\nCANDIDATE SHORTLIST")

display_columns = [
    "Composition_rank",
    "formula",
    "sintering_temperature_K",
    "sintering_time_h",
    "B__Predicted_log10_sigma",
    "B__Predicted_sigma_S_cm",
    "B__Bootstrap_SD_log10_sigma",
    "known_composition_in_dataset",
]

print(
    candidate_shortlist[display_columns]
    .round(4)
    .to_string(index=False)
)

print("\nBLOCK 15 COMPLETE")
print("Results:", SCREENING_RESULTS_FOLDER)
print("Figures:", FIGURES_FOLDER)
print("Candidates are computational proposals, not validated materials.")
print("No activation-energy predictions or Set C screening performed.")









# %% BLOCK 16 — MANUSCRIPT TABLES AND REPRODUCIBILITY PACKAGE

#
# No model fitting, model selection or test reevaluation.
#
# Creates:
#   Results/manuscript_<timestamp>/
#       Main and supplementary tables
#       Method/status report
#       Software versions
#       Model archives
#       Source/split snapshots
#       Figure inventory
#       File checksums
#
# Existing figures stay in the project's Figures folder.


import sys
import subprocess

subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "cloudpickle",
])
# Cloudpickle handles the custom selector defined in VS Code.
# This is the cloudpickle implementation bundled with joblib.
import cloudpickle


# ============================================================
# 16.1 CHECK COMPLETED CORE OUTPUTS
# ============================================================

required_objects = [
    "final_test_metrics",
    "final_test_predictions",
    "same_algorithm_feature_effects",
    "selected_models_table",
    "validation_metrics",
    "primary_final_models",
    "selected_model_keys",
    "split_summary",
    "cv_summary",
    "SPLIT_MANIFEST_PATH",
]

missing_objects = [
    name for name in required_objects
    if name not in globals()
]

if missing_objects:
    raise RuntimeError(
        f"Missing required outputs: {missing_objects}. "
        "Complete the corresponding earlier blocks first."
    )

FIGURES_FOLDER = os.path.join(PROJECT_FOLDER, "Figures")
RESULTS_FOLDER = os.path.join(PROJECT_FOLDER, "Results")

PACKAGE_RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

MANUSCRIPT_RESULTS_FOLDER = os.path.join(
    RESULTS_FOLDER,
    f"manuscript_{PACKAGE_RUN_ID}",
)

os.makedirs(MANUSCRIPT_RESULTS_FOLDER, exist_ok=False)

SAVE_BOOTSTRAP_MODELS = True


# ============================================================
# 16.2 TABLE-SAVING HELPER
# ============================================================

table_inventory_rows = []


def export_table(frame, filename, description):
    destination = os.path.join(
        MANUSCRIPT_RESULTS_FOLDER,
        filename,
    )

    frame.to_csv(destination, index=False)

    table_inventory_rows.append({
        "Filename": filename,
        "Description": description,
        "Rows": len(frame),
        "Columns": len(frame.columns),
    })

    return destination


# ============================================================
# 16.3 MAIN MANUSCRIPT TABLES
# ============================================================

export_table(
    split_summary,
    "Table_1_dataset_partitions.csv",
    "Measurement and composition counts for each partition.",
)

primary_results = final_test_metrics.loc[
    final_test_metrics["Primary_validation_selected"]
].copy()

main_metric_columns = [
    "Analysis",
    "Feature_set",
    "Model",
    "Development_rows",
    "Test_rows",
    "Test_groups",
    "RMSE",
    "MAE",
    "R2",
    "Composition_balanced_MAE",
    "Composition_balanced_RMSE",
]

export_table(
    primary_results[main_metric_columns],
    "Table_2_primary_test_results.csv",
    "Test performance of models selected using validation RMSE.",
)

export_table(
    same_algorithm_feature_effects,
    "Table_3_same_algorithm_feature_comparisons.csv",
    "Paired feature-set differences within each algorithm family.",
)

if "candidate_shortlist" in globals():

    shortlist_columns = [
        "candidate_id",
        "Composition_rank",
        "formula",
        "sintering_temperature_K",
        "sintering_time_h",
        "temperature_K",
        "B__Predicted_log10_sigma",
        "B__Predicted_sigma_S_cm",
        "B__Bootstrap_SD_log10_sigma",
        "B__Nearest_development_distance",
        "known_composition_in_dataset",
        "phase_stability_verified",
        "synthesis_feasibility_verified",
    ]

    available_columns = [
        column for column in shortlist_columns
        if column in candidate_shortlist.columns
    ]

    export_table(
        candidate_shortlist[available_columns],
        "Table_4_computational_candidate_shortlist.csv",
        "Proposed candidates; no independent validation implied.",
    )


# ============================================================
# 16.4 SUPPLEMENTARY TABLES
# ============================================================

export_table(
    final_test_metrics,
    "Supplement_all_model_test_metrics.csv",
    "All frozen-model test results; primary models explicitly marked.",
)

export_table(
    final_test_predictions,
    "Supplement_test_predictions.csv",
    "Source-linked test predictions and residuals.",
)

export_table(
    validation_metrics,
    "Supplement_validation_metrics.csv",
    "Validation results used for model selection.",
)

export_table(
    selected_models_table,
    "Supplement_validation_selected_models.csv",
    "Model choices made before final test evaluation.",
)

export_table(
    cv_summary,
    "Supplement_training_CV_partitions.csv",
    "Training-only grouped cross-validation fold counts.",
)

optional_tables = [
    (
        "tuning_summary",
        "Supplement_tuning_summary.csv",
        "Hyperparameter-tuning scores; not independent final-test scores.",
    ),
    (
        "final_test_metrics_by_composition",
        "Supplement_test_metrics_by_composition.csv",
        "Errors for each held-out composition.",
    ),
    (
        "classification_metrics",
        "Supplement_threshold_classification_metrics.csv",
        "Threshold-based assessment of the conductivity regressors.",
    ),
    (
        "classification_reports",
        "Supplement_classification_reports.csv",
        "Class-specific precision, recall and F1.",
    ),
    (
        "final_shap_importance_table",
        "Supplement_final_predictor_SHAP_importance.csv",
        "SHAP importance for validation-selected final predictors.",
    ),
    (
        "physics_retention_table",
        "Supplement_physics_feature_retention.csv",
        "Physics features retained or excluded by final selection.",
    ),
    (
        "shap_diagnostics_table",
        "Supplement_SHAP_diagnostics.csv",
        "Explanation methods, sample sizes and additivity checks.",
    ),
    (
        "applicability_summary",
        "Supplement_applicability_reference.csv",
        "Heuristic descriptor-distance references.",
    ),
    (
        "bootstrap_membership",
        "Supplement_bootstrap_membership.csv",
        "Composition-group resampling records.",
    ),
    (
        "candidate_screening",
        "Supplement_all_candidate_predictions.csv",
        "Candidate predictions, variability and applicability flags.",
    ),
    (
        "screening_summary",
        "Supplement_screening_coverage.csv",
        "Candidate counts and eligibility coverage.",
    ),
]

for object_name, filename, description in optional_tables:

    frame = globals().get(object_name)

    if isinstance(frame, pd.DataFrame):
        export_table(frame, filename, description)


# ============================================================
# 16.5 SAVE SOFTWARE VERSIONS
# ============================================================

software_versions = {
    "python": sys.version,
}

for package_name in [
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "matminer",
    "pymatgen",
    "shap",
    "xgboost",
    "matplotlib",
    "joblib",
    "openpyxl",
]:

    try:
        software_versions[package_name] = version(package_name)
    except PackageNotFoundError:
        software_versions[package_name] = "Distribution not found"

with open(
    os.path.join(
        MANUSCRIPT_RESULTS_FOLDER,
        "software_versions.json",
    ),
    "w",
    encoding="utf-8",
) as file:
    json.dump(software_versions, file, indent=2)


# ============================================================
# 16.6 SNAPSHOT CODE, SPLITS AND RUN SETTINGS
# ============================================================

snapshot_folder = os.path.join(
    MANUSCRIPT_RESULTS_FOLDER,
    "reproducibility",
)

os.makedirs(snapshot_folder, exist_ok=True)

snapshot_sources = {
    "master_split_manifest.csv": SPLIT_MANIFEST_PATH,
    "sourcecode_snapshot.py": os.path.join(
        PROJECT_FOLDER,
        "src",
        "sourcecode.py",
    ),
}

# Save sourcecode.py in VS Code before running this block.
# The snapshot copies the file on disk, not unsaved editor content.

run_folder_variables = [
    "TUNING_RESULTS_FOLDER",
    "VALIDATION_RESULTS_FOLDER",
    "FINAL_RESULTS_FOLDER",
    "CLASSIFICATION_RESULTS_FOLDER",
    "SHAP_RESULTS_FOLDER",
    "RELIABILITY_RESULTS_FOLDER",
    "CANDIDATE_RESULTS_FOLDER",
    "SCREENING_RESULTS_FOLDER",
]

snapshot_log = []

for destination_name, source_path in snapshot_sources.items():

    exists = os.path.isfile(source_path)

    if exists:
        shutil.copy2(
            source_path,
            os.path.join(snapshot_folder, destination_name),
        )

    snapshot_log.append({
        "Source": source_path,
        "Copied": exists,
        "Destination": destination_name if exists else "",
    })

# Copy small JSON method/settings records from each run.
for variable_name in run_folder_variables:

    source_folder = globals().get(variable_name)

    if not source_folder or not os.path.isdir(source_folder):
        continue

    for filename in sorted(os.listdir(source_folder)):

        if not filename.lower().endswith(".json"):
            continue

        destination_name = f"{variable_name}__{filename}"

        shutil.copy2(
            os.path.join(source_folder, filename),
            os.path.join(snapshot_folder, destination_name),
        )

        snapshot_log.append({
            "Source": os.path.join(source_folder, filename),
            "Copied": True,
            "Destination": destination_name,
        })

pd.DataFrame(snapshot_log).to_csv(
    os.path.join(snapshot_folder, "snapshot_inventory.csv"),
    index=False,
)


# ============================================================
# 16.7 SAVE FITTED PRIMARY MODELS
# ============================================================

primary_model_bundle = {
    "primary_models": primary_final_models,
    "selected_model_keys": selected_model_keys,
    "input_columns": {
        pair: analysis_features[pair[0]][pair[1]].columns.tolist()
        for pair in primary_final_models
    },
    "target": "log10_sigma_S_cm",
    "software_versions": software_versions,
}

PRIMARY_MODEL_ARCHIVE = os.path.join(
    MANUSCRIPT_RESULTS_FOLDER,
    "primary_models.pkl.gz",
)

with gzip.open(PRIMARY_MODEL_ARCHIVE, "wb") as file:
    cloudpickle.dump(primary_model_bundle, file)

print("Saved primary fitted models.", flush=True)

# These models include the fitted preprocessing and SHAP selectors.


# ============================================================
# 16.8 SAVE CANDIDATE-SCREENING ENSEMBLES
# ============================================================

SCREENING_MODEL_ARCHIVE = None

if (
    SAVE_BOOTSTRAP_MODELS
    and "bootstrap_models" in globals()
    and "applicability_models" in globals()
):

    screening_bundle = {
        "primary_models": primary_final_models,
        "bootstrap_models": bootstrap_models,
        "applicability_models": applicability_models,
        "software_versions": software_versions,
        "interpretation": (
            "Bootstrap prediction variability is not a calibrated "
            "prediction interval."
        ),
    }

    SCREENING_MODEL_ARCHIVE = os.path.join(
        MANUSCRIPT_RESULTS_FOLDER,
        "screening_ensembles.pkl.gz",
    )

    print(
        "Saving bootstrap ensembles; this may take some time.",
        flush=True,
    )

    with gzip.open(SCREENING_MODEL_ARCHIVE, "wb") as file:
        cloudpickle.dump(screening_bundle, file)

    print("Saved screening ensembles.", flush=True)


# ============================================================
# 16.9 FIGURE INVENTORY
# ============================================================

# Index project figures without moving or altering them.
# The inventory may include earlier runs; use run IDs to identify
# the intended manuscript versions.

figure_inventory_rows = []

for filename in sorted(os.listdir(FIGURES_FOLDER)):

    if not filename.lower().endswith((".png", ".pdf")):
        continue

    figure_inventory_rows.append({
        "Filename": filename,
        "Absolute_path": os.path.join(FIGURES_FOLDER, filename),
        "Format": os.path.splitext(filename)[1].lstrip("."),
        "Selected_for_manuscript": "",
        "Proposed_figure_number": "",
        "Caption_notes": "",
    })

figure_inventory = pd.DataFrame(figure_inventory_rows)

figure_inventory.to_csv(
    os.path.join(
        MANUSCRIPT_RESULTS_FOLDER,
        "figure_inventory.csv",
    ),
    index=False,
)


# ============================================================
# 16.10 RECORD COMPLETED AND UNFINISHED WORK
# ============================================================

project_status = pd.DataFrame([
    {
        "Component": "Conductivity model evaluation",
        "Status": "Completed",
        "Notes": (
            "Grouped training CV, validation selection and final "
            "composition-held-out test benchmark."
        ),
    },
    {
        "Component": "Processing and physics feature comparisons",
        "Status": "Completed",
        "Notes": (
            "Report paired common-subset results, including "
            "non-improvements."
        ),
    },
    {
        "Component": "Threshold classification assessment",
        "Status": (
            "Outputs available"
            if "classification_metrics" in globals()
            else "Not completed in this session"
        ),
        "Notes": (
            "Supplementary assessment of regression predictions; "
            "state conductivity cutoff and temperature scope."
        ),
    },
    {
        "Component": "Final-predictor SHAP",
        "Status": (
            "Outputs available"
            if "final_shap_importance_table" in globals()
            else "Not completed in this session"
        ),
        "Notes": (
            "State each explainer's dependence assumptions and "
            "the development-data sampling method."
        ),
    },
    {
        "Component": "Bootstrap and applicability",
        "Status": (
            "Outputs available"
            if "applicability_models" in globals()
            else "Not completed in this session"
        ),
        "Notes": (
            "Model variability and heuristic distances; "
            "not calibrated prediction intervals."
        ),
    },
    {
        "Component": "Candidate screening",
        "Status": (
            "Outputs available"
            if "candidate_screening" in globals()
            else "Not completed in this session"
        ),
        "Notes": (
            "Computational proposals only. Phase stability and "
            "synthesis feasibility are not independently verified."
        ),
    },
    {
        "Component": "All-feature versus top-30 comparison",
        "Status": "Not performed in these blocks",
        "Notes": (
            "Current results concern the specified SHAP-selection "
            "workflow; they do not establish that top 30 is optimal."
        ),
    },
])

export_table(
    project_status,
    "project_completion_status.csv",
    "Explicit scope, completed outputs and remaining work.",
)


# ============================================================
# 16.11 WRITE A METHODS AND REPORTING NOTE
# ============================================================

report_lines = [
    "# Conductivity study: methods and reporting notes",
    "",
    "## Target and feature sets",
    "",
    "Target: log10 ionic conductivity, with conductivity in S/cm.",
    "Set A: chemical descriptors plus operating temperature.",
    "Set B: Set A plus sintering temperature and time.",
    "Set C: Set B plus available physics descriptors.",
    "Activation energy, grain size and relative density are not "
    "conductivity predictors in this workflow.",
    "",
    "## Evaluation",
    "",
    "The master split groups equivalent cation compositions.",
    "Full-data A/B and common-subset A/B/C are distinct analyses.",
    "Imputation, constant-feature removal, scaling and SHAP selection "
    "are fitted within training CV folds.",
    "Hyperparameters are selected by mean grouped-CV RMSE.",
    "Primary models are selected by validation RMSE.",
    "Frozen pipelines are refitted on training plus validation.",
    "Test results are reported for all benchmark models, with "
    "validation-selected primary models identified.",
    "CV tuning scores, validation scores and test scores must be "
    "labelled separately.",
    "",
    "## Interpretation",
    "",
    "SHAP describes model associations, not causal mechanisms.",
    "Report the explainer method recorded for each predictor.",
    "Bootstrap quantiles describe model variability and are not "
    "calibrated prediction intervals.",
    "Distance and feature-range checks are heuristic applicability "
    "indicators, not feasibility tests.",
    "",
    "## Candidate screening",
    "",
    "Report the assumed valences, cation-fraction grid, fixed operating "
    "temperature, processing schedules and applicability rules.",
    "A composition absent from the dataset is not necessarily "
    "novel in the literature.",
    "Candidate recommendations are not independently validated "
    "conductivity improvements.",
    "",
    "## Scope",
    "",
    "Bayesian optimization is excluded.",
    "Activation-energy modelling remains pending source-supported "
    "sample/regime records.",
    "Paper-held-out performance and unseen-family transfer were "
    "not evaluated in these blocks.",
    "",
    "## Reproducibility",
    "",
    "The model archives contain fitted primary pipelines and, when "
    "enabled, bootstrap screening ensembles.",
    "Load pickle archives only when they come from a trusted source.",
    "Use the recorded software versions; cross-version model loading "
    "is not guaranteed.",
    "The code snapshot is the saved src/sourcecode.py file.",
    "Raw data are not copied by this block. Preserve the exact input "
    "workbooks and their source provenance alongside the study.",
]

with open(
    os.path.join(
        MANUSCRIPT_RESULTS_FOLDER,
        "methods_and_reporting_notes.md",
    ),
    "w",
    encoding="utf-8",
) as file:
    file.write("\n".join(report_lines))


# ============================================================
# 16.12 SAVE TABLE INVENTORY AND FILE CHECKSUMS
# ============================================================

pd.DataFrame(table_inventory_rows).to_csv(
    os.path.join(
        MANUSCRIPT_RESULTS_FOLDER,
        "table_inventory.csv",
    ),
    index=False,
)

checksum_rows = []

for root, _, filenames in os.walk(MANUSCRIPT_RESULTS_FOLDER):

    for filename in sorted(filenames):

        if filename == "file_checksums.csv":
            continue

        path = os.path.join(root, filename)
        digest = hashlib.sha256()

        with open(path, "rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)

        checksum_rows.append({
            "Relative_path": os.path.relpath(
                path,
                MANUSCRIPT_RESULTS_FOLDER,
            ),
            "Size_bytes": os.path.getsize(path),
            "SHA256": digest.hexdigest(),
        })

pd.DataFrame(checksum_rows).to_csv(
    os.path.join(
        MANUSCRIPT_RESULTS_FOLDER,
        "file_checksums.csv",
    ),
    index=False,
)


# ============================================================
# 16.13 SUMMARY
# ============================================================

print("\nPROJECT STATUS")
print(project_status.to_string(index=False))

print("\nMANUSCRIPT PACKAGE SAVED")
print(MANUSCRIPT_RESULTS_FOLDER)

print("\nPRIMARY MODEL ARCHIVE")
print(PRIMARY_MODEL_ARCHIVE)

if SCREENING_MODEL_ARCHIVE is not None:
    print("\nSCREENING MODEL ARCHIVE")
    print(SCREENING_MODEL_ARCHIVE)

print("\nFigures remain in:", FIGURES_FOLDER)
print("No models were trained or test results recalculated.")




# %% BLOCK 17 — CURATED MANUSCRIPT FIGURES AND TABLES

#
# Uses existing results only.
# No fitting, model selection, new predictions or test reevaluation.
#
# Main figures:
#   Figure 1: retain the existing workflow diagram
#   Figure 2: dataset coverage
#   Figure 3: all-model benchmark
#   Figure 4: primary-model test parity panels
#   Figure 5a/b: selected final-predictor SHAP beeswarms
#   Figure 6: candidate heatmap at fixed temperature
#
# Figures: Figures/manuscript_<run>/
# Tables:  Results/manuscript_visuals_<run>/




# ============================================================
# 17.1 SETTINGS AND REQUIRED OUTPUTS
# ============================================================

required_objects = [
    "data",
    "analysis_features",
    "split_summary",
    "final_test_metrics",
    "final_test_predictions",
    "same_algorithm_feature_effects",
    "selected_model_keys",
    "primary_final_models",
]

missing_objects = [
    name for name in required_objects
    if name not in globals()
]

if missing_objects:
    raise RuntimeError(
        f"Missing objects: {missing_objects}. "
        "Restore or complete the corresponding blocks first."
    )

VISUAL_RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

MANUSCRIPT_FIGURES = os.path.join(
    PROJECT_FOLDER, "Figures", f"manuscript_{VISUAL_RUN_ID}"
)

MANUSCRIPT_TABLES = os.path.join(
    PROJECT_FOLDER, "Results", f"manuscript_visuals_{VISUAL_RUN_ID}"
)

os.makedirs(MANUSCRIPT_FIGURES, exist_ok=False)
os.makedirs(MANUSCRIPT_TABLES, exist_ok=False)

# Keep the heatmap readable. Select schedules by their development
# data frequency, not by their predicted conductivity.
MAX_HEATMAP_SCHEDULES = 12

plot_style = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
}

feature_colors = {
    "A": "#0072B2",
    "B": "#009E73",
    "C": "#D55E00",
}

captions = []


def save_manuscript_figure(fig, name, caption):
    for extension in ("png", "pdf"):
        fig.savefig(
            os.path.join(
                MANUSCRIPT_FIGURES,
                f"{name}.{extension}",
            ),
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
        )

    captions.append({
        "Figure": name,
        "Caption_draft": caption,
    })


def save_manuscript_table(frame, filename):
    # Preserve underlying numerical precision for later formatting.
    frame.to_csv(
        os.path.join(MANUSCRIPT_TABLES, filename),
        index=False,
    )


def panel_label(ax, label):
    ax.text(
        -0.12, 1.06, label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
    )


# ============================================================
# 17.2 TABLE 1 — DATASET PARTITIONS
# ============================================================

save_manuscript_table(
    split_summary,
    "Table_1_dataset_partitions.csv",
)


# ============================================================
# 17.3 TABLE 2 — PRIMARY MODELS AND SETTINGS
# ============================================================

primary_metrics = final_test_metrics.loc[
    final_test_metrics["Primary_validation_selected"]
].copy()

# Add RMSE / target standard deviation, as in your example.
# This is descriptive and uses ddof=0 explicitly.
# For a fixed test set it is redundant with R²:
# normalized RMSE = sqrt(1 - R²).
normalized_error_rows = []

for key, group in final_test_predictions.groupby(
    ["Analysis", "Feature_set", "Model"]
):
    observed = group["Measured_log10_sigma"].to_numpy()
    predicted = group["Predicted_log10_sigma"].to_numpy()

    target_sd = np.std(observed, ddof=0)
    rmse = np.sqrt(np.mean((predicted - observed) ** 2))

    normalized_error_rows.append({
        "Analysis": key[0],
        "Feature_set": key[1],
        "Model": key[2],
        "Test_target_SD_ddof0": target_sd,
        "RMSE_over_target_SD": (
            rmse / target_sd if target_sd > 0 else np.nan
        ),
    })

normalized_error_table = pd.DataFrame(normalized_error_rows)

primary_table = primary_metrics.merge(
    normalized_error_table,
    on=["Analysis", "Feature_set", "Model"],
    validate="one_to_one",
)

# Summarize settings selected by training CV when available.
if "tuning_summary" in globals():
    primary_table = primary_table.merge(
        tuning_summary[
            ["Analysis", "Feature_set", "Model", "Best_parameters"]
        ],
        on=["Analysis", "Feature_set", "Model"],
        how="left",
        validate="one_to_one",
    )

save_manuscript_table(
    primary_table,
    "Table_2_primary_models_and_test_performance.csv",
)

save_manuscript_table(
    final_test_metrics.merge(
        normalized_error_table,
        on=["Analysis", "Feature_set", "Model"],
        validate="one_to_one",
    ),
    "Table_S1_all_model_test_performance.csv",
)


# ============================================================
# 17.4 TABLES 3 AND 4
# ============================================================

save_manuscript_table(
    same_algorithm_feature_effects,
    "Table_3_paired_feature_set_comparisons.csv",
)

if "candidate_shortlist" in globals():

    desired_columns = [
        "candidate_id",
        "Composition_rank",
        "formula",
        "temperature_K",
        "sintering_temperature_K",
        "sintering_time_h",
        "B__Predicted_log10_sigma",
        "B__Predicted_sigma_S_cm",
        "B__Bootstrap_SD_log10_sigma",
        "B__Bootstrap_prediction_q05",
        "B__Bootstrap_prediction_q95",
        "known_composition_in_dataset",
        "phase_stability_verified",
        "synthesis_feasibility_verified",
    ]

    save_manuscript_table(
        candidate_shortlist[
            [
                column for column in desired_columns
                if column in candidate_shortlist.columns
            ]
        ],
        "Table_4_candidate_shortlist.csv",
    )


# ============================================================
# 17.5 FIGURE 2 — DATASET COVERAGE
# ============================================================

comparison_index = analysis_features["comparison"]["C"].index

dataset_coverage = pd.DataFrame([
    {
        "Dataset": "Full",
        "Measurements": len(data),
        "Compositions": data["composition_group"].nunique(),
        "Sources": data["paper_id"].nunique(),
    },
    {
        "Dataset": "Physics-complete",
        "Measurements": len(comparison_index),
        "Compositions": data.loc[
            comparison_index, "composition_group"
        ].nunique(),
        "Sources": data.loc[
            comparison_index, "paper_id"
        ].nunique(),
    },
])

save_manuscript_table(
    dataset_coverage,
    "Table_S2_dataset_coverage.csv",
)

physics_frame = analysis_features["full"]["B"]  # not used as physics

# X_C_all retains all measurement rows, including missing physics.
if "X_C_all" in globals():
    physics_columns_for_plot = [
        column for column in X_C_all.columns
        if column.startswith("phys_")
    ]
    physics_frame = X_C_all[physics_columns_for_plot]

elif "X_physics" in globals():
    physics_frame = X_physics

else:
    physics_frame = None

with plt.rc_context(plot_style):

    fig, axes = plt.subplots(
        2, 2,
        figsize=(12, 9),
        layout="constrained",
    )

    histogram_bins = np.histogram_bin_edges(
        data["log10_sigma_S_cm"],
        bins=25,
    )

    axes[0, 0].hist(
        data["log10_sigma_S_cm"],
        bins=histogram_bins,
        alpha=0.6,
        color="#0072B2",
        label="Full dataset",
    )

    axes[0, 0].hist(
        data.loc[comparison_index, "log10_sigma_S_cm"],
        bins=histogram_bins,
        histtype="step",
        linewidth=2,
        color="#D55E00",
        label="Physics-complete subset",
    )

    axes[0, 0].set_xlabel("Measured log10(σ / [S cm⁻¹])")
    axes[0, 0].set_ylabel("Measurement rows")
    axes[0, 0].legend()

    density = axes[0, 1].hexbin(
        data["temperature_K"],
        data["log10_sigma_S_cm"],
        gridsize=25,
        mincnt=1,
        cmap="viridis",
    )

    axes[0, 1].set_xlabel("Operating temperature (K)")
    axes[0, 1].set_ylabel("Measured log10(σ / [S cm⁻¹])")
    fig.colorbar(
        density,
        ax=axes[0, 1],
        label="Measurement rows per bin",
    )

    positions = np.arange(3)
    width = 0.35

    for offset, row in enumerate(
        dataset_coverage.itertuples(index=False)
    ):
        axes[1, 0].bar(
            positions + (offset - 0.5) * width,
            [row.Measurements, row.Compositions, row.Sources],
            width=width,
            label=row.Dataset,
            color=["#0072B2", "#D55E00"][offset],
        )

    axes[1, 0].set_xticks(positions)
    axes[1, 0].set_xticklabels(
        ["Measurements", "Compositions", "Sources"]
    )
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylabel("Count (log scale)")
    axes[1, 0].legend()

    if physics_frame is not None:

        availability = (
            physics_frame.notna().mean().mul(100).sort_values()
        )

        labels = [
            column.replace("phys_", "").replace("_", " ")
            for column in availability.index
        ]

        axes[1, 1].barh(
            labels,
            availability,
            color="#009E73",
        )

        axes[1, 1].set_xlim(0, 105)
        axes[1, 1].set_xlabel("Available measurement rows (%)")
        axes[1, 1].tick_params(axis="y", labelsize=8)

        save_manuscript_table(
            availability.rename("Available_percent")
            .rename_axis("Feature")
            .reset_index(),
            "Table_S3_physics_descriptor_availability.csv",
        )

    else:
        axes[1, 1].text(
            0.5, 0.5,
            "Full-row physics coverage not available",
            ha="center",
            transform=axes[1, 1].transAxes,
        )

    for label, ax in zip(
        ["(a)", "(b)", "(c)", "(d)"],
        axes.flat,
    ):
        panel_label(ax, label)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    save_manuscript_figure(
        fig,
        "Figure_2_dataset_coverage",
        "Dataset coverage: (a) measured conductivity distributions; "
        "(b) temperature–conductivity measurement density; "
        "(c) measurement, composition and source counts; "
        "(d) physics-descriptor availability. Descriptive summaries "
        "include all partitions and were not used to retune models.",
    )

    plt.show()
    plt.close(fig)


# ============================================================
# 17.6 FIGURE 3 — CONSOLIDATED ALL-MODEL COMPARISON
# ============================================================

model_order = sorted(final_test_metrics["Model"].unique())

with plt.rc_context(plot_style):

    fig, axes = plt.subplots(
        2, 3,
        figsize=(18, 10),
        layout="constrained",
    )

    for row_number, analysis in enumerate(
        ["full", "comparison"]
    ):

        feature_sets = (
            ["A", "B"] if analysis == "full" else ["A", "B", "C"]
        )

        width = 0.8 / len(feature_sets)
        positions = np.arange(len(model_order))

        for column_number, metric in enumerate(
            ["R2", "MAE", "RMSE"]
        ):

            ax = axes[row_number, column_number]

            for offset, feature_set in enumerate(feature_sets):

                table = final_test_metrics.loc[
                    final_test_metrics["Analysis"].eq(analysis)
                    & final_test_metrics["Feature_set"].eq(feature_set)
                ].set_index("Model").reindex(model_order)

                bar_positions = positions + (
                    offset - (len(feature_sets) - 1) / 2
                ) * width

                ax.bar(
                    bar_positions,
                    table[metric],
                    width=width,
                    color=feature_colors[feature_set],
                    label=f"Set {feature_set}",
                )

                # A star identifies validation-selected models.
                for position, value, primary in zip(
                    bar_positions,
                    table[metric],
                    table["Primary_validation_selected"],
                ):
                    if primary and np.isfinite(value):
                        ax.annotate(
                            "*",
                            (position, value),
                            xytext=(0, 4),
                            textcoords="offset points",
                            ha="center",
                            fontsize=13,
                        )

            ax.set_xticks(positions)
            ax.set_xticklabels(
                model_order,
                rotation=55,
                ha="right",
                fontsize=8,
            )

            ax.set_title(
                f"{analysis.capitalize()}: "
                f"{'R²' if metric == 'R2' else metric}"
            )

            ax.set_ylabel(
                "R²" if metric == "R2" else "Log10 conductivity units"
            )

            ax.axhline(0, color="black", linewidth=0.6)
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
            ax.legend()

            panel_label(
                ax,
                f"({chr(97 + row_number * 3 + column_number)})",
            )

    save_manuscript_figure(
        fig,
        "Figure_3_all_model_test_comparison",
        "Final-test model comparison. Top row: full-data A/B; bottom "
        "row: common-subset A/B/C. Stars mark models selected using "
        "validation RMSE. Analyses use different test populations "
        "and must not be interpreted as paired across rows. "
        "No error bars are shown because repeated independent "
        "test evaluations were not performed.",
    )

    plt.show()
    plt.close(fig)


# ============================================================
# 17.7 FIGURE 4 — PRIMARY-MODEL PARITY PANELS
# ============================================================

panel_order = [
    ("full", "A"),
    ("full", "B"),
    ("comparison", "A"),
    ("comparison", "B"),
    ("comparison", "C"),
]

primary_prediction_mask = final_test_predictions[
    "Primary_validation_selected"
]

primary_predictions = final_test_predictions.loc[
    primary_prediction_mask
]

all_values = primary_predictions[
    ["Measured_log10_sigma", "Predicted_log10_sigma"]
].to_numpy()

lower, upper = np.nanmin(all_values), np.nanmax(all_values)
padding = max(0.1, 0.05 * (upper - lower))
limits = [lower - padding, upper + padding]

with plt.rc_context(plot_style):

    fig, axes = plt.subplots(
        2, 3,
        figsize=(15, 10),
        layout="constrained",
    )

    for panel_number, pair in enumerate(panel_order):

        analysis, feature_set = pair
        model_name = selected_model_keys[pair][2]
        ax = axes.flat[panel_number]

        table = primary_predictions.loc[
            primary_predictions["Analysis"].eq(analysis)
            & primary_predictions["Feature_set"].eq(feature_set)
            & primary_predictions["Model"].eq(model_name)
        ]

        metric = primary_metrics.loc[
            primary_metrics["Analysis"].eq(analysis)
            & primary_metrics["Feature_set"].eq(feature_set)
        ].iloc[0]

        ax.scatter(
            table["Measured_log10_sigma"],
            table["Predicted_log10_sigma"],
            s=28,
            alpha=0.7,
            color=feature_colors[feature_set],
            edgecolors="none",
        )

        ax.plot(limits, limits, "--", color="black", linewidth=1)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_aspect("equal", adjustable="box")

        ax.set_title(
            f"{analysis.capitalize()} {feature_set}: {model_name}"
        )
        ax.set_xlabel("Measured log10(σ / [S cm⁻¹])")
        ax.set_ylabel("Predicted log10(σ / [S cm⁻¹])")

        ax.text(
            0.04, 0.96,
            (
                f"R² = {metric['R2']:.3f}\n"
                f"RMSE = {metric['RMSE']:.3f}\n"
                f"MAE = {metric['MAE']:.3f}\n"
                f"{int(metric['Test_rows'])} rows; "
                f"{int(metric['Test_groups'])} compositions"
            ),
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )

        panel_label(ax, f"({chr(97 + panel_number)})")

    axes.flat[5].axis("off")

    save_manuscript_figure(
        fig,
        "Figure_4_primary_test_parity",
        "Measured versus predicted test conductivity for the five "
        "validation-selected pipelines after refitting on development "
        "data. Dashed lines indicate identity. Test counts are shown "
        "per panel. No uncertainty interval is implied.",
    )

    plt.show()
    plt.close(fig)


# ============================================================
# 17.8 FIGURE 5 — MAIN-TEXT FINAL-PREDICTOR SHAP
# ============================================================

# Reuse completed explanations; do not recompute SHAP.
if "final_shap_explanations" in globals():

    for suffix, pair in [
        ("a", ("full", "B")),
        ("b", ("comparison", "C")),
    ]:

        if pair not in final_shap_explanations:
            continue

        model_name = selected_model_keys[pair][2]

        with plt.rc_context(plot_style):

            shap.plots.beeswarm(
                final_shap_explanations[pair],
                max_display=15,
                show=False,
            )

            fig = plt.gcf()

            plt.title(
                f"{pair[0].capitalize()} {pair[1]} — {model_name}\n"
                "Development-data explanation of the final predictor"
            )

            save_manuscript_figure(
                fig,
                f"Figure_5{suffix}_SHAP",
                f"SHAP beeswarm for {pair[0]} Set {pair[1]} "
                f"({model_name}). Observations and reference information "
                "come from development data. Feature colours indicate "
                "feature values; horizontal positions indicate SHAP "
                "contributions. Report the explainer method from "
                "SHAP_explanation_diagnostics.csv. Associations are "
                "not evidence of causality.",
            )

            plt.show()
            plt.close(fig)

else:
    print("Figure 5 skipped: complete the corrected Block 12 first.")


# ============================================================
# 17.9 FIGURE 6 — FIXED-TEMPERATURE CANDIDATE HEATMAP
# ============================================================

if (
    "candidate_screening" in globals()
    and "candidate_schedules" in globals()
):

    screening = candidate_screening.copy()

    if screening["temperature_K"].nunique() != 1:
        raise ValueError("Figure 6 requires one operating temperature.")

    operating_temperature = screening["temperature_K"].iloc[0]

    # Choose schedules by development frequency, before considering
    # predictions. Record precisely which schedules are displayed.
    frequencies = (
        data.loc[
            development_indices["full"],
            ["sintering_temperature_K", "sintering_time_h"],
        ]
        .groupby(
            ["sintering_temperature_K", "sintering_time_h"]
        )
        .size()
        .rename("Development_measurement_count")
        .reset_index()
    )

    displayed_schedules = (
        candidate_schedules.merge(
            frequencies,
            on=["sintering_temperature_K", "sintering_time_h"],
            how="left",
            validate="one_to_one",
        )
        .sort_values(
            ["Development_measurement_count", "schedule_id"],
            ascending=[False, True],
        )
        .head(MAX_HEATMAP_SCHEDULES)
    )

    schedule_ids = displayed_schedules["schedule_id"].tolist()

    eligible_values = screening[
        "B__Predicted_log10_sigma"
    ].where(screening["B__Passes_heuristic_checks"])

    screening["Heatmap_value"] = eligible_values

    heatmap_data = screening.pivot(
        index="formula",
        columns="schedule_id",
        values="Heatmap_value",
    ).reindex(columns=schedule_ids)

    schedule_labels = [
        f"{row.sintering_temperature_K:g} K\n"
        f"{row.sintering_time_h:g} h"
        for row in displayed_schedules.itertuples(index=False)
    ]

    save_manuscript_table(
        displayed_schedules,
        "Table_S4_heatmap_processing_schedules.csv",
    )

    heatmap_data.to_csv(
        os.path.join(
            MANUSCRIPT_TABLES,
            "Figure_6_heatmap_values.csv",
        )
    )

    if heatmap_data.notna().any().any():

        with plt.rc_context(plot_style):

            fig, ax = plt.subplots(
                figsize=(
                    max(8, 0.65 * len(schedule_ids) + 4),
                    max(6, 0.32 * len(heatmap_data) + 2),
                ),
                layout="constrained",
            )

            ax.set_facecolor("#D9D9D9")

            sns.heatmap(
                heatmap_data,
                mask=heatmap_data.isna(),
                cmap="viridis",
                ax=ax,
                linewidths=0.3,
                linecolor="white",
                xticklabels=schedule_labels,
                yticklabels=True,
                cbar_kws={
                    "label": "Predicted log10(σ / [S cm⁻¹])"
                },
            )

            ax.set_xlabel("Sintering temperature and time")
            ax.set_ylabel("Candidate formula")

            ax.set_title(
                f"Candidate screening at {operating_temperature:g} K\n"
                "Grey cells: flagged by applicability checks"
            )

            ax.tick_params(axis="y", labelsize=8)

            save_manuscript_figure(
                fig,
                "Figure_6_candidate_heatmap",
                f"Set B predicted candidate conductivity at "
                f"{operating_temperature:g} K for selected observed "
                "processing schedules. Displayed schedules were chosen "
                "by development measurement frequency, not predicted "
                "performance. Grey cells fail the existing heuristic "
                "applicability checks. Predictions do not establish "
                "phase stability or synthesis feasibility.",
            )

            plt.show()
            plt.close(fig)

    else:
        print(
            "Figure 6 omitted: no displayed candidates pass the "
            "existing applicability checks."
        )

else:
    print("Figure 6 skipped: complete candidate screening first.")


# ============================================================
# 17.10 SUPPLEMENTARY TABLES AND CAPTION INDEX
# ============================================================

supplementary_objects = [
    ("classification_metrics", "Table_S5_classification_metrics.csv"),
    ("classification_reports", "Table_S6_classification_reports.csv"),
    ("final_shap_importance_table", "Table_S7_SHAP_rankings.csv"),
    ("physics_retention_table", "Table_S8_physics_retention.csv"),
    (
        "final_test_metrics_by_composition",
        "Table_S9_test_errors_by_composition.csv",
    ),
    ("validation_metrics", "Table_S10_validation_metrics.csv"),
    ("tuning_summary", "Table_S11_tuning_summary.csv"),
]

for object_name, filename in supplementary_objects:

    frame = globals().get(object_name)

    if isinstance(frame, pd.DataFrame):
        save_manuscript_table(frame, filename)

pd.DataFrame(captions).to_csv(
    os.path.join(
        MANUSCRIPT_TABLES,
        "figure_caption_drafts.csv",
    ),
    index=False,
)

with open(
    os.path.join(
        MANUSCRIPT_TABLES,
        "manuscript_figure_notes.json",
    ),
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        {
            "Figure_1": "Use the existing workflow diagram.",
            "classification": (
                "Use Block 11B figures as supplementary material. "
                "Do not choose a cutoff by maximizing test F1."
            ),
            "all_model_parity": (
                "Use Block 11 all-model parity figures as supplementary."
            ),
            "uncertainty": (
                "Use Block 14/16 figures as supplementary; "
                "bootstrap spread is not a calibrated prediction interval."
            ),
            "activation_energy": (
                "Do not include prediction figures until the "
                "sample/regime dataset and model evaluation are complete."
            ),
            "carrier_and_atmosphere_maps": (
                "Not supported by the current targets and atmosphere data."
            ),
        },
        file,
        indent=2,
    )

print("\nBLOCK 17 COMPLETE")
print("Main figures:", MANUSCRIPT_FIGURES)
print("Main and supplementary tables:", MANUSCRIPT_TABLES)
print("Draft captions: figure_caption_drafts.csv")
print("No models were fitted or selected.")


