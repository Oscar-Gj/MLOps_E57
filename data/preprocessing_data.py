import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, OrdinalEncoder
import category_encoders as ce


def preprocess(input_csv: Path, output_csv: Path) -> None:
    # Ensure output directory exists
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    if not input_csv.exists():
        raise FileNotFoundError(f"No se encontró el archivo de entrada: {input_csv}")
    df = pd.read_csv(input_csv)

    # Drop known garbage column if present
    if 'mixed' in df.columns:
        df = df.drop(columns=['mixed'])

    # Define columns by type as in the notebook
    num_col = ['duration', 'amount', 'age']
    ord_col = ['employment_duration', 'installment_rate', 'present_residence',
               'property', 'number_credits', 'job']
    cat_col = [
        'status', 'credit_history', 'purpose', 'savings', 'people_liable',
        'personal_status_sex', 'other_debtors', 'other_installment_plans',
        'housing', 'telephone', 'foreign_worker'
    ]

    # Coerce dtypes similar to the notebook setup
    for c in num_col:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    for c in ord_col:
        if c in df.columns:
            df[c] = df[c].astype('object')
    for c in cat_col:
        if c in df.columns:
            df[c] = df[c].astype('object')

    # Clean object columns: remove internal whitespace and mask rare categories (<7)
    obj_cols = df.select_dtypes(include=['object', 'string']).columns
    if len(obj_cols) > 0:
        df[obj_cols] = df[obj_cols].replace(r"\s+", "", regex=True)
        counts = df[obj_cols].apply(lambda s: s.map(s.value_counts(dropna=False)))
        df[obj_cols] = df[obj_cols].mask(counts < 7)

    # Target handling: convert to numeric, keep only 0/1, drop NaNs, invert 0<->1
    if 'credit_risk' not in df.columns:
        raise KeyError("La columna objetivo 'credit_risk' no existe en el dataset.")
    df['credit_risk'] = pd.to_numeric(df['credit_risk'], errors='coerce')
    df = df[df['credit_risk'].isin([0.0, 1.0])]
    df = df.dropna(subset=['credit_risk'])
    df['credit_risk'] = df['credit_risk'].apply(lambda x: 0 if x == 1 else 1)

    # Split features/target
    X = df.drop(columns=['credit_risk'])
    y = df['credit_risk'].astype(int)

    # Prepare containers with existing indices to preserve alignment
    parts = []

    # Numeric pipeline: impute median -> MinMax [0,1] -> PowerTransformer (Yeo-Johnson)
    present_num = [c for c in num_col if c in X.columns]
    if present_num:
        num_values = X[present_num].copy()
        num_imputed = pd.DataFrame(
            SimpleImputer(strategy='median').fit_transform(num_values),
            columns=present_num, index=X.index
        )
        num_scaled = pd.DataFrame(
            MinMaxScaler(feature_range=(0, 1)).fit_transform(num_imputed),
            columns=present_num, index=X.index
        )
        num_final = pd.DataFrame(
            PowerTransformer(method='yeo-johnson').fit_transform(num_scaled),
            columns=present_num, index=X.index
        )
        parts.append(num_final)

    # Nominal pipeline: impute most_frequent -> BinaryEncoder
    present_cat = [c for c in cat_col if c in X.columns]
    if present_cat:
        cat_values = X[present_cat].copy()
        cat_imputed = pd.DataFrame(
            SimpleImputer(strategy='most_frequent').fit_transform(cat_values),
            columns=present_cat, index=X.index
        )
        be = ce.BinaryEncoder(cols=present_cat, return_df=True)
        cat_encoded = be.fit_transform(cat_imputed)
        # category_encoders typically returns the original columns plus encoded ones; drop originals if duplicated
        cat_encoded = cat_encoded.drop(columns=[c for c in present_cat if c in cat_encoded.columns], errors='ignore')
        parts.append(cat_encoded.set_index(X.index))

    # Ordinal pipeline: impute most_frequent -> OrdinalEncoder
    present_ord = [c for c in ord_col if c in X.columns]
    if present_ord:
        ord_values = X[present_ord].copy()
        ord_imputed = pd.DataFrame(
            SimpleImputer(strategy='most_frequent').fit_transform(ord_values),
            columns=present_ord, index=X.index
        )
        ord_encoded_arr = OrdinalEncoder().fit_transform(ord_imputed)
        ord_encoded = pd.DataFrame(ord_encoded_arr, columns=present_ord, index=X.index)
        parts.append(ord_encoded)

    # Any leftover columns not in the above lists (if any), pass-through after basic fill for objects
    accounted = set(present_num + present_cat + present_ord)
    leftover = [c for c in X.columns if c not in accounted]
    if leftover:
        # Simple strategy: forward-fill then back-fill for object leftovers to avoid NaN-only cols
        lf = X[leftover].copy()
        for c in lf.columns:
            if lf[c].dtype == object:
                mode = lf[c].mode(dropna=True)
                if not mode.empty:
                    lf[c] = lf[c].fillna(mode.iloc[0])
        parts.append(lf)

    # Concatenate all processed parts
    X_processed = pd.concat(parts, axis=1)

    # Align columns order deterministically: numeric, nominal-encoded, ordinal, leftover
    # (Already appended in that order.)

    # Append target as last column
    X_processed['credit_risk'] = y.values

    # Save CSV
    X_processed.to_csv(output_csv, index=False)


def main():
    script_dir = Path(__file__).resolve().parent
    default_in = script_dir / 'raw' / 'german_credit_modified.csv'
    default_out = script_dir / 'processed' / 'data_preprocessedad.csv'

    parser = argparse.ArgumentParser(description='Preprocesamiento del dataset German Credit (modificado)')
    parser.add_argument('--input', '-i', type=Path, default=default_in,
                        help='Ruta al CSV de entrada (modificado).')
    parser.add_argument('--output', '-o', type=Path, default=default_out,
                        help='Ruta al CSV procesado de salida.')
    args = parser.parse_args()

    preprocess(args.input, args.output)


if __name__ == '__main__':
    main()

