"""
Preview script to analyze and compare Porto Seguro and IEEE-CIS Fraud Detection datasets
This script generates summary statistics and previews for both datasets
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATASETS_DIR = Path(__file__).parent

def merge_ieee_datasets(transaction_path, identity_path, sample_rows=None):
    """Merge IEEE transaction and identity datasets"""
    print(f"\nMerging IEEE datasets...")
    print(f"  Transaction file: {transaction_path.name}")
    print(f"  Identity file: {identity_path.name}")
    
    # Load transaction data
    if sample_rows:
        df_transaction = pd.read_csv(transaction_path, nrows=sample_rows)
        print(f"  Loaded {len(df_transaction)} transaction rows (sample)")
    else:
        df_transaction = pd.read_csv(transaction_path)
        print(f"  Loaded {len(df_transaction)} transaction rows (full)")
    
    # Load identity data
    if sample_rows:
        df_identity = pd.read_csv(identity_path, nrows=sample_rows)
        print(f"  Loaded {len(df_identity)} identity rows (sample)")
    else:
        df_identity = pd.read_csv(identity_path)
        print(f"  Loaded {len(df_identity)} identity rows (full)")
    
    # Merge on TransactionID
    df_merged = df_transaction.merge(df_identity, on='TransactionID', how='left')
    print(f"  Merged dataset: {df_merged.shape[0]:,} rows × {df_merged.shape[1]} columns")
    
    # Calculate identity coverage (how many transactions have identity data)
    identity_cols = [col for col in df_merged.columns if col not in df_transaction.columns or col == 'TransactionID']
    if len(identity_cols) > 1:  # More than just TransactionID
        has_identity = df_merged[identity_cols].iloc[:, 1:].notna().any(axis=1).sum()
        coverage = (has_identity / len(df_merged)) * 100
        print(f"  Identity coverage: {coverage:.2f}% of transactions have identity data")
    
    return df_merged

def analyze_dataset(dataset_path, dataset_name, is_dataframe=False, df=None):
    """Analyze a single dataset and return summary statistics
    
    Args:
        dataset_path: Path to dataset file (or used for display if is_dataframe=True)
        dataset_name: Name of the dataset
        is_dataframe: If True, analyze the provided dataframe instead of loading from path
        df: Dataframe to analyze (required if is_dataframe=True)
    """
    print(f"\n{'='*80}")
    print(f"ANALYZING: {dataset_name.upper()}")
    print(f"{'='*80}")
    
    try:
        # Load dataset
        if is_dataframe and df is not None:
            print(f"\nAnalyzing merged dataset")
            df = df.copy()
        else:
            print(f"\nLoading dataset from: {dataset_path}")
            df = pd.read_csv(dataset_path, nrows=100000)  # Load sample for faster analysis
        
        print(f"Loaded {len(df)} rows" + (" (sample)" if not is_dataframe else ""))
        
        # Basic Info
        print(f"\n{'-'*80}")
        print("BASIC INFORMATION")
        print(f"{'-'*80}")
        print(f"Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Data Types
        print(f"\n{'-'*80}")
        print("DATA TYPES")
        print(f"{'-'*80}")
        dtype_counts = df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"  {dtype}: {count} columns")
        
        # Missing Values
        print(f"\n{'-'*80}")
        print("MISSING VALUES")
        print(f"{'-'*80}")
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Missing %': missing_pct
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
        
        if len(missing_df) > 0:
            print(f"Columns with missing values: {len(missing_df)}")
            print("\nTop 10 columns with most missing values:")
            print(missing_df.head(10).to_string())
            print(f"\nTotal missing values: {missing.sum():,} ({missing_pct.sum()/len(df.columns):.2f}% of all cells)")
        else:
            print("No missing values found!")
        
        # Target Distribution (if exists)
        target_cols = ['target', 'isFraud', 'is_claim', 'TARGET']
        target_col = None
        for col in target_cols:
            if col in df.columns:
                target_col = col
                break
        
        if target_col:
            print(f"\n{'-'*80}")
            print(f"TARGET VARIABLE: '{target_col}'")
            print(f"{'-'*80}")
            target_counts = df[target_col].value_counts()
            target_pct = df[target_col].value_counts(normalize=True) * 100
            
            print("\nDistribution:")
            for value, count in target_counts.items():
                pct = target_pct[value]
                print(f"  {value}: {count:,} ({pct:.2f}%)")
            
            print(f"\nClass Imbalance Ratio: {target_counts.min() / target_counts.max():.4f}")
            if target_pct.min() < 5:
                print("WARNING: Highly imbalanced dataset (<5% minority class)")
        
        # Numerical Features Summary
        print(f"\n{'-'*80}")
        print("NUMERICAL FEATURES SUMMARY")
        print(f"{'-'*80}")
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col and target_col in numerical_cols:
            numerical_cols.remove(target_col)
        
        if numerical_cols:
            print(f"Number of numerical features: {len(numerical_cols)}")
            print(f"\nSample numerical statistics (first 5 features):")
            print(df[numerical_cols[:5]].describe().to_string())
        
        # Categorical Features Summary
        print(f"\n{'-'*80}")
        print("CATEGORICAL FEATURES SUMMARY")
        print(f"{'-'*80}")
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            print(f"Number of categorical features: {len(categorical_cols)}")
            print(f"\nSample categorical features (first 5):")
            for col in categorical_cols[:5]:
                unique_count = df[col].nunique()
                print(f"  {col}: {unique_count} unique values")
                if unique_count <= 10:
                    print(f"    Values: {df[col].value_counts().to_dict()}")
        
        # Sample Data
        print(f"\n{'-'*80}")
        print("SAMPLE DATA (First 3 rows)")
        print(f"{'-'*80}")
        print(df.head(3).to_string())
        
        # Feature Names Preview
        print(f"\n{'-'*80}")
        print(f"ALL COLUMN NAMES ({len(df.columns)} total)")
        print(f"{'-'*80}")
        cols_per_line = 5
        for i in range(0, len(df.columns), cols_per_line):
            print("  " + ", ".join(df.columns[i:i+cols_per_line]))
        
        return {
            'shape': df.shape,
            'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'numerical_count': len(numerical_cols),
            'categorical_count': len(categorical_cols),
            'missing_count': missing.sum(),
            'missing_pct': (missing.sum() / (len(df) * len(df.columns))) * 100,
            'target_col': target_col,
            'target_distribution': df[target_col].value_counts().to_dict() if target_col else None
        }
        
    except FileNotFoundError:
        print(f"ERROR: Dataset file not found at {dataset_path}")
        return None
    except Exception as e:
        print(f"ERROR analyzing dataset: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("="*80)
    print("DATASET PREVIEW AND COMPARISON TOOL")
    print("="*80)
    
    # Define dataset paths
    datasets = {
        'Porto Seguro': {
            'train': DATASETS_DIR / 'porto_seguro' / 'train.csv',
            'test': DATASETS_DIR / 'porto_seguro' / 'test.csv'
        },
        'IEEE Fraud': {
            'train_transaction': DATASETS_DIR / 'ieee dataset' / 'train_transaction.csv',
            'train_identity': DATASETS_DIR / 'ieee dataset' / 'train_identity.csv'
        }
    }
    
    results = {}
    
    # Analyze Porto Seguro
    porto_train = datasets['Porto Seguro']['train']
    if porto_train.exists():
        results['Porto Seguro'] = analyze_dataset(porto_train, 'Porto Seguro Safe Driver Prediction')
    else:
        print(f"\nPorto Seguro dataset not found at: {porto_train}")
        print("Please download it first using download_datasets.py")
    
    # Analyze IEEE Fraud (with merging)
    ieee_transaction = datasets['IEEE Fraud']['train_transaction']
    ieee_identity = datasets['IEEE Fraud']['train_identity']
    
    if ieee_transaction.exists() and ieee_identity.exists():
        # Merge the datasets
        df_ieee_merged = merge_ieee_datasets(ieee_transaction, ieee_identity, sample_rows=100000)
        # Analyze the merged dataset
        results['IEEE Fraud'] = analyze_dataset(
            ieee_transaction,  # Path for display purposes
            'IEEE-CIS Fraud Detection (Merged)',
            is_dataframe=True,
            df=df_ieee_merged
        )
    elif ieee_transaction.exists():
        print(f"\nIEEE Fraud transaction file found, but identity file not found at: {ieee_identity}")
        print("Analyzing transaction file only...")
        results['IEEE Fraud'] = analyze_dataset(ieee_transaction, 'IEEE-CIS Fraud Detection (Transaction Only)')
    else:
        print(f"\nIEEE Fraud dataset not found at: {ieee_transaction}")
        print("Please download it first using download_datasets.py")
    
    # Comparison Summary
    if len(results) == 2:
        print(f"\n\n{'='*80}")
        print("COMPARISON SUMMARY")
        print(f"{'='*80}")
        
        print(f"\n{'Metric':<30} {'Porto Seguro':<25} {'IEEE Fraud':<25}")
        print(f"{'-'*80}")
        
        for metric in ['Rows', 'Columns', 'Memory (MB)', 'Numerical Features', 
                      'Categorical Features', 'Missing Values %']:
            porto_val = 'N/A'
            ieee_val = 'N/A'
            
            if 'Porto Seguro' in results and results['Porto Seguro']:
                ps = results['Porto Seguro']
                if metric == 'Rows':
                    porto_val = f"{ps['shape'][0]:,}"
                elif metric == 'Columns':
                    porto_val = f"{ps['shape'][1]}"
                elif metric == 'Memory (MB)':
                    porto_val = f"{ps['memory_mb']:.2f}"
                elif metric == 'Numerical Features':
                    porto_val = f"{ps['numerical_count']}"
                elif metric == 'Categorical Features':
                    porto_val = f"{ps['categorical_count']}"
                elif metric == 'Missing Values %':
                    porto_val = f"{ps['missing_pct']:.2f}%"
            
            if 'IEEE Fraud' in results and results['IEEE Fraud']:
                ie = results['IEEE Fraud']
                if metric == 'Rows':
                    ieee_val = f"{ie['shape'][0]:,}"
                elif metric == 'Columns':
                    ieee_val = f"{ie['shape'][1]}"
                elif metric == 'Memory (MB)':
                    ieee_val = f"{ie['memory_mb']:.2f}"
                elif metric == 'Numerical Features':
                    ieee_val = f"{ie['numerical_count']}"
                elif metric == 'Categorical Features':
                    ieee_val = f"{ie['categorical_count']}"
                elif metric == 'Missing Values %':
                    ieee_val = f"{ie['missing_pct']:.2f}%"
            
            print(f"{metric:<30} {porto_val:<25} {ieee_val:<25}")
        
        print(f"\n{'='*80}")
        print("RECOMMENDATION")
        print(f"{'='*80}")
        print("\nBased on the analysis above:")
        print("- Porto Seguro: Better for moderate complexity, cleaner data, good for initial exploration")
        print("- IEEE Fraud: More complex, many features, highly imbalanced - good for advanced preprocessing")
    
    elif len(results) == 0:
        print("\n" + "="*80)
        print("NO DATASETS FOUND")
        print("="*80)
        print("\nPlease download the datasets first:")
        print("1. Set up Kaggle API credentials (see KAGGLE_SETUP_INSTRUCTIONS.md)")
        print("2. Run: py datasets/download_datasets.py")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
