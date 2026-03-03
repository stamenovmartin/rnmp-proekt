import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    roc_curve
)

from src.utils.config import get_config


class FraudDetectionPipeline:
    """Complete pipeline for fraud detection baseline."""

    def __init__(self):
        """Initialize pipeline."""
        self.config = get_config()
        self.data_config = self.config['data']
        self.model_config = self.config['model']
        self.proc_config = self.config['processing']

        # Storage for artifacts
        self.label_encoders = {}
        self.scaler = None
        self.feature_names = None

        # Create output directory - use absolute path
        project_root = Path(__file__).parent.parent
        self.output_dir = project_root / 'output'
        self.output_dir.mkdir(exist_ok=True)

        print("=" * 70)
        print("FRAUD DETECTION - BASELINE ML PIPELINE")
        print("=" * 70)

    # ========================================================================
    # DATA LOADING
    # ========================================================================

    def load_data(self):
        """Load and merge transaction + identity data.

        Returns:
            pandas.DataFrame: Merged dataset
        """
        print("\n STEP 1: LOADING DATA")
        print("-" * 70)

        # Get project root directory
        project_root = Path(__file__).parent.parent
        raw_path = project_root / self.data_config['raw_path']

        # Load transaction data
        trans_file = raw_path / self.data_config['train_transaction']
        print(f"Loading transactions: {trans_file}")

        if not trans_file.exists():
            raise FileNotFoundError(
                f"Transaction file not found: {trans_file}\n"
                f"Make sure your data is in: {raw_path}"
            )

        df_trans = pd.read_csv(trans_file)
        print(f"✓ Transactions loaded: {df_trans.shape}")

        # Load identity data
        identity_file = raw_path / self.data_config['train_identity']
        print(f"Loading identity: {identity_file}")

        if not identity_file.exists():
            raise FileNotFoundError(
                f"Identity file not found: {identity_file}\n"
                f"Make sure your data is in: {raw_path}"
            )

        df_identity = pd.read_csv(identity_file)
        print(f"✓ Identity loaded: {df_identity.shape}")

        # Merge on TransactionID
        print("\nMerging on TransactionID...")
        df = pd.merge(df_trans, df_identity, on='TransactionID', how='left')
        print(f"✓ Merged dataset: {df.shape}")

        # Show target distribution
        target_col = self.data_config['target_column']
        fraud_count = df[target_col].sum()
        fraud_rate = (fraud_count / len(df)) * 100

        print(f"\n Target Distribution:")
        print(f"   Total samples: {len(df):,}")
        print(f"   Fraud cases: {fraud_count:,} ({fraud_rate:.2f}%)")
        print(f"   Non-fraud cases: {len(df) - fraud_count:,} ({100 - fraud_rate:.2f}%)")

        return df

    # ========================================================================
    # DATA CLEANING
    # ========================================================================

    def clean_data(self, df):
        """Clean dataset: remove high-missing columns and duplicates.

        Args:
            df: Raw DataFrame

        Returns:
            pandas.DataFrame: Cleaned data
        """
        print("\n STEP 2: DATA CLEANING")
        print("-" * 70)

        initial_shape = df.shape
        target_col = self.data_config['target_column']

        # Remove columns with too many missing values
        max_missing = self.proc_config['max_missing_ratio']
        print(f"Removing columns with >{max_missing * 100}% missing values...")

        missing_pct = df.isnull().sum() / len(df)
        cols_to_drop = missing_pct[missing_pct > max_missing].index.tolist()

        # Don't drop target column
        if target_col in cols_to_drop:
            cols_to_drop.remove(target_col)

        # Don't drop TransactionID
        if 'TransactionID' in cols_to_drop:
            cols_to_drop.remove('TransactionID')

        print(f"   Dropping {len(cols_to_drop)} columns")
        df = df.drop(columns=cols_to_drop)

        # Remove duplicates
        print("\nRemoving duplicate rows...")
        before_dedup = len(df)
        df = df.drop_duplicates()
        removed = before_dedup - len(df)
        print(f"   Removed {removed:,} duplicates")

        # Drop TransactionID (not a feature)
        if 'TransactionID' in df.columns:
            df = df.drop(columns=['TransactionID'])

        print(f"\n✓ Cleaned dataset: {initial_shape} → {df.shape}")

        return df

    def handle_missing_values(self, df):
        """Fill missing values.

        Args:
            df: DataFrame with missing values

        Returns:
            pandas.DataFrame: Data with filled missing values
        """
        print("\n STEP 3: HANDLING MISSING VALUES")
        print("-" * 70)

        target_col = self.data_config['target_column']

        # Make a copy to avoid warnings
        df = df.copy()

        # Numeric columns: fill with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != target_col]

        print(f"Filling {len(numeric_cols)} numeric columns with median...")
        for col in numeric_cols:
            missing = df[col].isnull().sum()
            if missing > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)  # Changed this line

        # Categorical columns: fill with 'Unknown'
        categorical_cols = df.select_dtypes(include=['object']).columns

        print(f"Filling {len(categorical_cols)} categorical columns with 'Unknown'...")
        for col in categorical_cols:
            missing = df[col].isnull().sum()
            if missing > 0:
                df[col] = df[col].fillna('Unknown')  # Changed this line

        # DOUBLE CHECK - fill any remaining NaN with 0
        remaining = df.isnull().sum().sum()
        if remaining > 0:
            print(f"\n⚠  Still {remaining} NaN values, filling with 0...")
            df = df.fillna(0)

        print(f"\n✓ Remaining missing values: {df.isnull().sum().sum()}")

        return df

    # ========================================================================
    # FEATURE ENCODING
    # ========================================================================

    def encode_categorical(self, df):
        """Encode categorical features.

        Args:
            df: DataFrame with categorical features

        Returns:
            pandas.DataFrame: Encoded data
        """
        print("\n STEP 4: ENCODING CATEGORICAL FEATURES")
        print("-" * 70)

        categorical_cols = df.select_dtypes(include=['object']).columns
        max_categories = self.proc_config['max_categories']

        print(f"Found {len(categorical_cols)} categorical columns")

        for col in categorical_cols:
            n_unique = df[col].nunique()

            if n_unique <= max_categories:
                print(f"   Label encoding '{col}' ({n_unique} categories)")
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                print(f"   Dropping high-cardinality '{col}' ({n_unique} categories)")
                df = df.drop(columns=[col])

        print(f"\n✓ Encoded {len(self.label_encoders)} categorical features")

        return df

    # ========================================================================
    # TRAIN/TEST SPLIT
    # ========================================================================

    def split_data(self, df):
        """Split into train and test sets.

        Args:
            df: Preprocessed DataFrame

        Returns:
            X_train, X_test, y_train, y_test
        """
        print("\n️  STEP 5: TRAIN/TEST SPLIT")
        print("-" * 70)

        target_col = self.data_config['target_column']
        test_size = self.data_config['train_test_split']
        random_state = self.data_config['random_state']

        # Separate features and target
        y = df[target_col]
        X = df.drop(columns=[target_col])

        # Store feature names
        self.feature_names = X.columns.tolist()

        print(f"Total samples: {len(X):,}")
        print(f"Total features: {len(self.feature_names):,}")

        # Split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        print(f"\n✓ Training set: {len(X_train):,} samples ({len(X_train) / len(X) * 100:.1f}%)")
        print(f"✓ Test set: {len(X_test):,} samples ({len(X_test) / len(X) * 100:.1f}%)")
        print(f"   Train fraud rate: {y_train.mean() * 100:.2f}%")
        print(f"   Test fraud rate: {y_test.mean() * 100:.2f}%")

        # Scale features
        print("\nScaling features with StandardScaler...")
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    # ========================================================================
    # MODEL TRAINING
    # ========================================================================

    def train_models(self, X_train, y_train):
        """Train baseline models.

        Args:
            X_train: Training features
            y_train: Training target

        Returns:
            dict: Trained models
        """
        print("\n STEP 6: TRAINING BASELINE MODELS")
        print("-" * 70)

        models = {}

        # Logistic Regression
        print("\n1. Logistic Regression")
        lr = LogisticRegression(
            max_iter=self.model_config['lr_max_iter'],
            class_weight=self.model_config['lr_class_weight'],
            random_state=self.data_config['random_state'],
            n_jobs=-1
        )
        lr.fit(X_train, y_train)
        models['Logistic Regression'] = lr
        print("   ✓ Trained")

        # Decision Tree
        print("\n2. Decision Tree")
        dt = DecisionTreeClassifier(
            max_depth=self.model_config['dt_max_depth'],
            min_samples_split=self.model_config['dt_min_samples_split'],
            min_samples_leaf=self.model_config['dt_min_samples_leaf'],
            class_weight='balanced',
            random_state=self.data_config['random_state']
        )
        dt.fit(X_train, y_train)
        models['Decision Tree'] = dt
        print("   ✓ Trained")

        # Random Forest
        print("\n3. Random Forest")
        rf = RandomForestClassifier(
            n_estimators=self.model_config['rf_n_estimators'],
            max_depth=self.model_config['rf_max_depth'],
            class_weight='balanced',
            random_state=self.data_config['random_state'],
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        models['Random Forest'] = rf
        print("   ✓ Trained")

        return models

    # ========================================================================
    # MODEL EVALUATION
    # ========================================================================

    def evaluate_models(self, models, X_test, y_test):
        """Evaluate all models.

        Args:
            models: Dict of trained models
            X_test: Test features
            y_test: Test target

        Returns:
            pandas.DataFrame: Results table
        """
        print("\n STEP 7: MODEL EVALUATION")
        print("-" * 70)

        results = []

        for name, model in models.items():
            print(f"\n{name}:")

            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Metrics
            auc = roc_auc_score(y_test, y_pred_proba)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            print(f"   AUC:       {auc:.4f}")
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall:    {recall:.4f}")
            print(f"   F1 Score:  {f1:.4f}")

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            print(f"\n   Confusion Matrix:")
            print(f"   TN: {cm[0, 0]:,}  FP: {cm[0, 1]:,}")
            print(f"   FN: {cm[1, 0]:,}  TP: {cm[1, 1]:,}")

            results.append({
                'Model': name,
                'AUC': auc,
                'Precision': precision,
                'Recall': recall,
                'F1': f1
            })

        results_df = pd.DataFrame(results)
        return results_df

    def plot_results(self, models, X_test, y_test, results_df):
        """Create visualization plots.

        Args:
            models: Dict of trained models
            X_test: Test features
            y_test: Test target
            results_df: Results DataFrame
        """
        print("\n STEP 8: CREATING VISUALIZATIONS")
        print("-" * 70)

        # 1. ROC Curves
        plt.figure(figsize=(10, 6))

        for name, model in models.items():
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc = roc_auc_score(y_test, y_pred_proba)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)

        plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Baseline Models', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        roc_path = self.output_dir / 'roc_curves.png'
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved ROC curves: {roc_path}")
        plt.close()

        # 2. Metrics Comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        metrics = ['AUC', 'Precision', 'Recall']
        colors = ['#3b82f6', '#ef4444', '#22c55e']

        for idx, (metric, color) in enumerate(zip(metrics, colors)):
            axes[idx].bar(results_df['Model'], results_df[metric], color=color, alpha=0.7)
            axes[idx].set_title(f'{metric} Comparison', fontweight='bold')
            axes[idx].set_ylabel(metric)
            axes[idx].set_ylim([0, 1])
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        metrics_path = self.output_dir / 'metrics_comparison.png'
        plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved metrics comparison: {metrics_path}")
        plt.close()

    # ========================================================================
    # SAVE CLEAN DATASET
    # ========================================================================
    def save_clean_dataset(self, df, X_train, X_test, y_train, y_test):
        """Save cleaned and preprocessed datasets."""
        print("\n SAVING CLEAN DATASETS")
        print("-" * 70)

        # Always resolve absolute project path
        project_root = Path(__file__).parent.parent
        processed_dir = project_root / 'data' / 'processed'
        processed_dir.mkdir(parents=True, exist_ok=True)

        # Save preprocessed full dataset
        full_clean_path = processed_dir / 'preprocessed_full.csv'
        df.to_csv(full_clean_path, index=False)
        print(f"✓ Saved full preprocessed data: {full_clean_path}")

        # Save train/test splits
        X_train_df = pd.DataFrame(X_train, columns=self.feature_names)
        X_test_df = pd.DataFrame(X_test, columns=self.feature_names)

        train_data = pd.concat([X_train_df, y_train.reset_index(drop=True)], axis=1)
        test_data = pd.concat([X_test_df, y_test.reset_index(drop=True)], axis=1)

        train_path = processed_dir / 'train_clean.csv'
        test_path = processed_dir / 'test_clean.csv'

        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)

        print(f"✓ Saved train data: {train_path} ({len(train_data):,} rows)")
        print(f"✓ Saved test data: {test_path} ({len(test_data):,} rows)")

        # Save pickle versions
        train_pkl = processed_dir / 'train_clean.pkl'
        test_pkl = processed_dir / 'test_clean.pkl'

        train_data.to_pickle(train_pkl)
        test_data.to_pickle(test_pkl)

        print(f"✓ Saved train pickle: {train_pkl}")
        print(f"✓ Saved test pickle: {test_pkl}")
    # ========================================================================
    # SAVE ARTIFACTS
    # ========================================================================


    def save_artifacts(self, models, results_df, X_train):
        """Save all artifacts for next phase.

        Args:
            models: Trained models
            results_df: Results DataFrame
            X_train: Training data (for shape info)
        """
        print("\n STEP 9: SAVING ARTIFACTS")
        print("-" * 70)

        project_root = Path(__file__).parent.parent
        baseline_dir = project_root / self.model_config['baseline_dir']
        baseline_dir.mkdir(parents=True, exist_ok=True)

        # Save models
        for name, model in models.items():
            model_file = baseline_dir / f"{name.lower().replace(' ', '_')}.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            print(f"✓ Saved model: {model_file}")

        # Save preprocessing artifacts
        artifacts = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }

        artifacts_file = baseline_dir / 'preprocessing_artifacts.pkl'
        with open(artifacts_file, 'wb') as f:
            pickle.dump(artifacts, f)
        print(f"✓ Saved preprocessing: {artifacts_file}")

        # Save results table
        results_file = self.output_dir / 'baseline_results.csv'
        results_df.to_csv(results_file, index=False)
        print(f"✓ Saved results: {results_file}")

        # Save summary report
        report_file = self.output_dir / 'baseline_report.txt'
        with open(report_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("FRAUD DETECTION - BASELINE MODEL REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")

            f.write("DATASET SUMMARY:\n")
            f.write(f"  Training samples: {len(X_train):,}\n")
            f.write(f"  Features: {len(self.feature_names):,}\n\n")

            f.write("MODEL RESULTS:\n")
            f.write("-" * 70 + "\n")
            f.write(results_df.to_string(index=False))
            f.write("\n\n")

            best_model = results_df.loc[results_df['AUC'].idxmax()]
            f.write(f"BEST MODEL: {best_model['Model']}\n")
            f.write(f"  AUC: {best_model['AUC']:.4f}\n")
            f.write(f"  Precision: {best_model['Precision']:.4f}\n")
            f.write(f"  Recall: {best_model['Recall']:.4f}\n")

        print(f"✓ Saved report: {report_file}")

    # ========================================================================
    # MAIN PIPELINE
    # ========================================================================

    def run(self):
        """Run complete pipeline."""

        # 1. Load data
        df = self.load_data()

        # 2. Clean data
        df = self.clean_data(df)

        # 3. Handle missing values
        df = self.handle_missing_values(df)

        # 4. Encode categorical
        df = self.encode_categorical(df)

        # 5. Train/test split
        X_train, X_test, y_train, y_test = self.split_data(df)

        # 6. Train models
        models = self.train_models(X_train, y_train)

        # 7. Evaluate
        results_df = self.evaluate_models(models, X_test, y_test)

        # 8.5. Save clean datasets (NEW!)
        self.save_clean_dataset(df, X_train, X_test, y_train, y_test)

        # 8. Visualize
        self.plot_results(models, X_test, y_test, results_df)

        # 9. Save artifacts
        self.save_artifacts(models, results_df, X_train)

        # Final summary
        print("\n" + "=" * 70)
        print(" PIPELINE COMPLETE!")
        print("=" * 70)
        print(f"\nOutputs saved to:")
        print(f"   Models: {self.model_config['baseline_dir']}/")
        print(f"   Results: output/")
        print(f"\nFiles generated:")
        print(f"  - baseline_results.csv (metrics table)")
        print(f"  - baseline_report.txt (summary report)")
        print(f"  - roc_curves.png (ROC curve plot)")
        print(f"  - metrics_comparison.png (bar charts)")
        print(f"  - Models: logistic_regression.pkl, decision_tree.pkl, random_forest.pkl")
        print(f"  - Preprocessing: preprocessing_artifacts.pkl")



def main():
    """Main entry point."""
    pipeline = FraudDetectionPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()