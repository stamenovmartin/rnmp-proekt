"""Tests for data loader."""
import pytest
import pandas as pd
from pathlib import Path
from src.data.loader import DataLoader, DataPreprocessor


class TestDataLoader:
    """Test DataLoader class."""

    def test_init(self):
        """Test DataLoader initialization."""
        loader = DataLoader()
        assert loader is not None
        assert loader.config is not None
        print("✓ DataLoader initialized successfully")

    def test_load_train_data_without_identity(self):
        """Test loading transaction data only."""
        loader = DataLoader()

        # This will load real data if it exists
        try:
            df = loader.load_train_data(merge_identity=False)

            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            assert 'isFraud' in df.columns

            print(f"✓ Loaded {len(df)} transactions")
        except FileNotFoundError:
            pytest.skip("Training data not found")


class TestDataPreprocessor:
    """Test DataPreprocessor class."""

    def test_init(self):
        """Test DataPreprocessor initialization."""
        preprocessor = DataPreprocessor()
        assert preprocessor is not None
        print("✓ DataPreprocessor initialized successfully")

    def test_clean_data(self):
        """Test data cleaning with sample data."""
        preprocessor = DataPreprocessor()

        # Create sample data
        df = pd.DataFrame({
            'TransactionID': [1, 2, 3, 3],  # Has duplicate
            'amount': [100, 200, 300, 300],
            'missing_col': [1, None, None, None],  # >50% missing
            'isFraud': [0, 1, 0, 0]
        })

        cleaned = preprocessor.clean_data(df)

        # Should remove duplicates
        assert len(cleaned) == 3

        # Should drop high-missing column
        assert 'missing_col' not in cleaned.columns

        # Should keep target
        assert 'isFraud' in cleaned.columns

        print("✓ Data cleaning works correctly")

    def test_handle_missing_values(self):
        """Test missing value handling."""
        preprocessor = DataPreprocessor()

        df = pd.DataFrame({
            'numeric_col': [1.0, 2.0, None, 4.0],
            'categorical_col': ['A', 'B', None, 'C'],
            'isFraud': [0, 1, 0, 1]
        })

        filled = preprocessor.handle_missing_values(df)

