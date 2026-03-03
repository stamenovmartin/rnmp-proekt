## Setup Instructions
```
git clone <repo>
cd rnmp-proekt

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### Create the following directory in project root
```
data/raw/
├── train_transaction.csv
├── train_identity.csv
├── test_transaction.csv
└── test_identity.csv
```

### Run the following script
```
python src/train_baseline.py
```

This script will generate:

```commandline
data/processed/ → cleaned datasets
models/baseline/ → trained models (.pkl)
output/ → reports and evaluation plots
```

