Place the real dataset at `data/heart.csv`.

Expected format:
- CSV file
- target column: `target`
- feature columns:
  `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`

Useful commands:

```bash
python check_dataset.py --data data/heart.csv
python train_model.py --data data/heart.csv
python evaluate_model.py --data data/heart.csv
```

Notes:
- Do not commit the real dataset unless you are sure licensing/privacy allows it.
- Keep only sample or synthetic CSVs in the repository.
