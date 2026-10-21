# Student Performance Prediction System using Decision Tree Classifier

A machine-learning project that predicts student academic outcomes (Pass/Fail) using a Decision Tree Classifier.

## Dataset
`Student_Performance_DT.csv` contains student attributes including study hours, attendance, previous score, assignments completed, sleep hours, participation, internet access, extracurricular activity, and final result.

## Technologies
- Python
- Pandas
- NumPy
- Scikit-learn

## Files
- `main.py` — loads and preprocesses the dataset, trains a classifier, evaluates it, and predicts an example result.
- `Student_Performance_DT.csv` — dataset.
- `requirements.txt` — dependencies.
- `.gitignore` — ignores virtual environments and cache files.

## Run
Install dependencies:

```bash
pip install -r requirements.txt
```

Run the program:

```bash
python main.py
```

## Evaluation
The program reports accuracy, precision, recall, and F1-score on a held-out test set.

## Note
This is an educational demonstration. Predictions depend on the quality and representativeness of the supplied dataset.
