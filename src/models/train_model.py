import sklearn
import pandas as pd 
from sklearn import ensemble
import joblib
import numpy as np

print(joblib.__version__)

# -- Load datasets
X_train = pd.read_csv("data/preprocessed/X_train.csv", low_memory=False)
X_test = pd.read_csv("data/preprocessed/X_test.csv", low_memory=False)
y_train = pd.read_csv("data/preprocessed/y_train.csv")
y_test = pd.read_csv("data/preprocessed/y_test.csv")

# -- Convert y to 1D arrays
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

# -- Clean X_train and X_test (remove weird chars, force numeric)
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.applymap(lambda x: str(x).replace("\xa0", "").replace("Â", "") if isinstance(x, str) else x)
    df = df.apply(pd.to_numeric, errors="coerce")  # force numbers, NaN if impossible
    return df

X_train = clean_dataframe(X_train)
X_test = clean_dataframe(X_test)

# -- Handle missing values if any (simple strategy: fill with 0, or you can use an Imputer)
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# -- Train the model
rf_classifier = ensemble.RandomForestClassifier(n_jobs=-1, random_state=42)
rf_classifier.fit(X_train, y_train)

# -- Save the trained model to a file
model_filename = "./models/trained_model.joblib"
joblib.dump(rf_classifier, model_filename)
print("Model trained and saved successfully.")
