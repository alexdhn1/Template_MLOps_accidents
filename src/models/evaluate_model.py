import pandas as pd
import numpy as np
from joblib import load
import json
from pathlib import Path
from sklearn.metrics import accuracy_score


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les colonnes : supprime caractères bizarres et force en numérique"""
    df = df.applymap(lambda x: str(x).replace("\xa0", "").replace("Â", "") if isinstance(x, str) else x)
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.fillna(0)  # ou utiliser un Imputer si tu veux


# -- Load data
X_train = pd.read_csv("data/preprocessed/X_train.csv", low_memory=False)
X_test = pd.read_csv("data/preprocessed/X_test.csv", low_memory=False)
y_train = pd.read_csv("data/preprocessed/y_train.csv")
y_test = pd.read_csv("data/preprocessed/y_test.csv")

# -- Convert y
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

# -- Clean X data
X_train = clean_dataframe(X_train)
X_test = clean_dataframe(X_test)


def main(repo_path: Path):
    model = load(repo_path / "models/trained_model.joblib")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    metrics = {"accuracy": accuracy}
    accuracy_path = repo_path / "metrics/accuracy.json"
    accuracy_path.parent.mkdir(parents=True, exist_ok=True)  # crée le dossier s'il n'existe pas
    accuracy_path.write_text(json.dumps(metrics, indent=4))

    print(f"✅ Accuracy calculée : {accuracy:.4f}")
    print(f"📂 Sauvegardée dans {accuracy_path}")


if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent.parent
    main(repo_path)
