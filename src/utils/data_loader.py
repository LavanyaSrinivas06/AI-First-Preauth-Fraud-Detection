import pandas as pd

def load_dataset(path):
    """
    Loads a CSV dataset and splits it into:
    - full DataFrame
    - X (features only)
    - y (target column 'Class')
    """
    df = pd.read_csv(path)
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return df, X, y
