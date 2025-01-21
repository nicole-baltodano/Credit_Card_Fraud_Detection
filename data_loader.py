import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Load the dataset."""
    df = pd.read_csv(file_path)
    X = df.drop(columns=['Class'],axis=1).values
    y = df['Class'].values
    return X, y

def split_data(X, y, test_size=0.3, val_size=0.3):
    """Split the data into train, validation, and test sets."""
    # Step 1: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, stratify=y)

    # Step 2: Split the train data further into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = val_size,stratify=y_train)

    return X_train, X_val, X_test, y_train, y_val, y_test
