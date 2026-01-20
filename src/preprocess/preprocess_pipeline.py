from typing import List
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def build_preprocess_pipeline(
        categorical_features: List[str],
        numerical_features: List[str],
)-> ColumnTransformer:
   
    # Preprocessing for numerical features
    numerical_transformer = Pipeline(
        steps=[
            ('scaler', StandardScaler())
        ]
    )
    # Preprocessing for categorical fea
    categorical_transformer = Pipeline(
        steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
        ]
    )
    # Column-wise transformer that applies the right pipeline to each subset of column
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    return preprocessor 

