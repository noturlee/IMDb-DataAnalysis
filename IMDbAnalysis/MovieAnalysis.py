import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

file_path = '/Users/leighchejaikarran/Downloads/IMDbAnalysis/IMDb-Movies-India.csv'
df = pd.read_csv(file_path, encoding='latin-1')  

df.drop_duplicates(inplace=True)

df.fillna({'Rating': df['Rating'].mean()}, inplace=True)

df['Total Actors'] = df[['Actor 1', 'Actor 2', 'Actor 3']].apply(lambda row: sum(row.notna()), axis=1)

X = df.drop('Rating', axis=1)
y = df['Rating']


X.replace('', np.nan, inplace=True)

missing_values_count = X.isnull().sum()
print("Missing Values:\n", missing_values_count)

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))])  

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)


models = [
    ('Linear Regression', LinearRegression()),
    ('Ridge Regression', Ridge()),
    ('Lasso Regression', Lasso()),
    ('Random Forest', RandomForestRegressor())
]

for name, model in models:
    scores = cross_val_score(model, X_train_processed, y_train, cv=5, scoring='r2')
    print(f"{name}:")
    print(f"  Cross-validated R-squared: {np.mean(scores):.3f} (+/- {np.std(scores):.3f})")

    model.fit(X_train_processed, y_train)
    y_pred = model.predict(X_test_processed)

    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)  
    r2 = r2_score(y_test, y_pred)

    print(f"  Mean Squared Error (MSE): {mse:.3f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.3f}")
    print(f"  R-squared (R2 Score): {r2:.3f}")
    print()


    plt.figure(figsize=(8, 6))
    sns.histplot(df['Rating'], bins=20, kde=True, color='blue', edgecolor='black')
    plt.title('Distribution of Movie Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.show()

    numeric_data = X_train.select_dtypes(include=['int64', 'float64'])
    corr_matrix = numeric_data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix')
    plt.show()


    plt.figure(figsize=(10, 6))
    sns.barplot(x=[name for name, _ in models], y=[np.mean(cross_val_score(model, X_train_processed, y_train, cv=5, scoring='r2')) for _, model in models], palette='viridis')
    plt.ylim(0, 1)
    plt.title('Comparison of R-squared Scores')
    plt.xlabel('Models')
    plt.ylabel('R-squared Score')
    plt.xticks(rotation=45)
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, color='blue', alpha=0.7)
    plt.title('Actual vs. Predicted Ratings')
    plt.xlabel('Actual Ratings')
    plt.ylabel('Predicted Ratings')
    plt.show()

    if isinstance(model, RandomForestRegressor):
        feature_importance = model.feature_importances_
        feature_names = preprocessor.transformers_[1][1].named_steps['encoder'].get_feature_names(input_features=categorical_features)
        all_feature_names = np.concatenate([numeric_features, feature_names])

        plt.figure(figsize=(10, 6))
        sns.barplot(x=feature_importance, y=all_feature_names, palette='viridis')
        plt.title('Feature Importance')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.show()

    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=residuals, color='blue', alpha=0.7)
    plt.title('Residual Plot')
    plt.xlabel('Predicted Ratings')
    plt.ylabel('Residuals')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.show()
