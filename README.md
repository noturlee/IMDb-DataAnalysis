<img src="Images/banner.gif"/>

# Movie Rating Prediction Project

## Table of Contents
1. [Objective](#objective)
2. [Data Preprocessing](#data-preprocessing)
   - [Loading the Dataset](#loading-the-dataset)
   - [Handling Missing Values](#handling-missing-values)
   - [Feature Engineering](#feature-engineering)
   - [Handling Non-numeric Values](#handling-non-numeric-values)
3. [Model Building and Evaluation](#model-building-and-evaluation)
   - [Splitting the Data](#splitting-the-data)
   - [Preprocessing Pipeline](#preprocessing-pipeline)
   - [Model Selection](#model-selection)
   - [Cross-Validation](#cross-validation)
   - [Model Fitting and Testing](#model-fitting-and-testing)
4. [Visualizations](#visualizations)
   - [Distribution of Movie Ratings](#distribution-of-movie-ratings)
   - [Correlation Matrix](#correlation-matrix)
   - [Comparison of R-squared Scores](#comparison-of-r-squared-scores)
   - [Actual vs. Predicted Ratings](#actual-vs-predicted-ratings)
   - [Feature Importance](#feature-importance)
   - [Residual Plot](#residual-plot)
5. [Outputs](#outputs)
6. [Conclusion](#conclusion)

## Objective
The objective of this project is to build a model that predicts the rating of a movie based on features such as genre, director, and actors. By analyzing historical movie data, we aim to develop a regression model that accurately estimates the rating given to a movie by users or critics. This project involves data analysis, preprocessing, feature engineering, and machine learning modeling techniques to gain insights into the factors that influence movie ratings and build a reliable prediction model.

## Data Preprocessing
### Loading the Dataset
The dataset is loaded with a specified encoding to ensure proper reading of data.

### Handling Missing Values
- The 'Rating' column's missing values are filled with the mean rating.
- Missing values in other columns are imputed with the mean for numeric columns and the most frequent value for categorical columns.

### Feature Engineering
A new feature 'Total Actors' is created to capture the number of actors listed in each movie.

### Handling Non-numeric Values
Non-numeric columns with empty strings are converted to NaN for proper handling.

<img src="https://cdn.pixabay.com/animation/2023/10/10/13/26/13-26-07-593_512.gif" width="200"/>

## Model Building and Evaluation
### Splitting the Data
The dataset is split into training and testing sets.

### Preprocessing Pipeline
- Numeric features are scaled, and missing values are imputed with the mean.
- Categorical features are one-hot encoded, and missing values are imputed with the most frequent value.

### Model Selection
Four regression models are selected for evaluation:
- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor

### Cross-Validation
Each model is evaluated using 5-fold cross-validation to assess its performance on the training data.

### Model Fitting and Testing
Models are fitted on the full training set and evaluated on the test set using metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R2 Score).

## Visualizations
The following visualizations are generated to provide insights into the model's performance and data characteristics:
<p float="center">
<img width="400" alt="Screenshot 2024-06-14 at 23 26 40" src="https://github.com/noturlee/IMDb-DataAnalysis-CODSOFT/assets/100778149/5c2bb534-c756-4bf7-a527-c64c6e5e8767">
<img width="380" alt="Screenshot 2024-06-14 at 23 28 01" src="https://github.com/noturlee/IMDb-DataAnalysis-CODSOFT/assets/100778149/52190b2e-a589-42d6-b49f-684184e9a3e9">
</p>

### Distribution of Movie Ratings
A histogram showing the distribution of movie ratings in the dataset.

### Correlation Matrix
A heatmap showing the correlation between numeric features.

### Comparison of R-squared Scores
A bar plot comparing the cross-validated R-squared scores of different models.

### Actual vs. Predicted Ratings
A scatter plot showing the relationship between actual and predicted ratings for the test set.

### Feature Importance
A bar plot showing the importance of different features for the Random Forest model.

### Residual Plot
A scatter plot showing the residuals (errors) of the predicted ratings.

## Outputs
The key outputs from the model evaluation are:
- **Cross-validated R-squared**: Measures the proportion of variance in the dependent variable that is predictable from the independent variables. A higher R-squared value indicates better model performance.
- **Mean Squared Error (MSE)**: Measures the average squared difference between the predicted and actual ratings. Lower MSE indicates better model performance.
- **Root Mean Squared Error (RMSE)**: The square root of MSE, providing a measure of prediction error in the same units as the ratings.
- **R-squared (R2 Score)**: Indicates how well the model's predictions approximate the actual data points. A higher R-squared value indicates better fit.

<img width="1279" alt="Screenshot 2024-06-14 at 23 30 03" src="https://github.com/noturlee/IMDb-DataAnalysis-CODSOFT/assets/100778149/2f2112d7-2c97-476c-90fa-be549b0e66b2">


## Conclusion
The models built in this project aim to predict movie ratings based on features like genre, director, and actors. The evaluation metrics show that while the models can provide some insights, their predictive power is relatively modest (R-squared values close to zero). This suggests that while these features contribute to movie ratings, other factors not captured in this dataset may also play significant roles. Further feature engineering, data enrichment, and model tuning could improve the accuracy of these predictions.

The project demonstrates the entire process of data analysis, preprocessing, feature engineering, and machine learning modeling to answer the question of predicting movie ratings, providing a foundation for further exploration and improvement.

<img src="https://cdn.pixabay.com/animation/2023/03/24/20/52/20-52-51-802_512.gif" width="200"/>

