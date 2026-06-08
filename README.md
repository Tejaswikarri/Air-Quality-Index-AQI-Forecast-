## Project Title

Air Quality Index (AQI) Forecasting Using Machine Learning

## Description

This project uses Machine Learning algorithms to predict the Air Quality Index (AQI) based on environmental and pollutant parameters such as PM2.5, PM10, NO2, CO, SO2, temperature, and humidity. The system analyzes air quality data and forecasts AQI values to help monitor environmental conditions.

## Objective

To build a machine learning model that accurately predicts the Air Quality Index (AQI) using air pollution and weather-related features.

## Dataset

The project uses an AQI dataset containing pollutant and meteorological attributes.

Attributes:

1. PM2.5
2. PM10
3. NO2
4. CO
5. SO2
6. Temperature
7. Humidity

Target:
* Air Quality Index (AQI)

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

## Modules Used

* numpy
* pandas
* matplotlib
* seaborn
* train_test_split
* SimpleImputer
* StandardScaler
* Pipeline
* RandomForestRegressor
* GradientBoostingRegressor
* cross_val_score
* mean_absolute_error
* mean_squared_error
* r2_score
## Steps Involved

1. Load the AQI dataset.
2. Perform data preprocessing.
3. Handle missing values using SimpleImputer.
4. Standardize the features using StandardScaler.
5. Split the dataset into training and testing sets.
6. Train multiple regression models.
7. Evaluate model performance using MAE, RMSE, and R² Score.
8. Compare model performances.
9. Select the best-performing model.
10. Predict AQI values for new data samples.

## Models Used

* Random Forest Regressor
* Gradient Boosting Regressor

## Evaluation Metrics

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score
* Cross Validation Score
  ## Output

  The system predicts AQI values and displays:

* Model Performance Metrics
* Actual vs Predicted AQI Graph
* Feature Importance Visualization
* Best Model Selection
* Sample AQI Prediction

* ## Applications

* Environmental Monitoring
* Smart City Solutions
* Pollution Control Systems
* Weather and Climate Analysis
* Public Health Monitoring

* ## Conclusion
* The AQI Forecasting System successfully predicts air quality levels using Machine Learning techniques.By comparing Random Forest and Gradient Boosting models,the system identifies the best-performing model and provides reliable AQI forecasts that can support environmental monitoring and decision-making.


