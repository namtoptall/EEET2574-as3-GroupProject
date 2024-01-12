# Import necessary libraries
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin

# Add the parent directory to the sys.path
if os.path.abspath('../') not in sys.path:
    sys.path.insert(0, os.path.abspath('../'))

# Function to read data
def read_data():
    economic = pd.read_excel('../data/processed/economic.xlsx') 
    forest_carbon = pd.read_excel('../data/processed/deforest_carbon.xlsx')
    Vie_weather = pd.read_excel('../data/processed/Vie_weather.xlsx')

    # Convert to upper case
    economic['province'] = economic['province'].str.upper()
    forest_carbon['province'] = forest_carbon['province'].str.upper()
    Vie_weather['province'] = Vie_weather['province'].str.upper()

    return economic, forest_carbon, Vie_weather

# Function to concatenate data
def concatenate_data(economic, forest_carbon, Vie_weather):
    vndata = pd.concat([economic.set_index('province'), forest_carbon.set_index('province'), Vie_weather.set_index('province')], axis=1)
    vndata.reset_index(inplace=True)
    vndata = vndata.loc[:, ~vndata.columns.duplicated(keep='first')]
    
    return vndata

# Custom Transformer class for cleaning data
class CleanData(TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Rename columns
        X.rename(columns={'AREA OF LAND (Thous. ha)': 'AREA OF LAND', 'POPULATION (Thous. pers.)': 'Population',
                           'POPULATION DENSITY (Person/km2)': 'population density',
                           'At current prices (Bill. dongs)': 'GROSS REGIONAL DOMESTIC PRODUCT',
                           'State budget revenue (Bill. dongs)': 'STATE BUDGET REVENUE',
                           'State budget expenditure (Bill. dongs)': 'STATE BUDGET EXPENDITURE',
                           'Investment at current prices (Bill. dongs)': 'INVESTMENT AT CURRENT PRICES',
                           'Number of farms': 'NUMBER OF FARM', 'Planted area of cereals (Thous. ha)': 'PLANTED AREA OF CEREALS',
                           'Production of fishery (Ton)': 'PRODUCTION OF FISHERY',
                           'Index of industrial production (%)': 'INDEX OF INDUSTRIAL PRODUCTION',
                           'Retail sales of goods at current prices (Bill. dongs)': 'RETAIL SALES OF GOODS',
                           'Number of schools (School)': 'NUMBER OF SCHOOLS',
                           'Number of medical establishments (Esta.)': 'NUMBER OF MEDICAL ESTABLISHMENTS',
                           'carbon_gross_emissions': 'CARBON GROSS EMISSIONS', 'tc_loss_ha': 'TROPICAL FOREST LOSS',
                           'FEELS_LIKE': 'FEELS LIKE', 'TEMP_MIN': 'TEMP MIN', 'TEMP_MAX': 'TEMP MAX'}, inplace=True)
        
        X.columns = X.columns.str.upper()

        # Fill NaN values
        columns_to_fill = ['AREA OF LAND', 'POPULATION DENSITY', 'GROSS REGIONAL DOMESTIC PRODUCT',
                           'STATE BUDGET REVENUE', 'STATE BUDGET EXPENDITURE', 'NUMBER OF FARM',
                           'RETAIL SALES OF GOODS', 'NUMBER OF SCHOOLS']

        mean_values_by_province = X.groupby('PROVINCE')[columns_to_fill].mean()

        for col in columns_to_fill:
            X[col] = X.apply(lambda row: round(row[col], 2) if pd.notnull(row[col]) else round(mean_values_by_province.loc[row['PROVINCE'], col], 2), axis=1)
        
        X['RETAIL SALES OF GOODS'].fillna(X['RETAIL SALES OF GOODS'].mean(), inplace=True)

        return X

# Custom Transformer class for transforming data
class TransformData(TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
        numerical_features = numerical_features[numerical_features != 'year']
        vn_feature = X[numerical_features]

        # Handle outliers using IQR method
        def handle_outliers_iqr(column):
            Q1 = column.quantile(0.25)
            Q3 = column.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            column_copy = column.copy()
            column_copy.loc[column < lower_bound] = lower_bound
            column_copy.loc[column > upper_bound] = upper_bound
            return column_copy

        vn_features_copy = vn_feature.copy()

        for col in vn_features_copy.columns:
            vn_features_copy[col] = handle_outliers_iqr(vn_features_copy[col])

        vn_feature = vn_features_copy

        return vn_feature

# Custom Transformer class for splitting data
class SplitData(TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_train, X_test, Y_train, Y_test = train_test_split(X.drop(['TROPICAL FOREST LOSS', 'CARBON GROSS EMISSIONS'], axis=1),
                                                            X['TROPICAL FOREST LOSS'],
                                                            test_size=0.2, random_state=42)

        return X_train, X_test, Y_train, Y_test

# Custom Transformer class for creating a preprocessing pipeline
class CreatePipeline(TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return Pipeline([('scaler', StandardScaler())])

# Custom Transformer class for training and evaluating a model
class TrainEvaluateModel(TransformerMixin):
    def __init__(self, model_type):
        self.model_type = model_type

    def fit(self, X, y=None):
        if self.model_type == 'DecisionTreeRegressor':
            self.model = DecisionTreeRegressor(random_state=42)
        elif self.model_type == 'LogisticRegression':
            self.model = LogisticRegression(random_state=42)
        elif self.model_type == 'SVM':
            self.model = SVC(kernel='linear', random_state=42)
        elif self.model_type == 'DecisionTreeClassifier':
            self.model = DecisionTreeClassifier(random_state=42)
        elif self.model_type == 'RandomForestClassifier':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        X_train, X_test, Y_train, Y_test = X
        self.model.fit(X_train, Y_train)
        self.Y_pred = self.model.predict(X_test)
        return self

    def transform(self, X):
        print(f"{self.model_type}:\n")

        if 'Regressor' in self.model_type:
            # For regression
            rmse = np.sqrt(mean_squared_error(X[3], self.Y_pred))
            mae = mean_absolute_error(X[3], self.Y_pred)
            r2 = r2_score(X[3], self.Y_pred)

            print(f'RMSE: {rmse:.4f}')
            print(f'MAE: {mae:.4f}')
            print(f'R2: {r2:.4f}')
        elif 'Classifier' in self.model_type:
            # For classification
            accuracy = accuracy_score(X[3], self.Y_pred)
            report = classification_report(X[3], self.Y_pred)

            print(f'Accuracy: {accuracy:.4f}\nClassification Report:\n{report}')

        print('=' * 40 + '\n')
        return self

# Execute the extended pipeline
if __name__ == "__main__":
    #read
    economic, forest_carbon, Vie_weather = read_data()
    # Concatenate data
    vndata = concatenate_data(economic, forest_carbon, Vie_weather)

    comprehensive_pipeline = Pipeline([
        ('clean_data', CleanData()),
        ('transform_data', TransformData()),
        ('split_data', SplitData()),
        ('create_pipeline', CreatePipeline()),
        ('train_evaluate_decision_tree', TrainEvaluateModel('DecisionTreeRegressor')),
        ('train_evaluate_logistic_regression', TrainEvaluateModel('LogisticRegression')),
        ('train_evaluate_svm', TrainEvaluateModel('SVM')),
        ('train_evaluate_decision_tree_classifier', TrainEvaluateModel('DecisionTreeClassifier')),
        ('train_evaluate_random_forest_classifier', TrainEvaluateModel('RandomForestClassifier'))
    ])

    comprehensive_pipeline.fit_transform(vndata)
