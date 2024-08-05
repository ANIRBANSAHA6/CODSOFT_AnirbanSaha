#importing the libraries required for the given task
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

#loading the dataset using pandas
churn_dataset = pd.read_csv(r'D:\my software projects\CODSOFT_TASK_03\archive (6)\Churn_Modelling.csv')

#printing first 10 rows of the dataset
#print(churn_dataset.head(10))

#printing last 10 rows of the dataset
#print(churn_dataset.tail(10))

#droping the not needed columns from the dataset 
churn_dataset_modified = churn_dataset.drop(columns=['RowNumber','CustomerId','Surname','Exited'])

#copying the exited column which is the result to another variable
churn_dataset_exited_data = churn_dataset['Exited']

print("the dataset showing : churn dataset modified")
print(churn_dataset_modified)

print("the dataset showing exited data")
print(churn_dataset_exited_data)

#define preprocessing functioning for numerical features
numeric_features_in_churn_dataset = ['CreditScore','Age','Tenure','Balance','NumOfProducts','EstimatedSalary']\

numeric_transformer = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='mean')),
    ('scaler',StandardScaler())
])

#define preprocessing for categorical features
categorical_features_in_churn_dataset = ['Geography','Gender','HasCrCard','IsActiveMember']

categorical_transformer = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('onehot',OneHotEncoder(handle_unknown='ignore'))
])

#combining the numerical features and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num',numeric_transformer,numeric_features_in_churn_dataset),
        ('cat',categorical_transformer,categorical_features_in_churn_dataset)
    ])

#split the train data and test data from the existing dataset
X_train,X_test,Y_train,Y_test = train_test_split(churn_dataset_modified, churn_dataset_exited_data, test_size=0.2, random_state=42,)

#building the models

#Logistic Regression

logistic_regression = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('classifier',LogisticRegression())
])

#train the model
logistic_regression.fit(X_train,Y_train)

#predict the model and evaluate 
y_predictions = logistic_regression.predict(X_test)
print("Logistic Regression :")
print(classification_report(Y_test,y_predictions))
print("ROC-AUC Score :",roc_auc_score(Y_test,y_predictions))

X_test_with_predictions = X_test.copy()
X_test_with_predictions['Predicted Exited'] = y_predictions

print("X_test_with_predictions: ")
print(X_test_with_predictions.head())



