#importing the required libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

"""understanding the dataset"""

#loading the dataset for the task
fraud_dataset_train = pd.read_csv(r'D:\my software projects\CODSOFT_TASK_02\archive (6)\fraudTrain.csv', on_bad_lines ='skip')
fraud_dataset_test = pd.read_csv(r'D:\my software projects\CODSOFT_TASK_02\archive (6)\fraudTest.csv', on_bad_lines = 'skip')

#strip any spaces from the column name 
fraud_dataset_train.columns = fraud_dataset_train.columns.str.strip()

#ensuring that the target column has no spaces and it is present 
assert 'is_fraud' in fraud_dataset_train.columns , 'column "is_fraud" not found'

#checking the first 10 rows of the train dataset
print(fraud_dataset_train.head(10))

#checking the first 10 rows of the test dataset
print(fraud_dataset_test.head(10))

#checking for the missing value in the dataset
print(fraud_dataset_train.isnull().sum())
print('\n the null data for test')
print(fraud_dataset_test.isnull().sum())

#dropping the missing data
fraud_dataset_train_modified = fraud_dataset_train.dropna()
fraud_dataset_test_modified = fraud_dataset_test.dropna()

#rechecking the dataset null values
print(fraud_dataset_train_modified.isnull().sum())
print(fraud_dataset_test_modified.isnull().sum())

#seperating the feature X from target Y in all datasets
X_train = fraud_dataset_train_modified.drop(columns= ['trans_date_trans_time', 'trans_num','is_fraud'],axis = 1)
Y_train = fraud_dataset_train_modified['is_fraud']

X_test = fraud_dataset_test_modified.drop(columns= ['trans_date_trans_time', 'trans_num','is_fraud'])
Y_test = fraud_dataset_test_modified['is_fraud']

#convert categorical variables to dummy variable with limited cardinality
def convert_to_dummy(df, max_categories=100):
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique()> max_categories:
            df[col]=df[col].astype('category').cat.codes
        else:
            df = pd.get_dummies(df,columns=[col], drop_first=True)
    return df

#converting the categorical data to its dummy
X_train = convert_to_dummy(X_train)
X_test = convert_to_dummy(X_test)

#ensure both the train and test datasets have same dummy variable column
X_train, X_test = X_train.align(X_test,join='left',fill_value=0, axis=1)


#optimize datatypes to save memory
for col in X_train.select_dtypes(include=['Float64']).columns:
    X_train[col]= X_train[col].astype('float32')
    X_test[col]= X_test[col].astype('float32')

for col in X_train.select_dtypes(include=['int64']).columns:
    X_train[col]=X_train[col].astype('int32')
    X_test[col]= X_test[col].astype('int32')
    
#standardizing the features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.transform(X_test)

#building the models 

#Logistic Regression 
#initialize and train the model
logistic_regression = LogisticRegression(random_state=42,max_iter=1000)
logistic_regression.fit(x_train_scaled,Y_train)

#make predictions
y_pred_logistic_regression = logistic_regression.predict(x_test_scaled)

#evaluate the model
print('Logistic Regression Model')
print('accuracy score : ', accuracy_score(Y_test,y_pred_logistic_regression))
print('confusion matrix \n', confusion_matrix(Y_test,y_pred_logistic_regression))
print('classification report \n', classification_report(Y_test,y_pred_logistic_regression))


#Decision tree
#initialize and train the model
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train,Y_train)

#make predictions
y_pred_dt = dt.predict(X_test)

#evaluate the model
print('Decision tree Model')
print('accuracy score : ', accuracy_score(Y_test,y_pred_dt))
print('confusion matrix \n', confusion_matrix(Y_test,y_pred_dt))
print('classification report \n', classification_report(Y_test,y_pred_dt))


#Random Forest
#initialize and train the model
#rf = RandomForestClassifier(random_state=42, n_estimators=100)
#rf.fit(X_train,Y_train)

#make predictions
#y_pred_rf= rf.predict(X_test)

#evaluate the model
#print('Random Forest Model')
#print('accuracy score : ', accuracy_score(Y_test,y_pred_rf))
#print('confusion matrix \n', confusion_matrix(Y_test,y_pred_rf))
#print('classification report \n', classification_report(Y_test,y_pred_rf))

#function to preprocess new data
def preprocess_new_data(new_data, scaler, feature_columns):
    new_data = new_data.dropna()
    new_data = convert_to_dummy(new_data)
    new_data = new_data.reindex(columns=feature_columns, fill_value=0)
    
    for col in new_data.select_dtypes(include=['float64']).columns:
        new_data[col] = new_data[col].astype('float32')
    for col in new_data.select_dtypes(include=['int64']).columns:
        new_data[col] = new_data[col].astype('float32')
        
    new_data_scaled = scaler.transform(new_data)
    return new_data_scaled

#functions to make predictions on the new data 
def predict_fraud(new_data, model, scaler, feature_columns):
    new_data_processed = preprocess_new_data(new_data, scaler, feature_columns)
    predictions = model.predict(new_data_processed)
    return predictions

#predicting the new data
new_data = pd.read_csv(r'D:\my software projects\CODSOFT_TASK_02\archive (6)\fraudTest.csv', on_bad_lines = 'skip')
new_data_features = new_data.drop(['trans_date_trans_time', 'trans_num','is_fraud'], axis=1)
feature_columns = X_train.columns
predictions = predict_fraud(new_data_features, dt, scaler, feature_columns)

print('the predictions for the transactions "0 for legitmate" and "1 for fraudelent" is :')
print(predictions)