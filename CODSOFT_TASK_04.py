# importing the required libraries for the task
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk

#download the necessary nltk files
nltk.download('stopwords')
nltk.download('punkt')

#loading the dataset 
spam_dataset = pd.read_csv(r'D:\my software projects\CODSOFT_TASK_04\archive (6)\spam.csv', encoding='Latin1')

#removing the null values from the dataset
spam_dataset.dropna(axis=1, inplace=True)

#renaming the columns v1 ---> target , v2 ---> text
spam_dataset.rename(columns={'v1':'target', 'v2':'text'}, inplace=True)

#converting target variables to binary digits 0 and 1
spam_dataset['target']=spam_dataset['target'].map({'ham':0,'spam':1})

#creating a function to preprocess the text

def text_preprocessing(text):
    #convert all text to lower case
    text = text.lower()
    #remove punctuation and other marks and symbols
    text = re.sub(r'\W+',' ',text)
    #tokenize the text
    tokens = word_tokenize(text)
    #removing stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    #stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

spam_dataset['text'] = spam_dataset['text'].apply(text_preprocessing)

#feature extraction using TF-IDF technique
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(spam_dataset['text'])
Y = spam_dataset['target']

#splitting the train and test data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)


#training the log reg model 
model = LogisticRegression()
model.fit(X_train,Y_train)

#evaluating the model 
y_prediction = model.predict(X_test)

accuracy = accuracy_score(Y_test,y_prediction)
precision = precision_score(Y_test,y_prediction)
recall = recall_score(Y_test,y_prediction)
f1 = f1_score(Y_test,y_prediction)

print(f"accuracy : {accuracy}")
print(f"precision : {precision}")
print(f"recall: {recall}")
print(f"F1 score : {f1}")



#creating a dataframe to save the predicted results
results_dataframe = pd.DataFrame({
    'actual' : Y_test,
    'Predicted':y_prediction 
})

#saving the results in a csv file at a given file location
results_dataframe.to_csv(r'D:\my software projects\CODSOFT_TASK_04\archive (6)\classification_results.csv',index=False)