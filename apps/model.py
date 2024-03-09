import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix

def assign_sentiment(rating):
        if rating == 5 or rating == 4:
           return 1
        else:
           return 0

def app():
    st.title('Model')
    
    st.title('Dataset')
  
    st.title("CSV File Uploader")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        st.write("File uploaded successfully!")
        
        df = pd.read_csv(uploaded_file)
        
        # Display the dataframe
        st.write(df)
    else:
        st.write("Upload a CSV file to get started.")
      
    df['Sentiment'] = df['rating'].apply(assign_sentiment)
    df[df['review_list'].isna() == True]
    df.dropna(inplace=True)

    feature_columns = ['product', 'rating', 'review_list']
    X = df[feature_columns]
    Y = df['Sentiment']

    cv = CountVectorizer()
    X = cv.fit_transform(df['review_list'])

    x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size=0.4,random_state=1)

    st.title('Print the values of train x, train y, test x & y test')

    st.write("Size of x_train: ",(x_train.shape))
    st.write("Size of y_train: ",(y_train.shape))
    st.write("Size of x_test: ",(x_test.shape))
    st.write("Size of y_test: ",(y_test.shape))

   
    # Model building
    st.header('Model performance')
    
    x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size=0.4,random_state=1)

    st.title('LogisticRegression')

    logreg = LogisticRegression()
    logreg.fit(x_train,y_train)
    logreg_pred = logreg.predict(x_test)
    logreg_acc = accuracy_score(logreg_pred,y_test)
    st.write("Test Accuracy: {:.2f}%".format(logreg_acc*100))

    st.title('MultinomialNB')

    mnb = MultinomialNB()
    mnb.fit(x_train,y_train)
    mnb_pred = mnb.predict(x_test)
    mnb_acc = accuracy_score(mnb_pred,y_test)
    st.write("Test Accuracy: {:.2f}%".format(mnb_acc*100))


    st.title('RandomForestClassifier')

    rfc = RandomForestClassifier()
    rfc.fit(x_train,y_train)
    rfc_pred = rfc.predict(x_test)
    rfc_acc = accuracy_score(rfc_pred,y_test)
    st.write("Test Accuracy: {:.2f}%".format(rfc_acc*100))
    

    st.title('DecisionTreeClassifier')

    dtc = DecisionTreeClassifier()
    dtc.fit(x_train,y_train)
    dtc_pred = dtc.predict(x_test)
    dtc_acc = accuracy_score(dtc_pred,y_test)
    st.write("Test Accuracy: {:.2f}%".format(dtc_acc*100))

    
