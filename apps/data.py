import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets

def assign_sentiment(rating):
    if rating == 5 or rating == 4:
        return 1
    else:
        return 0

def color_negative_red(row):
    rating = row['Sentiment'] 
    if row['rating'] == 1:
        color = 'red'
    elif row['rating'] == 2:
        color = 'blue'
    elif row['rating'] == 3:
        color = 'green'
    else:
        color = 'black'
    return ['color: %s' % color] * len(row)


def app():
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
    
    enable_action = st.checkbox("Enable Action")

    if enable_action:
        st.title("Dataset Shape")  
        if st.button("Run Action", key="run_action_button0"):
              st.write(df.shape)
    
        st.title("Column names")
        if st.button("Run Action", key="run_action_button1"):
             st.write(df.columns.values)
        
        st.title("Check for null values")
        if st.button("Run Action", key="run_action_button2"):
             st.write(df.isnull().sum())

        st.title("Getting the record where verified_reviews is null")
        if st.button("Run Action", key="run_action_button3"):
             st.write(df[df['review_list'].isna() == True])

        st.write(df.dropna(inplace=True))
    
        st.title("Check for null values")
        if st.button("Run Action", key="run_action_button4"):
             st.write(df.isnull().sum())
 
        st.title("Creating a new column 'length' that will contain the length of the string in 'verified_reviews' column")
        if st.button("Run Action", key="run_action_button5"):
             df['length'] = df['review_list'].apply(len)
             st.write(df.head())
        st.title("Print the starting column")
        if st.button("Run Action", key="run_action_button6"):
             st.write(df.head())
   	
        st.title("Datatypes of the features")
        if st.button("Run Action", key="run_action_button7"):
              st.write(df.dtypes)	
    
        
        st.title("Print length of the dataset")
        if st.button("Run Action", key="run_action_button8"):
              st.write(len(df))


        st.title("Distinct values of 'rating' and its count")
        if st.button("Run Action", key="run_action_button9"):
              st.write(df['rating'].value_counts())

        st.title('Dataset Information')
        if st.button("Run Action", key="run_action_button10"):
             st.write("Number of Rows:", df.shape[0])
             st.write("Number of Rows:", df.shape[0])
             st.write("Number of Columns:", df.shape[1])
             st.write("Columns:", df.columns.tolist())
             st.write("Data Types:", df.dtypes)


        st.title(' Finding the percentage distribution of each rating - well divide the number of records for each rating by total number of records')
        if st.button("Run Action", key="run_action_button11"):
             st.write(round(df['rating'].value_counts()/df.shape[0]*100,2))

        df['Sentiment'] = df['rating'].apply(assign_sentiment)

        st.title('Sentiment value count')
        if st.button("Run Action", key="run_action_button12"):
             st.write(df['Sentiment'].value_counts())

        st.title("Finding the percentage distribution of each Sentiment - well divide the number of records for each Sentiment by total number of records")
        if st.button("Run Action", key="run_action_button13"):
            st.write(round(df['Sentiment'].value_counts()/df.shape[0]*100,2))

        st.title("Negative Review--Sentiment = 0")
        if st.button("Run Action", key="run_action_button14"):
            st.write(df[df['Sentiment'] == 0]['rating'].value_counts())

        st.title("Positive Review--Sentiment = 1")
        if st.button("Run Action", key="run_action_button15"):
           st.write(df[df['Sentiment'] == 1]['rating'].value_counts())

        st.title("Negative Review")
   
        if st.button("Run Action", key="run_action_button16"):
    
          neg_reviews = df[df['Sentiment'] == 0].head(188)
          neg_reviews_styled = neg_reviews.style.apply(color_negative_red, axis=1)
          st.write(neg_reviews_styled)

   
    

   

  
