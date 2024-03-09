import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns



# Function to create bar plot

def create_bar_plot(df):
    rating_counts = df['rating'].value_counts()
    fig, ax = plt.subplots()
    ax.bar(rating_counts.index, rating_counts.values)
    ax.set_xlabel('Rating')
    ax.set_ylabel('Count')
    ax.set_title('Total Counts of Each Rating')
    st.pyplot(fig)

def assign_sentiment(rating):
    if rating == 5 or rating == 4:
        return 1
    else:
        return 0

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

    st.title('Rating Counts Visualization')
    create_bar_plot(df)

    st.title('Lets plot the above values in a pie chart')

    fig = plt.figure(figsize=(7,7))
    colors = ("red","gold","yellowgreen")
    wp = {'linewidth':2,'edgecolor':'black'}
    tags = df['rating'].value_counts()
    explode = tuple(0 for _ in range(len(tags)))

    # Conditional assignment for explode
    if len(tags) >= 5:
        explode = (0.1, 0.2, 0.2, 0.3, 0.2)
    tags.plot(kind='pie',autopct='%1.1f',colors=colors, shadow=True,
         startangle=0, wedgeprops=wp, explode=explode,label='')

    st.title('Distribution of the different ratings')
    st.pyplot(fig)

   
    df['Sentiment'] = df['rating'].apply(assign_sentiment)
  
    sentiment_counts = df['Sentiment'].value_counts()
    df['Sentiment'].value_counts().plot.bar(color = 'blue')
    fig, ax = plt.subplots()

    ax.bar(sentiment_counts.index, sentiment_counts.values, color='blue')
    ax.set_title('Sentiment distribution count')
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Count')
    st.pyplot(fig)

    fig = plt.figure(figsize=(7, 7))
    colors = ('red', 'green')
    wp = {'linewidth': 1, "edgecolor": 'black'}
    tags = df['Sentiment'].value_counts() / df.shape[0]
    explode = (0.1, 0.1)
    tags.plot(kind='pie', autopct="%1.1f%%", shadow=True, colors=colors, startangle=90, wedgeprops=wp, explode=explode, label='Percentage wise distribution of feedback')
    st.pyplot(fig)


    df['review_list'].fillna('', inplace=True)
    df['length'] = df['review_list'].apply(lambda x: len(str(x)))
    fig = plt.figure(figsize=(7, 6))
    df.groupby('length')['rating'].mean().plot.hist(color='blue', bins=20)
    plt.title("Review length wise mean ratings")
    plt.xlabel('Ratings')
    plt.ylabel('Length')
    st.pyplot(fig)



    
