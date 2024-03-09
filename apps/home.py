import streamlit as st
import pandas as pd
def app():
    st.title('Home')

    st.title("CSV File Uploader")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        st.write("File uploaded successfully!")
        
        df = pd.read_csv(uploaded_file)
        
        # Display the dataframe
        st.write(df)
    else:
        st.write("Upload a CSV file to get started.")

