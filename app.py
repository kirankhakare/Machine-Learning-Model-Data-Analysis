import streamlit as st
from multiapp import MultiApp
from apps import home, data,graph, model # import the app modules here

app = MultiApp()

st.title("Machine Learning Model For Increasing Product Sales")

st.title("---------------------------------------------")

# Add all your application here

app.add_app("Home", home.app)
app.add_app("Data", data.app)
app.add_app("Graph",graph.app)
app.add_app("Model", model.app)

# The main app
app.run()
