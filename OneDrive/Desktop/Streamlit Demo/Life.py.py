import streamlit as st
st.write("Hii")
st.subheader("Hello")
st.selectbox("Which language do you like?", ["Me" , "Us" , "You"])
st.checkbox("In Love")
st.slider("How much do you love me?" , 0, 100)
st.select_slider("Select this for no reason", ["Best" , "Average", "Worst"])
st.progress(46)
st.button("Roll Number 46")

# Sidebar
st.sidebar.title("Into You")
st.sidebar.selectbox("Which language do you like?", ["Telugu" , "English" , "Hindi"])
st.sidebar.markdown("No Thankyou")