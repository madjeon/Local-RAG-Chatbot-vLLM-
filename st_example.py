# main.py

import streamlit as st
import pandas as pd
import numpy as np
import time
import uuid
import datetime
from PIL import Image

# 1. Page configuration
st.set_page_config(
    page_title="Streamlit Practice",
    page_icon="🚀",
    layout="wide"
)

# 2. Sidebar navigation
with st.sidebar:
    st.title("Navigation")
    page = st.radio("", ["Home", "Widgets", "Media", "Data", "Dashboard", "Forms", "Settings"])

# 3. Home page with st.echo demonstration
if page == "Home":
    st.title("🚀 Streamlit Practice")
    with st.echo():
        st.write("This code block is shown because it's wrapped in st.echo()")

# 4. Widgets page
elif page == "Widgets":
    st.header("🧩 Input Widgets")
    name = st.text_input("Your name")
    bio = st.text_area("Short bio")
    age = st.number_input("Your age", min_value=0, max_value=120, value=30)
    rating = st.slider("Rate this demo", 0, 10, 5)
    color = st.selectbox("Favorite color", ["Red", "Green", "Blue"])
    hobbies = st.multiselect("Pick your hobbies", ["Reading", "Music", "Sports"], default=["Music"])
    agree = st.checkbox("I agree to the terms")
    mood = st.radio("Your mood today", ["Happy", "Sad", "Neutral"])
    event_date = st.date_input("Event date", datetime.date.today())
    event_time = st.time_input("Event time", datetime.time(12, 0))
    hex_color = st.color_picker("Pick a color", "#00f900")
    selfie = st.camera_input("Take a selfie")
    uploaded_file = st.file_uploader("Upload a file")
    if uploaded_file:
        st.success("File uploaded!")

# 5. Media page
elif page == "Media":
    st.header("📷 Media Display")
    st.image("https://file2.nocutnews.co.kr/newsroom/image/2025/07/24/202507240924307464_0.jpg", caption="Sample image")
    st.audio("https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3")
    st.video("https://www.youtube.com/watch?v=2Vv-BfVoq4g")

# 6. Data visualization page
elif page == "Data":
    st.header("📊 Data Visualization")
    df = pd.DataFrame(np.random.randn(100, 3), columns=["A", "B", "C"])
    st.dataframe(df)
    st.table(df.head(5))
    st.line_chart(df)
    st.area_chart(df)
    st.bar_chart(df)
    map_data = pd.DataFrame(
        np.random.randn(100, 2) / [50, 50] + [37.76, -122.4],
        columns=["lat", "lon"]
    )
    st.map(map_data)

# 7. Dashboard page with metrics and progress
elif page == "Dashboard":
    st.header("📈 Dashboard")
    c1, c2, c3 = st.columns(3)
    c1.metric("Temperature", "70 °F", "+3 °F")
    c2.metric("Wind", "9 mph", "-8%")
    c3.metric("Humidity", "86%", "0%")
    with st.spinner("Processing..."):
        time.sleep(1)
    st.success("Done!")
    prog = st.progress(0)
    for i in range(100):
        prog.progress(i + 1)
        time.sleep(0.01)

# 8. Forms page for grouped inputs
elif page == "Forms":
    st.header("📝 Forms Example")
    with st.form("feedback_form"):
        fname = st.text_input("First name")
        feedback = st.text_area("Feedback")
        subscribe = st.checkbox("Subscribe to newsletter")
        submitted = st.form_submit_button("Submit")
    if submitted:
        st.write(f"Thank you, {fname}!")
        st.write("You said:", feedback)
        if subscribe:
            st.write("Subscribed!")

# 9. Settings & utilities page
elif page == "Settings":
    st.header("⚙️ Settings & Utilities")
    if "user_id" not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
    st.write("Your session ID:", st.session_state.user_id)

    @st.cache_data
    def load_data():
        return pd.DataFrame({"x": range(10), "y": np.random.randn(10)})

    cache_df = load_data()
    st.write("Cached data sample:", cache_df)
