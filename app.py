import streamlit as st
import pandas as pd
import joblib

# Load trained Isolation Forest model
model = joblib.load("isolation_forest_influencer.pkl")

st.title("Fake Influencer Detection System")

# User inputs (same units as training)
followers = st.number_input("Followers (in K)", min_value=0.0)
avg_likes = st.number_input("Average Likes (in K)", min_value=0.0)
new_post_avg_like = st.number_input("New Post Avg Likes (in K)", min_value=0.0)
eng_rate = st.number_input(
    "60-Day Engagement Rate (e.g. 0.02)",
    min_value=0.0,
    max_value=1.0,
    step=0.001,
    format="%.3f"
)

if st.button("Predict"):

    # Feature engineering (must match training)
    like_follower_ratio = avg_likes / (followers + 1)
    engagement_per_follower = eng_rate / (followers + 1)

    features = [
        "followers",
        "avg_likes",
        "new_post_avg_like",
        "like_follower_ratio",
        "engagement_per_follower"
    ]

    df_input = pd.DataFrame([{
        "followers": followers,
        "avg_likes": avg_likes,
        "new_post_avg_like": new_post_avg_like,
        "like_follower_ratio": like_follower_ratio,
        "engagement_per_follower": engagement_per_follower
    }])

    df_input = df_input[features]

    prediction = model.predict(df_input)

    if prediction[0] == -1:
        st.error( "Fake Influencer Detected (Anomalous Pattern)")
    else:
        st.success("Genuine Influencer")
