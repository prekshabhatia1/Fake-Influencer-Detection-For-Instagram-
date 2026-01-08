import streamlit as st
import pandas as pd
import joblib

# Load trained Isolation Forest model
model = joblib.load("isolation_forest_influencer.pkl")

st.title("Fake Influencer Detection System")

# User inputs (same units as training: MILLIONS)
followers = st.number_input("Followers (in millions)", min_value=0.0)
avg_likes = st.number_input("Average Likes (in millions)", min_value=0.0)
new_post_avg_like = st.number_input("New Post Avg Likes (in millions)", min_value=0.0)
eng_rate = st.number_input("60-Day Engagement Rate (e.g. 0.02)", min_value=0.0)

if st.button("Predict"):

    # Feature engineering (MUST match training)
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

    # Ensure feature order
    df_input = df_input[features]

    # Prediction
    prediction = model.predict(df_input)

    if prediction[0] == -1:
        st.error("🚨 Fake Influencer Detected (Anomalous Pattern)")
    else:
        st.success("✅ Genuine Influencer")

