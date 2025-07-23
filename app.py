import streamlit as st
import joblib
import random

# Load model and vectorizer
model = joblib.load("interview_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

# Interview questions
questions = [
    "Tell me about yourself.",
    "What are your strengths and weaknesses?",
    "Why should we hire you?",
    "Describe a challenge you've faced and how you handled it.",
    "Where do you see yourself in 5 years?",
    "Why do you want to work in this company?",
    "Tell me about a time you worked in a team.",
    "How do you handle pressure and deadlines?"
]

# Set page config
st.set_page_config(page_title="AI Interview Evaluator", page_icon="🤖", layout="centered")

# Header
st.title("🤖 AI Interview Evaluator")
st.markdown("Welcome to your **AI-powered mock interview**. Get a random question, write your answer, and receive intelligent feedback instantly!")

# Load session state for question
if "question" not in st.session_state:
    st.session_state.question = ""

# Generate question
if st.button("🎤 Generate Interview Question"):
    st.session_state.question = random.choice(questions)
    st.session_state.answer = ""

# Show question
if st.session_state.question:
    st.markdown("### 💬 Interview Question")
    st.info(st.session_state.question)

    # Input answer
    user_answer = st.text_area("✍️ Write your answer here", value=st.session_state.get("answer", ""), height=180)
    st.session_state.answer = user_answer

    if st.button("🧠 Evaluate My Answer"):
        if not user_answer.strip():
            st.warning("⚠️ Please write your answer before evaluation.")
        else:
            # Combine question + answer
            combined_text = st.session_state.question + " " + user_answer
            vector = vectorizer.transform([combined_text])
            score = model.predict(vector)[0]
            score = round(score)

            # Results
            st.markdown("### 📊 Evaluation Results")
            st.metric("Predicted Score", f"{score}/100")

            # Feedback
            if score >= 85:
                st.success("✅ Excellent! You communicated your thoughts clearly and confidently.")
                st.markdown("**Tip:** Keep up this level of articulation and structure!")
            elif 65 <= score < 85:
                st.info("👍 Good attempt! Your answer is solid but could be improved.")
                st.markdown("**Tip:** Try to add specific examples, metrics, or reflections.")
            else:
                st.error("⚠️ Needs Improvement.")
                st.markdown("**Tip:** Work on grammar, clarity, and completeness. Be specific.")

            # Score bar
            st.progress(min(score, 100))

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; font-size: 13px;'>Built by Abhinav Pagadala using Streamlit and Machine Learning 🤖</div>",
    unsafe_allow_html=True
)
