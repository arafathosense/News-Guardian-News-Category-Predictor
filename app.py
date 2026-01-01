import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import plotly.express as px
import os
import time

# --- Configuration ---
st.set_page_config(
    page_title="News Guardian | AI Category Predictor",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Modern UI and Animations ---
st.markdown("""
<style>
    /* Global Styling */
    .main {
        background-color: #f8f9fa;
        color: #333;
    }
    
    /* Header with Animation */
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    h1 {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #2c3e50;
        font-weight: 700;
        text-align: center;
        margin-bottom: 20px;
        animation: fadeInDown 0.8s ease-out;
    }
    
    .subtitle {
        text-align: center;
        color: #7f8c8d;
        font-size: 1.2rem;
        margin-bottom: 40px;
        animation: fadeInDown 1s ease-out;
    }

    /* Text Area */
    .stTextArea textarea {
        border-radius: 12px;
        border: 1px solid #ced4da;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);
        transition: border-color 0.3s ease;
    }
    .stTextArea textarea:focus {
        border-color: #4b6cb7;
        box-shadow: 0 0 0 3px rgba(75, 108, 183, 0.2);
    }
    
    /* Button */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.7rem 2.5rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1.1rem;
        letter-spacing: 0.5px;
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        width: 100%;
        margin-top: 10px;
    }
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
        color: #fff;
    }
    
    /* Result Cards */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .metric-card {
        background: white;
        padding: 25px;
        border-radius: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.08);
        text-align: center;
        border-bottom: 5px solid #764ba2;
        margin-bottom: 20px;
        transition: transform 0.3s ease;
        animation: fadeInUp 0.6s ease-out both;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .category-tag {
        font-size: 1.4rem;
        font-weight: bold;
        color: #764ba2;
        margin-top: 10px;
    }
    
    .confidence-val {
        font-size: 2.5rem;
        font-weight: 800;
        color: #2c3e50;
        background: -webkit-linear-gradient(#2c3e50, #4b6cb7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .model-name {
        font-size: 0.95rem;
        color: #95a5a6;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }

    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---

@st.cache_resource
def load_models():
    """Load models and artifacts with error handling."""
    artifacts = {}
    try:
        artifacts['tfidf'] = joblib.load('tfidf_vectorizer.pkl')
        artifacts['le_ml'] = joblib.load('label_encoder_ml.pkl')
        artifacts['lr'] = joblib.load('LogisticRegression_model.pkl')
        artifacts['rf'] = joblib.load('RandomForest_model.pkl')
        artifacts['svm'] = joblib.load('SVM_model.pkl')
    except Exception as e:
        st.error(f"Error loading ML models: {e}. Please ensure models are generated.")
        return None

    # Try loading Deep Learning model separately
    try:
        from tensorflow.keras.models import load_model
        import pickle
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        
        artifacts['lstm'] = load_model('LSTM_model.h5')
        try:
             # Try first looking for tokenizer.pkl (from original script) or pickle file
            with open('tokenizer.pkl', 'rb') as handle:
                artifacts['tokenizer'] = pickle.load(handle)
        except Exception:
             # Just in case it was saved differently
             pass
        
        artifacts['has_dl'] = True
    except Exception as e:
        # st.warning(f"Deep Learning model could not be loaded (TensorFlow missing or file not found). Skipping.")
        artifacts['has_dl'] = False
        
    return artifacts

def clean_text_dl(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_ml(model, text, vectorizer, label_encoder):
    clean = clean_text_dl(text)
    vec = vectorizer.transform([clean])
    # Get probabilities
    probs = model.predict_proba(vec)[0]
    # Map to classes
    class_probs = {cls: prob for cls, prob in zip(label_encoder.classes_, probs)}
    # Top prediction
    pred_idx = model.predict(vec)[0]
    pred_label = label_encoder.inverse_transform([pred_idx])[0]
    return pred_label, class_probs

def predict_dl(model, text, tokenizer, label_encoder):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    clean = clean_text_dl(text)
    seq = tokenizer.texts_to_sequences([clean])
    padded = pad_sequences(seq, maxlen=200, padding='post', truncating='post')
    
    probs = model.predict(padded)[0]
    class_probs = {cls: prob for cls, prob in zip(label_encoder.classes_, probs)}
    
    pred_idx = np.argmax(probs)
    pred_label = label_encoder.classes_[pred_idx]
    
    return pred_label, class_probs

# --- Main App ---

def main():
    # Hero Section
    st.markdown("<h1>📰 News Guardian</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>✨ Intelligent News Classification powered by State-of-the-Art AI ✨</p>", unsafe_allow_html=True)
    
    # Project Description Expander
    with st.expander("ℹ️  **About this Project** (Click to Expand)", expanded=False):
        st.markdown("""
        ### 🚀 Project Overview
        **News Guardian** is an advanced NLP (Natural Language Processing) application designed to automatically categorize news articles into one of five major topics:
        
        *   🏢 **Business**
        *   💻 **Tech**
        *   ⚖️ **Politics**
        *   ⚽ **Sport**
        *   🎬 **Entertainment**
        
        ### 🧠 How it Works
        We utilize a powerful **Ensemble** of Machine Learning and Deep Learning models to analyze the semantic patterns in your text. By training on thousands of **BBC News** articles, our models have learned to distinguish even subtle differences between categories.
        
        ### 🛠️ Models Under the Hood
        1.  **Logistic Regression**: Excellent baseline for text classification.
        2.  **Random Forest**: Captures non-linear relationships and interactions.
        3.  **SVM (Support Vector Machine)**: Optimizes the boundary between different categories.
        4.  **LSTM (Deep Learning)**: A Recurrent Neural Network that understands the *sequence* and *context* of words.
        """)

    st.markdown("---")
    
    col1, col2 = st.columns([1.8, 1.2])
    
    with col1:
        st.markdown("### ✍️ Input Article")
        st.markdown("Paste the headline or body of the news article below:")
        article_text = st.text_area("", height=250, 
                                   placeholder="e.g., 'Apple announces new iPhone with revolutionary AI features...'")
        
        # Center the button or make it full width
        analyze_btn = st.button("🕵️ Analyze & Predict Category")

    with col2:
        # Using a container for visual balance
        st.markdown("### 🤖 Model Status")
        artifacts = load_models()
        if artifacts:
            st.success(f"✅ **{3 + (1 if artifacts.get('has_dl') else 0)} Active Models** Ready")
            st.markdown("""
            *   🟢 Logistic Regression
            *   🟢 Random Forest
            *   🟢 SVM
            *   {} LSTM (Deep Learning)
            """.format("🟢" if artifacts.get('has_dl') else "⚪"))
        else:
            st.error("❌ Models not loaded.")
        
        st.info("💡 **Tip:** For best results, paste at least 1-2 sentences of text.")

    if analyze_btn and article_text and artifacts:
        
        # Simulated Progress for UX
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for percent in range(0, 101, 20):
            time.sleep(0.05) 
            progress_bar.progress(percent)
            status_text.text(f"Processing... {percent}%")
            
        status_text.empty()
        progress_bar.empty()
        
        # Actual Analysis
        with st.spinner("🧠 Neural networks are thinking..."):
            results = []
            
            # ML Predictions
            for name, model_key in [("Logistic Regression", 'lr'), ("Random Forest", 'rf'), ("SVM", 'svm')]:
                if model_key in artifacts:
                    pred_label, probs = predict_ml(artifacts[model_key], article_text, artifacts['tfidf'], artifacts['le_ml'])
                    results.append({
                        "Model": name,
                        "Prediction": pred_label,
                        "Confidence": probs[pred_label],
                        "Probabilities": probs
                    })
            
            # DL Prediction
            if artifacts.get('has_dl') and 'lstm' in artifacts:
                try:
                    pred_label, probs = predict_dl(artifacts['lstm'], article_text, artifacts['tokenizer'], artifacts['le_ml'])
                    results.append({
                        "Model": "LSTM (Deep Learning)",
                        "Prediction": pred_label,
                        "Confidence": probs[pred_label],
                        "Probabilities": probs
                    })
                except Exception as e:
                    print(f"DL Prediction error: {e}")

            # --- Display Results ---
            st.balloons() # 🎉 Celebrate success!
            
            st.markdown("---")
            st.markdown("### 🎯 Prediction Results")
            
            # Create a clear summary with animation delays via indices
            summary_cols = st.columns(len(results))
            
            for i, res in enumerate(results):
                with summary_cols[i]:
                    # Determine emoji based on category
                    cat_lower = res['Prediction'].lower()
                    emoji = "📂"
                    if "sport" in cat_lower: emoji = "⚽"
                    elif "tech" in cat_lower: emoji = "💻"
                    elif "politic" in cat_lower: emoji = "⚖️"
                    elif "business" in cat_lower: emoji = "🏢"
                    elif "entertainment" in cat_lower: emoji = "🎬"
                    
                    st.markdown(f"""
                    <div class="metric-card" style="animation-delay: {i * 0.1}s;">
                        <div class="model-name">{res['Model']}</div>
                        <div class="category-tag">{emoji} {res['Prediction']}</div>
                        <div class="confidence-val">{res['Confidence']:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # --- Visualization ---
            st.markdown("### � Confidence Analysis")
            
            col_chart, col_details = st.columns([2, 1])
            
            with col_chart:
                chart_data = []
                for res in results:
                    for category, prob in res['Probabilities'].items():
                        chart_data.append({
                            "Model": res['Model'],
                            "Category": category,
                            "Probability": prob
                        })
                
                df_chart = pd.DataFrame(chart_data)
                
                fig = px.bar(df_chart, x="Model", y="Probability", color="Category", 
                             title="Prediction Probability Distribution",
                             barmode='group',
                             color_discrete_sequence=px.colors.qualitative.Bold)
                
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Helvetica Neue"),
                    xaxis_title="",
                    yaxis_title="Probability",
                    legend_title_text="Category",
                    margin=dict(t=30, l=0, r=0, b=0)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col_details:
                st.markdown("#### 🧐 What does this mean?")
                st.write("Higher bars indicate stronger confidence from that specific model. When all models agree (bars are high for the same category), the prediction is highly reliable.")
                st.info("The **LSTM** model is generally better at understanding context, while **SVM** and **Logistic Regression** are excellent at spotting keyword patterns.")

    elif analyze_btn and not article_text:
        st.warning("⚠️ Whoops! Please enter some text to analyze first.")

if __name__ == "__main__":
    main()
