"""
🍽️ AI Food Recognition - Complete Redesign
Modern, clean food classification app with nutrition & recipes
"""

import streamlit as st
from PIL import Image
from datetime import datetime
import pandas as pd

from src.AI_Food_Recognition_Nutrition_Assistant.utils.streamlit_utils import (
    load_trained_model_and_config,
    preprocess_image_for_inference,
    get_top_k_predictions,
    get_food_emoji,
    get_food_description,
)

from src.AI_Food_Recognition_Nutrition_Assistant.utils.spoonacular_utils import (
    get_food_data,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="🍽️ Food AI Recognition",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# MODERN CSS - AUTO LIGHT/DARK MODE
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Modern design - works in light & dark mode */
    :root {
        color-scheme: light dark;
    }

    /* Main container */
    .main {
        padding: 2rem 1rem;
    }

    /* Headers */
    h1 {
        text-align: center;
        font-weight: 800;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #FF8C00, #FF6B35);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* Section headers */
    h2 {
        font-weight: 700;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #FF8C00;
        padding-bottom: 0.5rem;
    }

    h3 {
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }

    /* Top-1 Prediction - Large card */
    .top-1-card {
        background: linear-gradient(135deg, #FF8C00 0%, #FF6B35 100%);
        border-radius: 20px;
        padding: 2.5rem;
        text-align: center;
        color: white;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(255, 140, 0, 0.3);
        animation: slideIn 0.5s ease-out;
    }

    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .emoji-large {
        font-size: 5rem;
        margin-bottom: 1rem;
        display: block;
    }

    .food-name-large {
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0.5rem 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }

    .confidence-large {
        font-size: 2rem;
        margin: 1rem 0;
        font-weight: 700;
    }

    .description-text {
        font-size: 1.1rem;
        margin-top: 0.5rem;
        opacity: 0.95;
    }

    /* Confidence bar - improved */
    .confidence-progress {
        height: 30px;
        background: rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        overflow: hidden;
        margin: 1rem 0;
    }

    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, rgba(255, 255, 255, 0.3), rgba(255, 255, 255, 0.6));
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 15px;
        color: white;
        font-weight: 700;
        transition: width 0.5s ease;
    }

    /* Nutrition table */
    .nutrition-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1.5rem 0;
        border-radius: 10px;
        overflow: hidden;
    }

    .nutrition-table th {
        background: linear-gradient(135deg, #FF8C00, #FF6B35);
        color: white;
        padding: 1rem;
        text-align: left;
        font-weight: 700;
    }

    .nutrition-table td {
        padding: 0.8rem 1rem;
        border-bottom: 1px solid rgba(0, 0, 0, 0.1);
    }

    .nutrition-table tr:last-child td {
        border-bottom: none;
    }

    .nutrition-value {
        font-weight: 700;
        font-size: 1.1rem;
        color: #FF8C00;
    }

    /* Recipe section */
    .recipe-step {
        background: rgba(255, 140, 0, 0.1);
        border-left: 4px solid #FF8C00;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }

    .step-number {
        display: inline-block;
        background: #FF8C00;
        color: white;
        width: 35px;
        height: 35px;
        border-radius: 50%;
        text-align: center;
        line-height: 35px;
        font-weight: 700;
        margin-right: 0.5rem;
    }

    /* Ingredient list */
    .ingredient-item {
        padding: 0.5rem 0;
        display: flex;
        align-items: center;
    }

    .ingredient-item::before {
        content: "🥘";
        margin-right: 0.5rem;
    }

    /* Other predictions */
    .prediction-item {
        background: rgba(255, 140, 0, 0.05);
        border: 2px solid rgba(255, 140, 0, 0.2);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }

    .prediction-item:hover {
        border-color: #FF8C00;
        box-shadow: 0 5px 15px rgba(255, 140, 0, 0.2);
        transform: translateX(5px);
    }

    .rank-badge {
        display: inline-block;
        background: #FF8C00;
        color: white;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        text-align: center;
        line-height: 40px;
        font-weight: 700;
        margin-right: 1rem;
    }

    /* Upload area */
    .upload-area {
        border: 3px dashed #FF8C00;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: rgba(255, 140, 0, 0.05);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #FF8C00, #FF6B35) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(255, 140, 0, 0.3) !important;
    }

    /* Info boxes */
    .info-box {
        background: rgba(255, 140, 0, 0.1);
        border-left: 4px solid #FF8C00;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }

    /* Loading spinner custom */
    .stSpinner > div {
        border-color: #FF8C00 !important;
    }

    /* Columns alignment */
    .row-flex {
        display: flex;
        gap: 1rem;
        align-items: stretch;
    }

    /* Footer */
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding-top: 2rem;
        border-top: 1px solid rgba(0, 0, 0, 0.1);
        font-size: 0.9rem;
        opacity: 0.7;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 70  # If confidence < 50%, show "Cannot detect"


if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'image' not in st.session_state:
    st.session_state.image = None
if 'history' not in st.session_state:
    st.session_state.history = []


@st.cache_resource
def load_model():
    """Load model once."""
    try:
        with st.spinner("🔄 Loading AI model..."):
            model, device, class_names, config = load_trained_model_and_config()
    except Exception as e:
        st.error(
            "❌ Failed to load model. Verify local artifacts or Hugging Face settings "
            "(HF_MODEL_REPO_ID, HF_MODEL_FILENAME, HF_CLASS_NAMES_FILENAME, HF_TOKEN)."
        )
        st.exception(e)
        st.stop()
    return model, device, class_names, config


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────

# Header
st.markdown("# 🍽️ AI Food Recognition")
st.markdown("### Identify any food with advanced AI • Get nutrition & recipes")
st.markdown("---")

# Load model
model, device, class_names, config = load_model()

# Layout: Two columns
col_input, col_output = st.columns([1, 1], gap="large")

# ─────────────────────────────────────────────────────────────────────────────
# INPUT SECTION
# ─────────────────────────────────────────────────────────────────────────────

with col_input:
    st.markdown("### 📸 Upload or Capture")

    input_method = st.radio("", ["📤 Upload Image", "📹 Webcam"], label_visibility="collapsed")

    image = None

    if input_method == "📤 Upload Image":
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image)

    else:
        st.info("📷 Webcam capture - take a photo of your food")
        # Simple camera input
        picture = st.camera_input("", label_visibility="collapsed")
        if picture:
            image = Image.open(picture).convert('RGB')

    if image:
        st.session_state.image = image

        # Predict button
        if st.button("🔮 Analyze Food", width="stretch", key="predict_btn"):
            with st.spinner("🔍 Analyzing..."):
                try:
                    image_tensor = preprocess_image_for_inference(image)
                    predictions = get_top_k_predictions(
                        model, image_tensor, class_names, device, k=5
                    )
                    st.session_state.predictions = predictions

                    # Add to history
                    st.session_state.history.append({
                        'time': datetime.now().strftime("%H:%M:%S"),
                        'food': predictions[0]['class_name'],
                        'confidence': predictions[0]['confidence']
                    })

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT SECTION
# ─────────────────────────────────────────────────────────────────────────────

with col_output:
    if st.session_state.predictions:
        predictions = st.session_state.predictions
        top_pred = predictions[0]
        confidence = top_pred['confidence']

        # CHECK CONFIDENCE THRESHOLD
        if confidence < CONFIDENCE_THRESHOLD:
            st.markdown(f"""
            <div style="background: rgba(255, 100, 100, 0.1); border-left: 4px solid #FF6464; padding: 2.5rem; border-radius: 15px; text-align: center; margin: 2rem 0;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">❓</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: #FF6464;">Cannot Detect Food Clearly</div>
                <div style="margin-top: 1rem; font-size: 1.1rem; opacity: 0.8;">Confidence: {confidence:.1f}% (below {CONFIDENCE_THRESHOLD}% threshold)</div>
                <div style="margin-top: 0.5rem; opacity: 0.7; font-size: 0.95rem;">Try uploading a clearer photo</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # TOP-1 PREDICTION
            emoji = get_food_emoji(top_pred['class_name'])
            description = get_food_description(top_pred['class_name'])

            st.markdown(f"""
            <div class="top-1-card">
                <span class="emoji-large">{emoji}</span>
                <div class="food-name-large">{top_pred['class_name'].replace('_', ' ').title()}</div>
                <div class="confidence-large">{confidence:.1f}% Confidence</div>
                <div class="description-text">{description}</div>
            </div>
            """, unsafe_allow_html=True)

            # OTHER PREDICTIONS
            if len(predictions) > 1:
                with st.expander("🔍 Other Predictions"):
                    for pred in predictions[1:]:
                        emoji = get_food_emoji(pred['class_name'])
                        st.markdown(f"""
                        <div class="prediction-item">
                            <span class="rank-badge">{pred['rank']}</span>
                            <span style="font-weight: 600; font-size: 1.1rem;">{emoji} {pred['class_name'].replace('_', ' ').title()}</span>
                            <div style="text-align: right; color: #FF8C00; font-weight: 700; margin-top: 0.5rem;">{pred['confidence']:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align: center; padding: 3rem 1rem; opacity: 0.6;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📸</div>
            <p style="font-size: 1.2rem;">Upload or capture an image to get started</p>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# NUTRITION & RECIPE SECTION (Full Width)
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.predictions:
    top_pred = st.session_state.predictions[0]
    confidence = top_pred['confidence']

    # Only show nutrition/recipe if confidence is above threshold
    if confidence >= CONFIDENCE_THRESHOLD:
        st.markdown("---")
        food_name = top_pred['class_name']

        # Fetch food details
        with st.spinner("📚 Fetching nutrition & recipe..."):
            food_data = get_food_data(food_name)

        # NUTRITION TABLE
        nutrition = food_data.get("nutrition")
        if nutrition:
            st.markdown("### 📊 Nutrition Information (per 100g)")

            def _format_nutrient(nutrition_data, key, fallback_unit):
                value = nutrition_data.get(key)
                if isinstance(value, dict):
                    amount = value.get("amount", 0)
                    unit = value.get("unit") or fallback_unit
                    return amount, unit
                if isinstance(value, (int, float)):
                    return value, fallback_unit
                if isinstance(value, str):
                    return value, ""
                return 0, fallback_unit

            nutrition_rows = []
            nutrient_config = [
                ("Calories", "calories", "🔥", "kcal"),
                ("Protein", "protein", "💪", "g"),
                ("Carbs", "carbs", "🌾", "g"),
                ("Fat", "fat", "🧈", "g"),
                ("Fiber", "fiber", "🌿", "g"),
                ("Sugar", "sugar", "🍬", "g"),
            ]
            for label, key, emoji, unit in nutrient_config:
                amount, unit_out = _format_nutrient(nutrition, key, unit)
                if isinstance(amount, (int, float)):
                    value_display = f"{amount:.1f}"
                else:
                    value_display = str(amount)
                nutrition_rows.append({
                    "Nutrient": f"{emoji} {label}",
                    "Value": value_display,
                    "Unit": unit_out
                })

            # Create DataFrame for better table display
            nutrition_df = pd.DataFrame(nutrition_rows)

            st.dataframe(nutrition_df, width="stretch", column_config={})

        # RECIPE
        recipe = food_data.get("recipe")
        if recipe:
            st.markdown("### 🍳 Recipe")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("⏱️ Ready in", f"{recipe.get('ready_in_minutes', 0)} min")
            with col2:
                st.metric("👥 Servings", recipe.get('servings', 1))
            with col3:
                if recipe.get('source_url'):
                    st.markdown(f"[🔗 Full Recipe]({recipe['source_url']})")

            # Ingredients
            st.markdown("#### 🥘 Ingredients")
            ingredients = recipe.get('ingredients', [])
            if ingredients:
                for ing in ingredients:
                    if isinstance(ing, dict):
                        amount = ing.get('amount', 0)
                        unit = ing.get('unit', '')
                        name = ing.get('name', '')
                        st.markdown(f"<div class='ingredient-item'>{amount} {unit} {name}</div>", unsafe_allow_html=True)
                    elif isinstance(ing, str):
                        st.markdown(f"<div class='ingredient-item'>{ing}</div>", unsafe_allow_html=True)

            # Instructions
            st.markdown("#### 📝 Instructions")
            instructions = recipe.get('instructions', [])
            if instructions:
                for idx, step in enumerate(instructions, 1):
                    if step:
                        st.markdown(f"""
                        <div class="recipe-step">
                            <span class="step-number">{idx}</span>
                            <span>{step}</span>
                        </div>
                        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR - HISTORY
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📋 History")

    if st.session_state.history:
        for item in reversed(st.session_state.history[-10:]):
            st.markdown(f"""
            <div style="padding: 0.8rem; background: rgba(255, 140, 0, 0.1); border-radius: 8px; margin: 0.5rem 0;">
                <div style="font-size: 0.8rem; opacity: 0.7;">⏰ {item['time']}</div>
                <div style="font-weight: 600; color: #FF8C00;">{item['food'].replace('_', ' ').title()}</div>
                <div style="font-size: 0.9rem;">{item['confidence']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No history yet")

    st.markdown("---")
    st.markdown(f"**Device:** {device.upper()}")

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="footer">
    <p>🤖 Powered by ConvNeXt AI | 🍽️ Food Recognition Assistant</p>
    <p>Upload any food photo to get nutrition info, recipes & more!</p>
</div>
""", unsafe_allow_html=True)
