import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ---------------- PAGE ----------------
st.set_page_config(
    page_title="Regression Analytics Engine",
    page_icon="📊",
    layout="centered"
)

# ---------------- THEME ----------------
theme = st.sidebar.radio("🎨 Theme", ["🌞 Light", "🌙 Dark"])

if theme == "🌙 Dark":
    bg = "#0e1117"
    card = "#1a1f2b"
    text = "#ffffff"
    border = "#2a3142"
else:
    bg = "#ffffff"
    card = "#f5f7fb"
    text = "#111111"
    border = "#e5e7eb"

st.markdown(f"""
<style>
.stApp {{
    background-color: {bg};
    color: {text};
}}

h1, h2, h3 {{
    color: {text} !important;
}}

.block {{
    background-color: {card};
    padding: 18px;
    border-radius: 15px;
    border: 1px solid {border};
    margin-bottom: 15px;
}}

div[data-testid="stMetric"] {{
    background-color: {card};
    border: 1px solid {border};
    padding: 15px;
    border-radius: 12px;
}}

.stButton>button {{
    background-color: #6C63FF;
    color: white;
    border-radius: 10px;
    padding: 10px;
}}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("📊 Machine Learning Regression Explorer")
st.caption("Explainable regression-based prediction system")

# ---------------- MODE ----------------
mode = st.radio("Choose Mode:", ["🏘️ Demo Dataset", "📂 Upload Dataset"])

# =========================================================
# 🏠 DEMO MODE
# =========================================================
if mode == "🏘️ Demo Dataset":

    demo_data = {
        "Area": [600, 800, 1000, 1200, 1500, 1800, 2000, 2500, 3000],
        "BHK":  [1, 1, 2, 2, 3, 3, 3, 4, 4],
        "Bath": [1, 1, 2, 2, 3, 3, 3, 4, 4],
        "Price":[30, 40, 55, 65, 90, 110, 130, 170, 210]
    }

    df = pd.DataFrame(demo_data)

    # dataset preview
    st.markdown("<div class='block'>", unsafe_allow_html=True)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df)
    st.markdown("</div>", unsafe_allow_html=True)

    X = df[["Area", "BHK", "Bath"]]
    y = df["Price"]

    model = LinearRegression()
    model.fit(X, y)

    preds = model.predict(X)

    r2 = r2_score(y, preds)
    mae = mean_absolute_error(y, preds)

    # metrics with units
    col1, col2 = st.columns(2)
    with col1:
        st.metric("R² Score", round(r2, 3))
    with col2:
        st.metric("MAE", f"₹{round(mae, 2)} Lakhs")

    # explanation
    st.markdown("<div class='block'>", unsafe_allow_html=True)
    st.subheader("🧠 What these metrics mean")

    st.markdown(f"""
### 📊 R² Score = {round(r2,3)}
- Measures how well the model explains variation in price  
- Closer to 1.0 = better model  
- Your model explains **{round(r2*100,1)}% of price variation**

---

### 📉 MAE = ₹{round(mae,2)} Lakhs
- Average prediction error in price  
- Lower is better  
- On average, error = **₹{round(mae,2)} Lakhs**
""")

    st.markdown("</div>", unsafe_allow_html=True)

    # equation
    st.markdown("<div class='block'>", unsafe_allow_html=True)
    st.subheader("📐 Regression Equation")

    eq = "Price (₹ Lakhs) = "
    for i, col in enumerate(["Area", "BHK", "Bath"]):
        eq += f"({round(model.coef_[i], 4)} × {col}) + "
    eq += f"({round(model.intercept_, 4)})"

    st.code(eq)
    st.markdown("</div>", unsafe_allow_html=True)

    # prediction
    st.markdown("<div class='block'>", unsafe_allow_html=True)
    st.subheader("🎯 Predict Price")

    area = st.slider("Area (sqft)", 400, 4000, 1200)
    bhk = st.slider("BHK (units)", 1, 5, 2)
    bath = st.slider("Bathrooms (units)", 1, 5, 2)

    pred = model.predict([[area, bhk, bath]])[0]
    st.success(f"💰 Predicted Price: ₹{round(pred, 2)} Lakhs")
    st.markdown("</div>", unsafe_allow_html=True)

    # feature impact
    st.markdown("<div class='block'>", unsafe_allow_html=True)
    st.subheader("📊 Feature Contribution")

    features = ["Area", "BHK", "Bath"]
    values = [area, bhk, bath]

    contrib = [model.coef_[i] * values[i] for i in range(3)]

    fig, ax = plt.subplots()
    ax.bar(features, contrib, color=["green" if c > 0 else "red" for c in contrib])
    ax.axhline(0, color="black")

    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# 📂 UPLOAD MODE
# =========================================================
else:

    st.markdown("<div class='block'>", unsafe_allow_html=True)
    st.subheader("📂 Upload Dataset")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:

        df = pd.read_csv(file)
        st.dataframe(df.head())

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        target = st.selectbox("Select Target Column", numeric_cols)
        features = [c for c in numeric_cols if c != target]

        # auto unit detection
        target_unit = ""
        if "price" in target.lower():
            target_unit = "₹"
        elif "salary" in target.lower():
            target_unit = "₹"
        elif "rent" in target.lower():
            target_unit = "₹"
        elif "score" in target.lower():
            target_unit = "points"
        elif "time" in target.lower():
            target_unit = "hours"
        else:
            target_unit = "units"

        X = df[features].dropna()
        y = df[target].dropna()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)

        st.markdown("</div>", unsafe_allow_html=True)

        # performance
        st.markdown("<div class='block'>", unsafe_allow_html=True)
        st.subheader("📊 Model Performance")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("R² Score", round(r2, 3))
        with col2:
            st.metric(f"MAE ({target_unit})", round(mae, 2))

        st.markdown("</div>", unsafe_allow_html=True)

        # actual vs predicted
        st.markdown("<div class='block'>", unsafe_allow_html=True)
        st.subheader("📈 Actual vs Predicted")

        y_actual = y_test.reset_index(drop=True)
        y_pred = pd.Series(preds).reset_index(drop=True)

        x = np.arange(len(y_actual))

        fig, ax = plt.subplots(figsize=(9, 5))

        ax.scatter(x, y_actual, color="blue", label="Actual", alpha=0.7)
        ax.scatter(x, y_pred, color="orange", label="Predicted", alpha=0.7)
        ax.plot(x, y_pred, color="red", linewidth=2.5, label="Trend")

        step = max(1, len(x)//10)
        ax.set_xticks(x[::step])
        ax.set_xticklabels(x[::step])

        ax.set_xlabel("Samples")
        ax.set_ylabel(f"Value ({target_unit})")
        ax.set_title("Actual vs Predicted Comparison")
        ax.legend()
        ax.grid(alpha=0.2)

        st.pyplot(fig)

        st.markdown("</div>", unsafe_allow_html=True)

        # feature importance
        st.markdown("<div class='block'>", unsafe_allow_html=True)
        st.subheader("📊 Feature Importance (Clean View)")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(features, model.coef_, color="#6C63FF")
        ax.set_xlabel("Coefficient Value")
        ax.set_title("Feature Impact on Prediction")
        ax.grid(axis='x', alpha=0.2)

        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("</div>", unsafe_allow_html=True)

        # prediction
        st.markdown("<div class='block'>", unsafe_allow_html=True)
        st.subheader("🔮 Make Your Own Prediction")

        user_input = {}
        cols = st.columns(min(len(features), 3))

        for i, col in enumerate(features):
            min_v = float(X[col].min())
            max_v = float(X[col].max())
            mean_v = float(X[col].mean())

            with cols[i % len(cols)]:
                user_input[col] = st.slider(f"{col} ({target_unit})", min_v, max_v, mean_v)

        input_df = pd.DataFrame([user_input])
        custom_pred = model.predict(input_df)[0]

        st.success(f"💰 Predicted {target}: {round(custom_pred, 2)} {target_unit}")

        st.markdown("</div>", unsafe_allow_html=True)
