"""
Streamlit app for Chemical Property Prediction.
Interactive interface for predicting octane numbers from GC data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Octane Predictor",
    page_icon="⚗️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #3498db 0%, #9b59b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        color: white;
    }
    .grade-regular { background: #f39c12; padding: 0.5rem 1rem; border-radius: 0.5rem; }
    .grade-mid { background: #3498db; padding: 0.5rem 1rem; border-radius: 0.5rem; }
    .grade-premium { background: #27ae60; padding: 0.5rem 1rem; border-radius: 0.5rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load sample data."""
    data_path = Path(__file__).parent.parent / "data" / "gc_analysis.csv"
    return pd.read_csv(data_path)


@st.cache_resource
def load_model():
    """Load trained predictor."""
    try:
        from src.octane_predictor import OctanePredictor
        model_path = Path(__file__).parent.parent / "models" / "octane_predictor.pkl"
        return OctanePredictor.load(str(model_path))
    except:
        return None


def get_grade(ron: float) -> tuple:
    """Determine fuel grade based on RON."""
    if ron >= 91:
        return "Premium", "#27ae60"
    elif ron >= 89:
        return "Mid-Grade", "#3498db"
    else:
        return "Regular", "#f39c12"


def main():
    st.markdown('<h1 class="main-header">⚗️ Octane Number Predictor</h1>', unsafe_allow_html=True)
    st.markdown("Predict Research Octane Number (RON) from Gas Chromatography analysis data using XGBoost machine learning.")
    st.markdown("---")
    
    # Load model
    predictor = load_model()
    df = load_data()
    
    if predictor is None:
        st.error("⚠️ Model not found! Please train the model first by running:")
        st.code("python src/generate_data.py\npython src/octane_predictor.py", language="bash")
        st.stop()
    
    # Tabs for different functions
    tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Data Explorer", "📈 Model Insights"])
    
    with tab1:
        st.subheader("Enter GC Analysis Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Component Percentages (must sum to ~100%)**")
            
            paraffins = st.slider("Paraffins (%)", 20.0, 70.0, 45.0, 0.5)
            olefins = st.slider("Olefins (%)", 2.0, 25.0, 10.0, 0.5)
            naphthenes = st.slider("Naphthenes (%)", 2.0, 20.0, 10.0, 0.5)
            aromatics = st.slider("Aromatics (%)", 10.0, 45.0, 25.0, 0.5)
            oxygenates = st.slider("Oxygenates (%)", 0.0, 15.0, 6.0, 0.5)
            
            total = paraffins + olefins + naphthenes + aromatics + oxygenates
            st.info(f"Total: {total:.1f}% (should be ~100%)")
        
        with col2:
            st.markdown("**Physical Properties**")
            
            density = st.slider("Density (g/mL)", 0.68, 0.80, 0.74, 0.01)
            rvp = st.slider("Reid Vapor Pressure (psi)", 5.0, 15.0, 9.0, 0.1)
            t50 = st.slider("Distillation T50 (°C)", 70.0, 130.0, 100.0, 1.0)
            t90 = st.slider("Distillation T90 (°C)", 120.0, 200.0, 160.0, 1.0)
            sulfur = st.slider("Sulfur (ppm)", 1.0, 100.0, 15.0, 1.0)
            benzene = st.slider("Benzene (%)", 0.1, 3.0, 0.8, 0.1)
        
        # Prepare input
        input_data = pd.DataFrame([{
            'paraffins_pct': paraffins,
            'olefins_pct': olefins,
            'naphthenes_pct': naphthenes,
            'aromatics_pct': aromatics,
            'oxygenates_pct': oxygenates,
            'density_gml': density,
            'rvp_psi': rvp,
            'distillation_t50_c': t50,
            'distillation_t90_c': t90,
            'sulfur_ppm': sulfur,
            'benzene_pct': benzene,
            'aromatic_paraffin_ratio': aromatics / paraffins,
            'distillation_range': t90 - t50
        }])
        
        if st.button("🔮 Predict Octane Number", type="primary", use_container_width=True):
            prediction = predictor.predict(input_data)[0]
            grade, color = get_grade(prediction)
            
            st.markdown("---")
            
            # Result display
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>Predicted RON</h2>
                    <h1 style="font-size: 4rem; margin: 0;">{prediction:.1f}</h1>
                    <p style="font-size: 1.5rem; background: {color}; display: inline-block; padding: 0.5rem 1rem; border-radius: 0.5rem; margin-top: 1rem;">
                        {grade}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Composition pie chart
            st.markdown("---")
            st.subheader("Sample Composition")
            
            fig = px.pie(
                values=[paraffins, olefins, naphthenes, aromatics, oxygenates],
                names=['Paraffins', 'Olefins', 'Naphthenes', 'Aromatics', 'Oxygenates'],
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("📊 Training Data Explorer")
        
        # RON distribution
        fig = px.histogram(df, x='ron', nbins=30, color='grade',
                          title="RON Distribution by Grade",
                          color_discrete_map={'regular': '#f39c12', 'mid': '#3498db', 'premium': '#27ae60'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation with RON
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(df, x='aromatics_pct', y='ron', color='grade',
                           title="Aromatics vs RON",
                           trendline="ols")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(df, x='paraffins_pct', y='ron', color='grade',
                           title="Paraffins vs RON",
                           trendline="ols")
            st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        st.subheader("Sample Data")
        st.dataframe(df.head(20), use_container_width=True)
    
    with tab3:
        st.subheader("📈 Model Performance")
        
        # Feature importance
        importance = predictor.get_feature_importance()
        
        fig = px.bar(importance, x='importance', y='feature', orientation='h',
                    title="Feature Importance",
                    color='importance',
                    color_continuous_scale='Viridis')
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Prediction vs Actual
        st.subheader("Model Predictions vs Actual")
        predictions = predictor.predict(df)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['ron'], y=predictions,
            mode='markers',
            marker=dict(color=df['grade'].map({'regular': '#f39c12', 'mid': '#3498db', 'premium': '#27ae60'})),
            name='Predictions'
        ))
        fig.add_trace(go.Scatter(
            x=[df['ron'].min(), df['ron'].max()],
            y=[df['ron'].min(), df['ron'].max()],
            mode='lines',
            line=dict(dash='dash', color='red'),
            name='Perfect Prediction'
        ))
        fig.update_layout(
            title="Predicted vs Actual RON",
            xaxis_title="Actual RON",
            yaxis_title="Predicted RON"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.caption("⚗️ Chemical Property Predictor | Built with Python, XGBoost, SHAP & Streamlit")


if __name__ == "__main__":
    main()
