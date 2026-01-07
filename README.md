# ⚗️ Chemical Property Predictor

Predict Research Octane Number (RON) from Gas Chromatography analysis data using XGBoost machine learning with SHAP explainability.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)
![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-purple.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)

## 🎯 Problem Statement

In petroleum refining, predicting fuel properties like **octane number** from laboratory analysis is critical for:

- **Quality Control** - Ensuring fuel meets grade specifications
- **Blending Optimization** - Minimizing costs while meeting targets
- **Process Control** - Adjusting refinery operations in real-time

This project demonstrates using **XGBoost regression** to predict Research Octane Number (RON) from Gas Chromatography (GC) component analysis.

## 📊 Features

- **Synthetic GC data** simulating real petroleum analysis
- **XGBoost regression** with hyperparameter tuning
- **SHAP explainability** for model interpretation
- **Interactive Streamlit app** for predictions

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate training data
python src/generate_data.py

# Train prediction model
python src/octane_predictor.py

# Launch prediction app
streamlit run app/streamlit_app.py
```

## 📁 Project Structure

```
chemical-property-predictor/
├── data/
│   └── gc_analysis.csv         # GC analysis data
├── models/
│   └── octane_predictor.pkl    # Trained XGBoost model
├── src/
│   ├── generate_data.py        # Data generation
│   └── octane_predictor.py     # ML pipeline
├── app/
│   └── streamlit_app.py        # Prediction interface
└── requirements.txt
```

## 🔬 Technical Details

### Input Features (GC Analysis)
| Feature | Description | Typical Range |
|---------|-------------|---------------|
| Paraffins | Saturated hydrocarbons | 35-60% |
| Olefins | Unsaturated hydrocarbons | 5-20% |
| Naphthenes | Cyclic hydrocarbons | 5-15% |
| Aromatics | Benzene ring compounds | 15-40% |
| Oxygenates | MTBE, Ethanol | 0-12% |
| Density | Sample density | 0.70-0.78 g/mL |
| RVP | Reid Vapor Pressure | 6-12 psi |
| Distillation | T50, T90 temperatures | Various |

### Target Variable
- **RON (Research Octane Number)**: 84-100

### Model Performance
- **R²**: ~0.95
- **RMSE**: ~0.8 RON
- **MAE**: ~0.6 RON

## 💡 Domain Expertise

This project applies 10+ years of experience in **petroleum quality testing** including:
- Gas Chromatography (GC) analysis
- ASTM/ISO testing standards
- Product specification compliance

## 👤 Author

**Firdaus** - Senior QC Lab Supervisor | Data Analyst

- GitHub: [@ehzl1234](https://github.com/ehzl1234)
- Portfolio: [ehzl1234.github.io](https://ehzl1234.github.io)
