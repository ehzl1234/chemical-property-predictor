"""
XGBoost model for predicting Research Octane Number (RON) from GC data.
Includes SHAP explainability for model interpretation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import joblib
import os


class OctanePredictor:
    """XGBoost-based octane number predictor with explainability."""
    
    def __init__(self):
        self.model = XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        self.feature_cols = [
            'paraffins_pct', 'olefins_pct', 'naphthenes_pct', 
            'aromatics_pct', 'oxygenates_pct', 'density_gml',
            'rvp_psi', 'distillation_t50_c', 'distillation_t90_c',
            'sulfur_ppm', 'benzene_pct', 'aromatic_paraffin_ratio',
            'distillation_range'
        ]
        self.target_col = 'ron'
        self.is_fitted = False
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features for model input."""
        return df[self.feature_cols].values
    
    def fit(self, df: pd.DataFrame) -> 'OctanePredictor':
        """Train the octane prediction model."""
        X = self.prepare_features(df)
        y = df[self.target_col].values
        
        self.model.fit(X, y)
        self.is_fitted = True
        
        print(f"Model trained on {len(df)} samples")
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict octane number."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X = self.prepare_features(df)
        return self.model.predict(X)
    
    def evaluate(self, df: pd.DataFrame) -> dict:
        """Evaluate model on test data."""
        X = self.prepare_features(df)
        y_true = df[self.target_col].values
        y_pred = self.model.predict(X)
        
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'predictions': y_pred
        }
    
    def cross_validate(self, df: pd.DataFrame, cv: int = 5) -> dict:
        """Perform cross-validation."""
        X = self.prepare_features(df)
        y = df[self.target_col].values
        
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='r2')
        
        return {
            'mean_r2': scores.mean(),
            'std_r2': scores.std(),
            'all_scores': scores
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from trained model."""
        importance = self.model.feature_importances_
        return pd.DataFrame({
            'feature': self.feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def get_shap_values(self, df: pd.DataFrame):
        """Calculate SHAP values for explainability."""
        try:
            import shap
            X = self.prepare_features(df)
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X)
            return shap_values, explainer.expected_value
        except ImportError:
            print("SHAP not installed. Run: pip install shap")
            return None, None
    
    def save(self, path: str):
        """Save model to disk."""
        model_data = {
            'model': self.model,
            'feature_cols': self.feature_cols,
            'target_col': self.target_col
        }
        joblib.dump(model_data, path)
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'OctanePredictor':
        """Load model from disk."""
        model_data = joblib.load(path)
        predictor = cls()
        predictor.model = model_data['model']
        predictor.feature_cols = model_data['feature_cols']
        predictor.target_col = model_data['target_col']
        predictor.is_fitted = True
        return predictor


def main():
    print("=" * 60)
    print("CHEMICAL PROPERTY PREDICTOR - MODEL TRAINING")
    print("=" * 60)
    
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'gc_analysis.csv')
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples")
    
    # Train/test split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Train model
    predictor = OctanePredictor()
    predictor.fit(train_df)
    
    # Cross-validation
    cv_results = predictor.cross_validate(train_df)
    print(f"\nCross-Validation R²: {cv_results['mean_r2']:.4f} (+/- {cv_results['std_r2']*2:.4f})")
    
    # Evaluate on test set
    results = predictor.evaluate(test_df)
    
    print("\n" + "=" * 40)
    print("TEST SET PERFORMANCE")
    print("=" * 40)
    print(f"RMSE:     {results['rmse']:.3f} RON")
    print(f"MAE:      {results['mae']:.3f} RON")
    print(f"R²:       {results['r2']:.4f}")
    
    # Feature importance
    print("\nFeature Importance (Top 5):")
    importance = predictor.get_feature_importance()
    for _, row in importance.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")
    
    # Save model
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'octane_predictor.pkl')
    predictor.save(model_path)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
