"""
Train Portfolio Risk Predictor Model
Custom ML model for volatility forecasting
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import mlflow
import mlflow.sklearn
from src.data_pipeline.collector import DataCollector
from src.risk_engine.garch_model import GARCHVolatilityModel

class PortfolioRiskPredictor:
    """
    Predict portfolio volatility using custom ML
    
    WHY THIS MODEL EXISTS (For Recruiters):
    - GPT-4 cannot forecast numerical time-series accurately
    - Requires domain features (volatility clustering, GARCH effects)
    - Runs locally: <100ms, $0 cost vs GPT-4: 2s, $0.03/query
    - Explainable: Feature importance shows what drives risk
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.feature_names = None
    
    def create_features(self, returns: pd.Series) -> pd.DataFrame:
        """
        Engineer time-series features
        
        Financial domain knowledge applied:
        - Volatility clustering (high vol follows high vol)
        - Mean reversion
        - Regime changes
        """
        
        features = pd.DataFrame(index=returns.index)
        
        # Rolling volatility (different windows capture different patterns)
        for window in [5, 10, 20]:
            features[f'vol_{window}d'] = returns.rolling(window).std()
            features[f'mean_{window}d'] = returns.rolling(window).mean()
        
        # Volatility of volatility (regime change detector)
        features['vol_of_vol'] = returns.rolling(20).std().rolling(5).std()
        
        # Momentum
        features['momentum_5d'] = returns.rolling(5).sum()
        
        # Extreme moves
        features['extreme_moves'] = (np.abs(returns) > returns.std() * 2).rolling(5).sum()
        
        return features.dropna()
    
    def train(self, returns: pd.Series) -> dict:
        """Train the model"""
        
        print("Creating time-series features...")
        features = self.create_features(returns)
        
        # Target: next day volatility
        target = returns.rolling(5).std().shift(-1)
        
        # Align
        data = pd.concat([features, target.rename('target')], axis=1).dropna()
        X = data.drop('target', axis=1)
        y = data['target']
        
        self.feature_names = X.columns.tolist()
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        # Scale and train
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("Training Random Forest model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_r2 = self.model.score(X_train_scaled, y_train)
        test_r2 = self.model.score(X_test_scaled, y_test)
        
        y_pred = self.model.predict(X_test_scaled)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        # Naive persistence baseline: predict tomorrow's vol = today's vol
        naive_pred = y_test.shift(1).dropna()
        y_aligned = y_test.loc[naive_pred.index]
        naive_mape = np.mean(np.abs((y_aligned - naive_pred) / y_aligned)) * 100
        naive_r2 = float(
            1 - np.sum((y_aligned - naive_pred) ** 2) /
            np.sum((y_aligned - y_aligned.mean()) ** 2)
        )
        improvement = (naive_mape - mape) / naive_mape * 100

        return {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'mape': mape,
            'n_features': len(self.feature_names),
            'naive_mape': round(naive_mape, 2),
            'naive_r2': round(naive_r2, 3),
            'improvement_over_naive_pct': round(improvement, 1),
        }
    
    def predict(self, recent_returns: pd.Series) -> dict:
        """Predict next-day volatility"""
        
        features = self.create_features(recent_returns)
        latest = features.iloc[-1:][self.feature_names]
        latest_scaled = self.scaler.transform(latest)
        
        predicted_vol = self.model.predict(latest_scaled)[0]
        
        # Confidence from tree predictions
        tree_preds = [tree.predict(latest_scaled)[0] for tree in self.model.estimators_]
        
        return {
            'predicted_volatility': predicted_vol,
            'confidence_lower': np.percentile(tree_preds, 5),
            'confidence_upper': np.percentile(tree_preds, 95)
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Feature importance for explainability"""
        
        return pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
    
    def save(self, path='models/risk_predictor.pkl'):
        """Save model"""
        Path(path).parent.mkdir(exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'features': self.feature_names
            }, f)
        print(f"✓ Model saved to {path}")


# Train the model
if __name__ == "__main__":
    print("=" * 70)
    print("TRAINING CUSTOM ML RISK PREDICTOR")
    print("=" * 70)
    
    print("\n1. Fetching historical market data...")
    collector = DataCollector()
    data = collector.get_stock_data("SPY", period="2y")
    
    if data is not None:
        returns = collector.calculate_returns(data)
        print(f"   ✓ Loaded {len(returns)} days of S&P 500 returns")
        
        print("\n2. Training model...")
        predictor = PortfolioRiskPredictor()
        metrics = predictor.train(returns)
        
        print(f"\n   ✓ Model trained!")
        print(f"   • Train R²:  {metrics['train_r2']:.3f}")
        print(f"   • Test R²:   {metrics['test_r2']:.3f}")
        print(f"   • MAPE:      {metrics['mape']:.2f}%")
        print(f"   • Features:  {metrics['n_features']}")
        print(f"\n   Baseline comparison (naive persistence):")
        print(f"   • RF MAPE:    {metrics['mape']:.2f}%")
        print(f"   • Naive MAPE: {metrics['naive_mape']:.2f}%")
        print(f"   • Naive R²:   {metrics['naive_r2']:.3f}")
        beat = metrics['improvement_over_naive_pct'] > 0
        print(f"   • Improvement over naive: {metrics['improvement_over_naive_pct']:.1f}%  "
              f"{'✓ RF beats baseline' if beat else '✗ RF does NOT beat baseline'}")
        
        print("\n3. Testing prediction...")
        prediction = predictor.predict(returns.tail(50))
        print(f"   ✓ Tomorrow's predicted volatility: {prediction['predicted_volatility']*100:.3f}%")
        print(f"   • 90% confidence: [{prediction['confidence_lower']*100:.3f}%, {prediction['confidence_upper']*100:.3f}%]")
        
        print("\n4. Top 5 most important features:")
        importance = predictor.get_feature_importance()
        for i, row in importance.head(5).iterrows():
            print(f"   {row['Feature']}: {row['Importance']:.3f}")

        predictor.save()

        # ── MLflow: log Random Forest run ─────────────────────────────────────
        print("\n5. Logging to MLflow...")
        mlflow.set_experiment("portfolio-volatility-forecasting")

        with mlflow.start_run(run_name="random_forest_baseline"):
            mlflow.log_params({
                "model":        "RandomForest",
                "n_estimators": 100,
                "max_depth":    10,
                "n_features":   metrics['n_features'],
                "ticker":       "SPY",
                "period":       "2y",
            })
            mlflow.log_metrics({
                "train_r2":                metrics['train_r2'],
                "test_r2":                 metrics['test_r2'],
                "mape":                    metrics['mape'],
                "naive_mape":              metrics['naive_mape'],
                "naive_r2":                metrics['naive_r2'],
                "improvement_over_naive":  metrics['improvement_over_naive_pct'],
            })
            mlflow.sklearn.log_model(predictor.model, "random_forest_model")
            print("   ✓ Random Forest run logged")

        # ── GARCH(1,1) ────────────────────────────────────────────────────────
        print("\n6. Training GARCH(1,1) model...")
        garch = GARCHVolatilityModel(p=1, q=1)
        g = garch.fit_and_forecast("SPY", returns)

        print(f"   ✓ GARCH trained")
        print(f"   • alpha (ARCH):      {g['alpha']:.4f}")
        print(f"   • beta  (GARCH):     {g['beta']:.4f}")
        print(f"   • persistence:       {g['persistence']:.4f}  "
              f"{'(high vol memory)' if g['persistence'] > 0.9 else '(moderate vol memory)'}")
        print(f"   • AIC:               {g['aic']:.2f}")
        print(f"   • Forecast ann. vol: {g['forecast_ann_vol']*100:.2f}%")

        garch.save()

        with mlflow.start_run(run_name="garch_1_1_student_t"):
            mlflow.log_params({
                "model":  "GARCH(1,1)",
                "p":      1,
                "q":      1,
                "dist":   "Student-t",
                "ticker": "SPY",
                "period": "2y",
            })
            mlflow.log_metrics({
                "aic":              g['aic'],
                "bic":              g['bic'],
                "alpha":            g['alpha'],
                "beta":             g['beta'],
                "persistence":      g['persistence'],
                "forecast_ann_vol": g['forecast_ann_vol'],
            })
            print("   ✓ GARCH run logged")

        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETE — 2 models logged to MLflow")
        print("=" * 70)
        print("\nTo view MLflow UI:  mlflow ui")
        print("Then open:          http://localhost:5000")