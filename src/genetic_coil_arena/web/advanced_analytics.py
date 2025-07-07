"""
Advanced Analytics & AI Integration for Genetic Coil Arena
Phase 5: ML-powered insights, predictive analytics, and intelligent optimization
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class AnalyticsInsight:
    """Individual analytics insight."""
    type: str
    title: str
    description: str
    confidence: float
    importance: float
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    actionable: bool = True
    category: str = "general"


@dataclass
class ExperimentPrediction:
    """Prediction about experiment outcome."""
    experiment_id: str
    predicted_best_fitness: float
    predicted_convergence_generation: int
    confidence_interval: Tuple[float, float]
    estimated_completion_time: datetime
    success_probability: float
    bottlenecks: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class PerformanceAnomaly:
    """Performance anomaly detection result."""
    experiment_id: str
    anomaly_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    detected_at: datetime
    metrics: Dict[str, float]
    suggested_actions: List[str] = field(default_factory=list)


class ExperimentDataProcessor:
    """Process and prepare experiment data for ML analysis."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=10)
        self.feature_names = []
        
    def extract_features(self, experiment_data: Dict[str, Any]) -> np.ndarray:
        """Extract numerical features from experiment data."""
        features = []
        
        # Basic experiment parameters
        config = experiment_data.get('config', {})
        features.extend([
            config.get('population_size', 100),
            config.get('generations', 100),
            config.get('mutation_rate', 0.1),
            config.get('crossover_rate', 0.8),
            config.get('elite_size', 5),
            config.get('grid_size', 10),
            config.get('max_steps', 500)
        ])
        
        # Performance metrics
        stats = experiment_data.get('stats', {})
        features.extend([
            stats.get('best_fitness', 0),
            stats.get('average_fitness', 0),
            stats.get('diversity_score', 0),
            stats.get('convergence_rate', 0),
            len(stats.get('generation_history', [])),
            stats.get('stagnation_count', 0)
        ])
        
        # Time-based features
        start_time = experiment_data.get('start_time')
        if start_time:
            if isinstance(start_time, str):
                start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            duration = (datetime.now() - start_time).total_seconds()
            features.extend([
                duration,
                duration / config.get('generations', 1),  # time per generation
                stats.get('best_fitness', 0) / max(duration / 3600, 0.1)  # fitness per hour
            ])
        else:
            features.extend([0, 0, 0])
        
        # Hardware/environment features
        features.extend([
            experiment_data.get('hardware_score', 1.0),
            experiment_data.get('memory_usage', 0.5),
            experiment_data.get('cpu_usage', 0.5)
        ])
        
        self.feature_names = [
            'population_size', 'generations', 'mutation_rate', 'crossover_rate',
            'elite_size', 'grid_size', 'max_steps', 'best_fitness', 'average_fitness',
            'diversity_score', 'convergence_rate', 'generation_count', 'stagnation_count',
            'duration', 'time_per_generation', 'fitness_per_hour', 'hardware_score',
            'memory_usage', 'cpu_usage'
        ]
        
        return np.array(features, dtype=float)
    
    def prepare_dataset(self, experiments: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare dataset for machine learning."""
        features_list = []
        targets = []
        
        for exp in experiments:
            try:
                features = self.extract_features(exp)
                if len(features) == len(self.feature_names):
                    features_list.append(features)
                    # Target is the final best fitness
                    targets.append(exp.get('stats', {}).get('best_fitness', 0))
            except Exception as e:
                logger.warning(f"Error processing experiment {exp.get('id', 'unknown')}: {e}")
                continue
        
        if not features_list:
            return np.array([]), np.array([])
        
        X = np.array(features_list)
        y = np.array(targets)
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return X, y


class MLPredictor:
    """Machine learning predictor for experiment outcomes."""
    
    def __init__(self):
        self.fitness_predictor = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.convergence_predictor = RandomForestRegressor(
            n_estimators=50,
            max_depth=8,
            random_state=42,
            n_jobs=-1
        )
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_jobs=-1
        )
        self.data_processor = ExperimentDataProcessor()
        self.is_trained = False
        self.model_path = Path("models/ml_predictor.pkl")
        
    def train(self, experiments: List[Dict[str, Any]]) -> Dict[str, float]:
        """Train the ML models on historical experiment data."""
        logger.info(f"Training ML models on {len(experiments)} experiments")
        
        X, y_fitness = self.data_processor.prepare_dataset(experiments)
        
        if len(X) < 10:
            logger.warning("Insufficient data for training ML models")
            return {"error": "insufficient_data", "data_points": len(X)}
        
        # Prepare convergence targets
        y_convergence = []
        for exp in experiments:
            gen_history = exp.get('stats', {}).get('generation_history', [])
            if gen_history:
                # Find convergence point (when improvement rate drops below threshold)
                convergence_gen = len(gen_history)
                for i in range(1, len(gen_history)):
                    if i > 10:  # Look after at least 10 generations
                        recent_improvement = np.mean(np.diff(gen_history[max(0, i-5):i+1]))
                        if abs(recent_improvement) < 0.001:  # Convergence threshold
                            convergence_gen = i
                            break
                y_convergence.append(convergence_gen)
            else:
                y_convergence.append(0)
        
        y_convergence = np.array(y_convergence[:len(X)])
        
        # Train fitness predictor
        fitness_score = 0
        try:
            self.fitness_predictor.fit(X, y_fitness)
            y_pred = self.fitness_predictor.predict(X)
            fitness_score = r2_score(y_fitness, y_pred)
            logger.info(f"Fitness predictor R² score: {fitness_score:.4f}")
        except Exception as e:
            logger.error(f"Error training fitness predictor: {e}")
        
        # Train convergence predictor
        convergence_score = 0
        try:
            self.convergence_predictor.fit(X, y_convergence)
            y_conv_pred = self.convergence_predictor.predict(X)
            convergence_score = r2_score(y_convergence, y_conv_pred)
            logger.info(f"Convergence predictor R² score: {convergence_score:.4f}")
        except Exception as e:
            logger.error(f"Error training convergence predictor: {e}")
        
        # Train anomaly detector
        try:
            self.anomaly_detector.fit(X)
            logger.info("Anomaly detector trained successfully")
        except Exception as e:
            logger.error(f"Error training anomaly detector: {e}")
        
        self.is_trained = True
        self.save_model()
        
        return {
            "fitness_r2": fitness_score,
            "convergence_r2": convergence_score,
            "training_samples": len(X),
            "features": len(self.data_processor.feature_names)
        }
    
    def predict_experiment_outcome(self, experiment_config: Dict[str, Any]) -> ExperimentPrediction:
        """Predict the outcome of an experiment based on its configuration."""
        if not self.is_trained:
            logger.warning("ML predictor not trained, loading from disk or using defaults")
            self.load_model()
        
        # Create dummy experiment data for feature extraction
        dummy_exp = {
            'config': experiment_config,
            'stats': {'best_fitness': 0, 'average_fitness': 0, 'diversity_score': 0.5},
            'start_time': datetime.now(),
            'hardware_score': 1.0,
            'memory_usage': 0.5,
            'cpu_usage': 0.5
        }
        
        features = self.data_processor.extract_features(dummy_exp).reshape(1, -1)
        
        try:
            # Predict fitness
            predicted_fitness = self.fitness_predictor.predict(features)[0]
            fitness_std = np.std([tree.predict(features)[0] for tree in self.fitness_predictor.estimators_[:10]])
            confidence_interval = (
                predicted_fitness - 1.96 * fitness_std,
                predicted_fitness + 1.96 * fitness_std
            )
            
            # Predict convergence
            predicted_convergence = max(1, int(self.convergence_predictor.predict(features)[0]))
            
            # Estimate completion time
            generations = experiment_config.get('generations', 100)
            estimated_time_per_gen = 2.0  # seconds, can be learned from historical data
            estimated_duration = timedelta(seconds=generations * estimated_time_per_gen)
            estimated_completion = datetime.now() + estimated_duration
            
            # Calculate success probability (based on predicted fitness relative to typical range)
            success_probability = min(1.0, max(0.0, predicted_fitness / 10.0))  # Assuming max fitness ~10
            
            # Generate recommendations
            recommendations = self._generate_recommendations(experiment_config, predicted_fitness)
            
            return ExperimentPrediction(
                experiment_id=experiment_config.get('id', 'unknown'),
                predicted_best_fitness=predicted_fitness,
                predicted_convergence_generation=predicted_convergence,
                confidence_interval=confidence_interval,
                estimated_completion_time=estimated_completion,
                success_probability=success_probability,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            # Return default prediction
            return ExperimentPrediction(
                experiment_id=experiment_config.get('id', 'unknown'),
                predicted_best_fitness=0.0,
                predicted_convergence_generation=experiment_config.get('generations', 100),
                confidence_interval=(0.0, 1.0),
                estimated_completion_time=datetime.now() + timedelta(hours=1),
                success_probability=0.5,
                recommendations=["Consider tuning hyperparameters"]
            )
    
    def detect_anomalies(self, experiment_data: Dict[str, Any]) -> List[PerformanceAnomaly]:
        """Detect performance anomalies in experiment data."""
        anomalies = []
        
        try:
            features = self.data_processor.extract_features(experiment_data).reshape(1, -1)
            anomaly_score = self.anomaly_detector.decision_function(features)[0]
            is_anomaly = self.anomaly_detector.predict(features)[0] == -1
            
            if is_anomaly:
                severity = "high" if anomaly_score < -0.3 else "medium"
                
                anomaly = PerformanceAnomaly(
                    experiment_id=experiment_data.get('id', 'unknown'),
                    anomaly_type="performance_deviation",
                    severity=severity,
                    description=f"Unusual performance pattern detected (score: {anomaly_score:.3f})",
                    detected_at=datetime.now(),
                    metrics={"anomaly_score": float(anomaly_score)},
                    suggested_actions=[
                        "Review experiment parameters",
                        "Check for hardware issues",
                        "Verify data quality"
                    ]
                )
                anomalies.append(anomaly)
            
            # Additional rule-based anomaly detection
            stats = experiment_data.get('stats', {})
            
            # Stagnation detection
            stagnation_count = stats.get('stagnation_count', 0)
            if stagnation_count > 10:
                anomaly = PerformanceAnomaly(
                    experiment_id=experiment_data.get('id', 'unknown'),
                    anomaly_type="stagnation",
                    severity="medium",
                    description=f"Evolution stagnated for {stagnation_count} generations",
                    detected_at=datetime.now(),
                    metrics={"stagnation_count": stagnation_count},
                    suggested_actions=[
                        "Increase mutation rate",
                        "Add diversity mechanisms",
                        "Consider restart with different parameters"
                    ]
                )
                anomalies.append(anomaly)
            
            # Diversity loss detection
            diversity = stats.get('diversity_score', 1.0)
            if diversity < 0.1:
                anomaly = PerformanceAnomaly(
                    experiment_id=experiment_data.get('id', 'unknown'),
                    anomaly_type="diversity_loss",
                    severity="high",
                    description=f"Population diversity critically low ({diversity:.3f})",
                    detected_at=datetime.now(),
                    metrics={"diversity_score": diversity},
                    suggested_actions=[
                        "Implement niching or speciation",
                        "Increase population size",
                        "Add novelty search components"
                    ]
                )
                anomalies.append(anomaly)
                
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
        
        return anomalies
    
    def _generate_recommendations(self, config: Dict[str, Any], predicted_fitness: float) -> List[str]:
        """Generate optimization recommendations based on config and predictions."""
        recommendations = []
        
        pop_size = config.get('population_size', 100)
        mut_rate = config.get('mutation_rate', 0.1)
        cross_rate = config.get('crossover_rate', 0.8)
        
        # Population size recommendations
        if pop_size < 50:
            recommendations.append("Consider increasing population size for better exploration")
        elif pop_size > 500:
            recommendations.append("Large population may slow convergence - consider reducing size")
        
        # Mutation rate recommendations
        if mut_rate < 0.05:
            recommendations.append("Low mutation rate may cause premature convergence")
        elif mut_rate > 0.3:
            recommendations.append("High mutation rate may prevent convergence")
        
        # Crossover rate recommendations
        if cross_rate < 0.6:
            recommendations.append("Consider increasing crossover rate for better recombination")
        
        # Fitness-based recommendations
        if predicted_fitness < 1.0:
            recommendations.append("Consider longer training or different hyperparameters")
        
        return recommendations
    
    def save_model(self):
        """Save trained models to disk."""
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            model_data = {
                'fitness_predictor': self.fitness_predictor,
                'convergence_predictor': self.convergence_predictor,
                'anomaly_detector': self.anomaly_detector,
                'data_processor': self.data_processor,
                'is_trained': self.is_trained
            }
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f"ML models saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Error saving ML models: {e}")
    
    def load_model(self):
        """Load trained models from disk."""
        try:
            if self.model_path.exists():
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.fitness_predictor = model_data['fitness_predictor']
                self.convergence_predictor = model_data['convergence_predictor']
                self.anomaly_detector = model_data['anomaly_detector']
                self.data_processor = model_data['data_processor']
                self.is_trained = model_data['is_trained']
                
                logger.info(f"ML models loaded from {self.model_path}")
            else:
                logger.warning("No saved ML models found")
        except Exception as e:
            logger.error(f"Error loading ML models: {e}")


class AdvancedAnalytics:
    """Advanced analytics and AI integration system."""
    
    def __init__(self):
        self.ml_predictor = MLPredictor()
        self.insights_cache = {}
        self.cache_ttl = timedelta(minutes=30)
        
    def generate_insights(self, experiments: List[Dict[str, Any]]) -> List[AnalyticsInsight]:
        """Generate AI-powered insights from experiment data."""
        insights = []
        
        if not experiments:
            return insights
        
        try:
            # Simple performance analysis
            if len(experiments) >= 2:
                recent_fitness = [exp.get('stats', {}).get('best_fitness', 0) for exp in experiments[-5:]]
                avg_fitness = np.mean(recent_fitness) if recent_fitness else 0
                
                if avg_fitness > 0.8:
                    insights.append(AnalyticsInsight(
                        type="performance",
                        title="High Performance Achieved",
                        description=f"Recent experiments show excellent performance with average fitness of {avg_fitness:.3f}",
                        confidence=0.9,
                        importance=0.8,
                        category="performance",
                        data={"average_fitness": avg_fitness}
                    ))
                elif avg_fitness < 0.3:
                    insights.append(AnalyticsInsight(
                        type="performance",
                        title="Performance Improvement Needed",
                        description=f"Recent experiments show low performance. Consider parameter tuning.",
                        confidence=0.8,
                        importance=0.9,
                        category="optimization",
                        data={"average_fitness": avg_fitness}
                    ))
            
            # Parameter analysis
            mutation_rates = [exp.get('config', {}).get('mutation_rate', 0.1) for exp in experiments]
            population_sizes = [exp.get('config', {}).get('population_size', 100) for exp in experiments]
            
            if mutation_rates:
                avg_mut_rate = np.mean(mutation_rates)
                if avg_mut_rate > 0.3:
                    insights.append(AnalyticsInsight(
                        type="parameter",
                        title="High Mutation Rate Detected",
                        description="High mutation rates may prevent convergence. Consider reducing.",
                        confidence=0.7,
                        importance=0.6,
                        category="optimization",
                        data={"average_mutation_rate": avg_mut_rate}
                    ))
                elif avg_mut_rate < 0.05:
                    insights.append(AnalyticsInsight(
                        type="parameter",
                        title="Low Mutation Rate Detected",
                        description="Low mutation rates may cause premature convergence. Consider increasing.",
                        confidence=0.7,
                        importance=0.6,
                        category="optimization",
                        data={"average_mutation_rate": avg_mut_rate}
                    ))
            
            # General insights
            insights.append(AnalyticsInsight(
                type="summary",
                title="Experiment Analysis Complete",
                description=f"Analyzed {len(experiments)} experiments with {len(insights)} specific insights generated.",
                confidence=1.0,
                importance=0.5,
                category="general",
                data={"experiment_count": len(experiments)}
            ))
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            insights.append(AnalyticsInsight(
                type="error",
                title="Analysis Error",
                description="Error occurred during insight generation. Please check experiment data format.",
                confidence=0.9,
                importance=0.8,
                category="error",
                data={"error": str(e)}
            ))
        
        return sorted(insights, key=lambda x: x.importance, reverse=True)
    
    def predict_experiment_outcome(self, experiment_data: Dict[str, Any]) -> Optional[ExperimentPrediction]:
        """Predict experiment outcome using ML models."""
        try:
            return self.ml_predictor.predict_experiment_outcome(experiment_data)
        except Exception as e:
            logger.error(f"Error predicting experiment outcome: {e}")
            return None
    
    def detect_anomalies(self, experiments: List[Dict[str, Any]]) -> List[PerformanceAnomaly]:
        """Detect performance anomalies in experiments."""
        anomalies = []
        
        for exp in experiments:
            try:
                exp_anomalies = self.ml_predictor.detect_anomalies(exp)
                anomalies.extend(exp_anomalies)
            except Exception as e:
                logger.error(f"Error detecting anomalies in experiment {exp.get('id', 'unknown')}: {e}")
        
        return anomalies
    
    def generate_optimization_recommendations(self, current_config: Dict[str, Any], 
                                           historical_data: List[Dict[str, Any]]) -> List[str]:
        """Generate optimization recommendations based on historical data."""
        recommendations = []
        
        try:
            # Analyze historical performance
            best_performers = sorted(historical_data, 
                                   key=lambda x: x.get('stats', {}).get('best_fitness', 0), 
                                   reverse=True)[:5]
            
            if best_performers:
                # Analyze best performing configurations
                avg_pop_size = np.mean([exp.get('config', {}).get('population_size', 100) 
                                       for exp in best_performers])
                avg_mut_rate = np.mean([exp.get('config', {}).get('mutation_rate', 0.1) 
                                       for exp in best_performers])
                avg_cross_rate = np.mean([exp.get('config', {}).get('crossover_rate', 0.8) 
                                         for exp in best_performers])
                
                current_pop = current_config.get('population_size', 100)
                current_mut = current_config.get('mutation_rate', 0.1)
                current_cross = current_config.get('crossover_rate', 0.8)
                
                # Generate specific recommendations
                if abs(current_pop - avg_pop_size) > 20:
                    recommendations.append(f"Consider adjusting population size to around {int(avg_pop_size)} (currently {current_pop})")
                
                if abs(current_mut - avg_mut_rate) > 0.05:
                    recommendations.append(f"Consider adjusting mutation rate to around {avg_mut_rate:.3f} (currently {current_mut:.3f})")
                
                if abs(current_cross - avg_cross_rate) > 0.1:
                    recommendations.append(f"Consider adjusting crossover rate to around {avg_cross_rate:.3f} (currently {current_cross:.3f})")
            
            # General recommendations based on config analysis
            if current_config.get('population_size', 100) < 50:
                recommendations.append("Small population size may limit exploration - consider increasing")
            
            if current_config.get('mutation_rate', 0.1) > 0.3:
                recommendations.append("High mutation rate may prevent convergence - consider reducing")
            
            if len(recommendations) == 0:
                recommendations.append("Current configuration appears well-balanced based on historical data")
                
        except Exception as e:
            logger.error(f"Error generating optimization recommendations: {e}")
            recommendations.append("Unable to generate specific recommendations - consider experimenting with different parameters")
        
        return recommendations
    
    def analyze_performance_trends(self, experiments: List[Dict[str, Any]], metric: str = "best_fitness") -> Dict[str, Any]:
        """Analyze performance trends over time."""
        trends = {
            "metric": metric,
            "data_points": len(experiments),
            "trend_direction": "stable",
            "trend_strength": 0.0,
            "recent_average": 0.0,
            "historical_average": 0.0,
            "improvement_rate": 0.0
        }
        
        if len(experiments) < 2:
            return trends
        
        try:
            # Sort by start time
            sorted_experiments = sorted(experiments, 
                                      key=lambda x: x.get('start_time', datetime.min))
            
            # Extract metric values
            values = []
            for exp in sorted_experiments:
                stats = exp.get('stats', {})
                if metric in stats:
                    values.append(stats[metric])
                elif metric == "efficiency":
                    generations = exp.get('config', {}).get('generations', 100)
                    fitness = stats.get('best_fitness', 0)
                    efficiency = fitness / generations if generations > 0 else 0
                    values.append(efficiency)
                else:
                    values.append(0)
            
            if values:
                # Calculate trend
                x = np.arange(len(values))
                if len(values) > 1:
                    trend_slope = np.polyfit(x, values, 1)[0]
                    trend_r2 = np.corrcoef(x, values)[0, 1] ** 2 if len(values) > 2 else 0
                    
                    trends["trend_strength"] = float(trend_r2)
                    trends["improvement_rate"] = float(trend_slope)
                    
                    if trend_slope > 0.01:
                        trends["trend_direction"] = "improving"
                    elif trend_slope < -0.01:
                        trends["trend_direction"] = "declining"
                    else:
                        trends["trend_direction"] = "stable"
                
                # Calculate averages
                recent_values = values[-min(5, len(values)):]  # Last 5 experiments
                trends["recent_average"] = float(np.mean(recent_values))
                trends["historical_average"] = float(np.mean(values))
                
                # Add detailed data
                trends["time_series"] = [
                    {
                        "timestamp": exp.get('start_time', datetime.now()).isoformat(),
                        "value": float(val)
                    }
                    for exp, val in zip(sorted_experiments, values)
                ]
                
        except Exception as e:
            logger.error(f"Error analyzing performance trends: {e}")
        
        return trends
    
    def train_ml_models(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train ML models on historical data."""
        try:
            metrics = self.ml_predictor.train(training_data)
            return metrics
        except Exception as e:
            logger.error(f"Error training ML models: {e}")
            return {"error": str(e)}
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of ML models."""
        try:
            return {
                "is_trained": self.ml_predictor.is_trained,
                "model_file_exists": self.ml_predictor.model_path.exists(),
                "last_training": "unknown",  # Could be enhanced to track training time
                "features_count": len(self.ml_predictor.data_processor.feature_names),
                "predictor_types": [
                    "fitness_predictor",
                    "convergence_predictor", 
                    "anomaly_detector"
                ]
            }
        except Exception as e:
            logger.error(f"Error getting model status: {e}")
            return {"error": str(e)}

# Global analytics instance
advanced_analytics = AdvancedAnalytics()
