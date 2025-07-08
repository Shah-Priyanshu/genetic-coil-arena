"""
Analytics API Routes for Genetic Coil Arena
Phase 5: Advanced Analytics & AI Integration
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta
import traceback

from ..advanced_analytics import AdvancedAnalytics, AnalyticsInsight, ExperimentPrediction

logger = logging.getLogger(__name__)

# Create analytics router
analytics_router = APIRouter(prefix="/analytics", tags=["analytics"])

# Initialize analytics engine
analytics_engine = AdvancedAnalytics()


# Request/Response models
class PredictionRequest(BaseModel):
    config: Dict[str, Any]
    experiment_id: Optional[str] = None


class OptimizationRequest(BaseModel):
    config: Optional[Dict[str, Any]] = None
    experiment_id: Optional[str] = None


class AnalyticsResponse(BaseModel):
    status: str
    data: Dict[str, Any]
    timestamp: str


@analytics_router.get('/health')
async def analytics_health():
    """Check analytics service health."""
    try:
        status = {
            'status': 'healthy',
            'service': 'Advanced Analytics Engine',
            'timestamp': datetime.now().isoformat(),
            'ml_models_loaded': analytics_engine.ml_predictor is not None,
            'data_processor_ready': hasattr(analytics_engine, 'data_processor'),
            'anomaly_detector_ready': hasattr(analytics_engine, 'anomaly_detector')
        }
        return status
    except Exception as e:
        logger.error(f"Analytics health check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
        )


@analytics_router.get('/insights')
async def get_insights(
    experiment_id: Optional[str] = Query(None, description="Specific experiment ID to analyze"),
    limit: int = Query(10, description="Maximum number of insights to return"),
    category: str = Query("all", description="Filter insights by category"),
    min_confidence: float = Query(0.5, description="Minimum confidence threshold")
):
    """Get AI-powered insights about experiments."""
    try:
        # Mock data for now since we don't have database integration
        # In production, this would fetch from the actual database
        experiment_data = [
            {
                'id': '1',
                'config': {
                    'population_size': 100,
                    'generations': 50,
                    'mutation_rate': 0.1,
                    'crossover_rate': 0.8
                },
                'results': {
                    'stats': {
                        'best_fitness': 0.85,
                        'average_fitness': 0.65,
                        'diversity_score': 0.7,
                        'convergence_rate': 0.02
                    }
                },
                'start_time': datetime.now() - timedelta(hours=1),
                'status': 'completed'
            }
        ]
        
        # Generate insights
        insights = analytics_engine.generate_insights(experiment_data)
        
        # Filter insights
        filtered_insights = []
        for insight in insights:
            if insight.confidence >= min_confidence:
                if category == 'all' or insight.category == category:
                    filtered_insights.append({
                        'type': insight.type,
                        'title': insight.title,
                        'description': insight.description,
                        'confidence': insight.confidence,
                        'importance': insight.importance,
                        'category': insight.category,
                        'actionable': insight.actionable,
                        'data': insight.data,
                        'timestamp': insight.timestamp.isoformat()
                    })
        
        # Sort by importance and limit
        filtered_insights.sort(key=lambda x: x['importance'], reverse=True)
        filtered_insights = filtered_insights[:limit]
        
        return {
            'insights': filtered_insights,
            'total_analyzed': len(experiment_data),
            'total_insights': len(insights),
            'filtered_insights': len(filtered_insights),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating insights: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Failed to generate insights',
                'details': str(e)
            }
        )


@analytics_router.post('/predictions')
async def create_prediction(prediction_request: PredictionRequest):
    """Create prediction for experiment outcome."""
    try:
        # Mock training data for now
        training_data = [
            {
                'id': '1',
                'config': {
                    'population_size': 100,
                    'generations': 50,
                    'mutation_rate': 0.1,
                    'crossover_rate': 0.8
                },
                'results': {
                    'stats': {
                        'best_fitness': 0.85,
                        'average_fitness': 0.65,
                        'convergence_rate': 0.02
                    }
                },
                'start_time': datetime.now() - timedelta(hours=1)
            }
        ]
        
        # Train model if needed
        if training_data:
            analytics_engine.train_ml_models(training_data)
        
        # Create prediction
        prediction_data = {
            'experiment_id': prediction_request.experiment_id or 'prediction',
            'config': prediction_request.config,
            'start_time': datetime.now()
        }
        
        prediction = analytics_engine.predict_experiment_outcome(prediction_data)
        
        if prediction:
            return {
                'prediction': {
                    'experiment_id': prediction.experiment_id,
                    'predicted_best_fitness': prediction.predicted_best_fitness,
                    'predicted_convergence_generation': prediction.predicted_convergence_generation,
                    'confidence_interval': prediction.confidence_interval,
                    'estimated_completion_time': prediction.estimated_completion_time.isoformat(),
                    'success_probability': prediction.success_probability,
                    'bottlenecks': prediction.bottlenecks,
                    'recommendations': prediction.recommendations
                },
                'timestamp': datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=400,
                detail={
                    'error': 'Could not generate prediction',
                    'message': 'Insufficient training data or model not ready'
                }
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating prediction: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Failed to create prediction',
                'details': str(e)
            }
        )


@analytics_router.get('/anomalies')
async def get_anomalies(
    hours: int = Query(24, description="Hours to look back for anomalies"),
    severity: str = Query("all", description="Filter by severity level"),
    limit: int = Query(20, description="Maximum number of anomalies to return")
):
    """Detect and return performance anomalies."""
    try:
        # Mock experiment data for now
        experiment_data = [
            {
                'id': '1',
                'config': {
                    'population_size': 100,
                    'generations': 50,
                    'mutation_rate': 0.1,
                    'crossover_rate': 0.8
                },
                'results': {
                    'stats': {
                        'best_fitness': 0.85,
                        'average_fitness': 0.65,
                        'diversity_score': 0.7,
                        'convergence_rate': 0.02
                    }
                },
                'start_time': datetime.now() - timedelta(hours=1),
                'status': 'completed'
            }
        ]
        
        # Detect anomalies
        anomalies = analytics_engine.detect_anomalies(experiment_data)
        
        # Filter by severity
        filtered_anomalies = []
        for anomaly in anomalies:
            if severity == 'all' or anomaly.severity == severity:
                filtered_anomalies.append({
                    'experiment_id': anomaly.experiment_id,
                    'anomaly_type': anomaly.anomaly_type,
                    'severity': anomaly.severity,
                    'description': anomaly.description,
                    'detected_at': anomaly.detected_at.isoformat(),
                    'metrics': anomaly.metrics,
                    'suggested_actions': anomaly.suggested_actions
                })
        
        # Sort by severity and limit
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        filtered_anomalies.sort(key=lambda x: severity_order.get(x['severity'], 4))
        filtered_anomalies = filtered_anomalies[:limit]
        
        return {
            'anomalies': filtered_anomalies,
            'total_experiments': len(experiment_data),
            'total_anomalies': len(anomalies),
            'filtered_anomalies': len(filtered_anomalies),
            'analysis_period_hours': hours,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error detecting anomalies: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Failed to detect anomalies',
                'details': str(e)
            }
        )


@analytics_router.post('/optimization/recommendations')
async def get_optimization_recommendations(optimization_request: OptimizationRequest):
    """Get AI-powered optimization recommendations."""
    try:
        current_config = optimization_request.config or {}
        
        if not current_config:
            raise HTTPException(
                status_code=400,
                detail={'error': 'No configuration provided'}
            )
        
        # Mock historical data for now
        historical_data = [
            {
                'id': '1',
                'config': {
                    'population_size': 100,
                    'generations': 50,
                    'mutation_rate': 0.1,
                    'crossover_rate': 0.8
                },
                'results': {
                    'stats': {
                        'best_fitness': 0.85,
                        'average_fitness': 0.65,
                        'convergence_rate': 0.02
                    }
                },
                'start_time': datetime.now() - timedelta(hours=1)
            }
        ]
        
        # Generate recommendations
        recommendations = analytics_engine.generate_optimization_recommendations(
            current_config, historical_data
        )
        
        return {
            'recommendations': recommendations,
            'current_config': current_config,
            'analysis_based_on': len(historical_data),
            'timestamp': datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Failed to generate recommendations',
                'details': str(e)
            }
        )


@analytics_router.get('/performance/trends')
async def get_performance_trends(
    days: int = Query(7, description="Number of days to analyze"),
    metric: str = Query("best_fitness", description="Performance metric to analyze")
):
    """Get performance trend analysis."""
    try:
        # Mock experiment data for now
        experiment_data = [
            {
                'id': '1',
                'config': {
                    'population_size': 100,
                    'generations': 50,
                    'mutation_rate': 0.1,
                    'crossover_rate': 0.8
                },
                'results': {
                    'stats': {
                        'best_fitness': 0.85,
                        'average_fitness': 0.65,
                        'diversity_score': 0.7,
                        'convergence_rate': 0.02
                    }
                },
                'start_time': datetime.now() - timedelta(hours=1),
                'status': 'completed'
            }
        ]
        
        # Analyze trends
        trends = analytics_engine.analyze_performance_trends(experiment_data, metric)
        
        return {
            'trends': trends,
            'metric': metric,
            'analysis_period_days': days,
            'experiments_analyzed': len(experiment_data),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error analyzing trends: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Failed to analyze trends',
                'details': str(e)
            }
        )


@analytics_router.get('/model/status')
async def get_model_status():
    """Get ML model training status and metrics."""
    try:
        status = analytics_engine.get_model_status()
        return {
            'model_status': status,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Failed to get model status',
                'details': str(e)
            }
        )


@analytics_router.post('/model/retrain')
async def retrain_models():
    """Manually trigger model retraining."""
    try:
        # Mock training data for now
        training_data = [
            {
                'id': '1',
                'config': {
                    'population_size': 100,
                    'generations': 50,
                    'mutation_rate': 0.1,
                    'crossover_rate': 0.8
                },
                'results': {
                    'stats': {
                        'best_fitness': 0.85,
                        'average_fitness': 0.65,
                        'convergence_rate': 0.02
                    }
                },
                'start_time': datetime.now() - timedelta(hours=1)
            }
        ]
        
        if len(training_data) < 10:
            # Create more mock data for demonstration
            for i in range(10):
                training_data.append({
                    'id': str(i + 2),
                    'config': {
                        'population_size': 100 + i * 10,
                        'generations': 50 + i * 5,
                        'mutation_rate': 0.1 + i * 0.01,
                        'crossover_rate': 0.8 - i * 0.01
                    },
                    'results': {
                        'stats': {
                            'best_fitness': 0.85 - i * 0.02,
                            'average_fitness': 0.65 - i * 0.01,
                            'convergence_rate': 0.02 + i * 0.001
                        }
                    },
                    'start_time': datetime.now() - timedelta(hours=i + 1)
                })
        
        # Retrain models
        metrics = analytics_engine.train_ml_models(training_data)
        
        return {
            'message': 'Models retrained successfully',
            'training_metrics': metrics,
            'training_data_size': len(training_data),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error retraining models: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Failed to retrain models',
                'details': str(e)
            }
        )
