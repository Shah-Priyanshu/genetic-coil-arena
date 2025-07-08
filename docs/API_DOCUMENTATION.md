# 📖 API Documentation

## 🌐 Genetic Coil Arena API Reference

Complete documentation for all API endpoints in the Genetic Coil Arena platform.

### 🏠 Base Information
- **Base URL**: `http://localhost:8000`
- **API Prefix**: `/api`
- **Documentation**: `/docs` (Swagger UI)
- **Alternative Docs**: `/redoc` (ReDoc)
- **Health Check**: `/health`

## 🔐 Authentication

Most endpoints require JWT authentication. Include the token in the Authorization header:

```bash
Authorization: Bearer <your_jwt_token>
```

### Get Access Token
```bash
POST /api/auth/token
Content-Type: application/x-www-form-urlencoded

username=admin&password=admin123
```

## 📋 Core Endpoints

### 🏥 System Health
```bash
GET /health
```
Returns system status and component availability.

**Response:**
```json
{
  "status": "healthy",
  "service": "Genetic Coil Arena",
  "version": "6.0.0",
  "components_available": true,
  "features": {
    "experiments": true,
    "analytics": true,
    "neural_evolution": true,
    "visualization": true,
    "collaboration": true,
    "mobile": true
  }
}
```

## 🧪 Experiments API

### List Experiments
```bash
GET /api/experiments/
```

### Create Experiment
```bash
POST /api/experiments/
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "My Experiment",
  "description": "Test experiment",
  "config": {
    "population_size": 100,
    "generations": 50,
    "mutation_rate": 0.1,
    "crossover_rate": 0.8,
    "elite_size": 5,
    "grid_size": 10,
    "max_steps": 500
  }
}
```

### Get Experiment Details
```bash
GET /api/experiments/{experiment_id}
```

### Start/Pause/Stop Experiment
```bash
POST /api/experiments/{experiment_id}/start
POST /api/experiments/{experiment_id}/pause
POST /api/experiments/{experiment_id}/stop
```

### Get Experiment Results
```bash
GET /api/experiments/{experiment_id}/results
```

## 🧠 Neural Evolution API

### Neural Health Check
```bash
GET /api/neural/health
```

### Get Neural Templates
```bash
GET /api/neural/templates
```

### Create Neural Experiment
```bash
POST /api/neural/experiments
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "Neural Evolution Test",
  "description": "NEAT-style evolution",
  "agent_config": {
    "input_size": 10,
    "hidden_layers": [64, 32],
    "output_size": 4,
    "activation": "relu",
    "dropout_rate": 0.1,
    "use_batch_norm": true,
    "network_type": "feedforward"
  },
  "evolution_config": {
    "population_size": 50,
    "generations": 100,
    "mutation_rate": 0.1,
    "mutation_strength": 0.02,
    "crossover_rate": 0.8,
    "elite_size": 5,
    "tournament_size": 3,
    "fitness_sharing": true,
    "speciation": true,
    "compatibility_threshold": 3.0
  },
  "environment_type": "grid_world",
  "environment_config": {
    "grid_size": 10,
    "max_steps": 100
  }
}
```

### Get Neural Experiment Status
```bash
GET /api/neural/experiments/{experiment_id}
```

### Test Best Agent
```bash
POST /api/neural/experiments/{experiment_id}/test-agent
```

## 📊 Analytics & AI API

### Analytics Health
```bash
GET /api/analytics/health
```

### Get ML Model Status
```bash
GET /api/analytics/model/status
```

### Get AI Insights
```bash
GET /api/analytics/insights?limit=10&category=all&min_confidence=0.5
```

### Create Prediction
```bash
POST /api/analytics/predictions
Content-Type: application/json

{
  "experiment_id": "optional_id",
  "config": {
    "population_size": 100,
    "generations": 50,
    "mutation_rate": 0.1
  }
}
```

### Get Performance Anomalies
```bash
GET /api/analytics/anomalies?hours=24&severity=all&limit=20
```

### Get Optimization Recommendations
```bash
POST /api/analytics/optimization/recommendations
Content-Type: application/json

{
  "config": {
    "population_size": 100,
    "generations": 50,
    "mutation_rate": 0.1,
    "crossover_rate": 0.8
  }
}
```

### Retrain ML Models
```bash
POST /api/analytics/model/retrain
Authorization: Bearer <token>
```

## 👥 Team Management API

### List Teams
```bash
GET /api/teams/
Authorization: Bearer <token>
```

### Get Team Details
```bash
GET /api/teams/{team_id}
Authorization: Bearer <token>
```

## 📦 Batch Processing API

### List Batch Jobs
```bash
GET /api/batch/
Authorization: Bearer <token>
```

### Create Batch Job
```bash
POST /api/batch/
Authorization: Bearer <token>
```

### Get Batch Progress
```bash
GET /api/batch/{batch_id}/progress
Authorization: Bearer <token>
```

## 🔔 Notifications API

### Get User Notifications
```bash
GET /api/notifications/?limit=50&unread_only=false
Authorization: Bearer <token>
```

### Get Notification Preferences
```bash
GET /api/notifications/preferences
Authorization: Bearer <token>
```

### Notification System Health
```bash
GET /api/notifications/health
```

## 📋 Templates API

### List Templates
```bash
GET /api/templates/
Authorization: Bearer <token>
```

## ⚡ Performance Monitoring API

### Get Performance Metrics
```bash
GET /api/performance/
```

## 📱 Mobile API

### Mobile Health Check
```bash
GET /api/mobile/
```

### Mobile App Configuration
```bash
GET /api/mobile/config
```

## 🎨 3D Visualization API

### 3D Visualization Health
```bash
GET /api/visualization_3d/
```

### Generate 3D Scene
```bash
POST /api/visualization_3d/scene
Content-Type: application/json

{
  "experiment_id": "exp_12345",
  "visualization_type": "population_fitness",
  "parameters": {
    "generation": 10,
    "show_agents": true
  }
}
```

## 🔄 Version Control API

### Get Experiment Versions
```bash
GET /api/version_control/experiments/{experiment_id}/versions
Authorization: Bearer <token>
```

## 📊 Comparison API

### Compare Experiments
```bash
POST /api/comparison/experiments
Authorization: Bearer <token>
Content-Type: application/json

{
  "experiment_ids": ["exp_1", "exp_2", "exp_3"],
  "comparison_metrics": ["best_fitness", "convergence_rate", "diversity"]
}
```

## ⏰ Scheduling API

### List Scheduled Tasks
```bash
GET /api/scheduling/tasks
Authorization: Bearer <token>
```

### Schedule Experiment
```bash
POST /api/scheduling/experiments
Authorization: Bearer <token>
```

## 📝 Response Formats

### Standard Success Response
```json
{
  "status": "success",
  "data": { ... },
  "timestamp": "2025-07-07T21:30:00Z"
}
```

### Error Response
```json
{
  "detail": "Error description",
  "status_code": 400,
  "timestamp": "2025-07-07T21:30:00Z"
}
```

### Paginated Response
```json
{
  "items": [ ... ],
  "total_count": 100,
  "page": 1,
  "page_size": 20,
  "total_pages": 5
}
```

## 🔧 Status Codes

- **200 OK** - Request successful
- **201 Created** - Resource created successfully
- **400 Bad Request** - Invalid request data
- **401 Unauthorized** - Authentication required
- **403 Forbidden** - Insufficient permissions
- **404 Not Found** - Resource not found
- **422 Unprocessable Entity** - Validation error
- **500 Internal Server Error** - Server error

## 🔍 Query Parameters

### Common Parameters
- `limit` - Maximum number of items to return (default: 20)
- `offset` - Number of items to skip (default: 0)
- `sort_by` - Field to sort by
- `order` - Sort order: `asc` or `desc`
- `filter` - Filter criteria

### Date Filters
- `start_date` - Filter from date (ISO 8601)
- `end_date` - Filter to date (ISO 8601)
- `hours` - Hours to look back from now

## 🔌 WebSocket Endpoints

### Real-time Experiment Updates
```
ws://localhost:8000/ws/experiments/{experiment_id}
```

### Real-time Notifications
```
ws://localhost:8000/ws/notifications
```

## 🚀 Rate Limiting

- **Default**: 100 requests per minute per IP
- **Authenticated**: 1000 requests per minute per user
- **Batch operations**: 10 requests per minute per user

## 📚 SDK Examples

### Python
```python
import requests

# Base configuration
BASE_URL = "http://localhost:8000"
token = "your_jwt_token"
headers = {"Authorization": f"Bearer {token}"}

# Create experiment
experiment = {
    "name": "Python SDK Test",
    "config": {
        "population_size": 100,
        "generations": 50
    }
}

response = requests.post(
    f"{BASE_URL}/api/experiments/",
    json=experiment,
    headers=headers
)

print(response.json())
```

### JavaScript
```javascript
const BASE_URL = 'http://localhost:8000';
const token = 'your_jwt_token';

const headers = {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
};

// Create experiment
const experiment = {
    name: 'JavaScript SDK Test',
    config: {
        population_size: 100,
        generations: 50
    }
};

fetch(`${BASE_URL}/api/experiments/`, {
    method: 'POST',
    headers: headers,
    body: JSON.stringify(experiment)
})
.then(response => response.json())
.then(data => console.log(data));
```

### cURL
```bash
# Get health status
curl http://localhost:8000/health

# Create experiment with authentication
curl -X POST http://localhost:8000/api/experiments/ \
  -H "Authorization: Bearer your_jwt_token" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "cURL Test",
    "config": {
      "population_size": 100,
      "generations": 50
    }
  }'
```

## 🛡️ Security

- **JWT Authentication** for protected endpoints
- **HTTPS** recommended for production
- **Rate limiting** to prevent abuse
- **Input validation** on all endpoints
- **CORS** configured for cross-origin requests

## 📞 Support

- **Interactive Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **GitHub Issues**: Repository issues page
- **API Status**: Monitor via health endpoints

---

**📖 For the most up-to-date API documentation, visit `/docs` when the server is running.**
