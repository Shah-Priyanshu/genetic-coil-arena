"""
Phase 4A: Experiment Templates API Routes
RESTful endpoints for managing and using experiment templates.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

from .templates import template_manager, ExperimentType, ExperimentTemplate
from .auth import get_current_user

# Create router
templates_router = APIRouter(prefix="/templates", tags=["templates"])


# Pydantic models for API
class TemplateResponse(BaseModel):
    """Response model for template data."""
    id: str
    name: str
    description: str
    category: str
    estimated_duration: str
    difficulty_level: str
    tags: List[str]
    version: str
    created_by: str


class TemplateDetailResponse(TemplateResponse):
    """Detailed template response including configuration."""
    config: Dict[str, Any]


class CreateExperimentFromTemplateRequest(BaseModel):
    """Request model for creating experiment from template."""
    experiment_name: str
    custom_config: Optional[Dict[str, Any]] = None


class CustomTemplateRequest(BaseModel):
    """Request model for creating custom templates."""
    name: str
    description: str
    category: str
    config: Dict[str, Any]
    estimated_duration: str
    difficulty_level: str
    tags: List[str]


@templates_router.get("/", response_model=List[TemplateResponse])
async def list_templates(
    category: Optional[str] = Query(None, description="Filter by category"),
    difficulty: Optional[str] = Query(None, description="Filter by difficulty level"),
    tags: Optional[str] = Query(None, description="Filter by tags (comma-separated)")
):
    """Get list of all available experiment templates."""
    
    # Get templates based on filters
    if category:
        try:
            category_enum = ExperimentType(category)
            templates = template_manager.list_templates(category_enum)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid category")
    elif difficulty:
        templates = template_manager.get_templates_by_difficulty(difficulty)
    elif tags:
        tag_list = [tag.strip() for tag in tags.split(",")]
        templates = template_manager.get_templates_by_tags(tag_list)
    else:
        templates = template_manager.list_templates()
    
    # Convert to response models
    return [
        TemplateResponse(
            id=t.id,
            name=t.name,
            description=t.description,
            category=t.category.value,
            estimated_duration=t.estimated_duration,
            difficulty_level=t.difficulty_level,
            tags=t.tags,
            version=t.version,
            created_by=t.created_by
        )
        for t in templates
    ]


@templates_router.get("/categories")
async def get_template_categories():
    """Get all available template categories."""
    return {
        "categories": [
            {
                "value": category.value,
                "name": category.value.replace("_", " ").title(),
                "description": _get_category_description(category)
            }
            for category in ExperimentType
        ]
    }


@templates_router.get("/difficulties")
async def get_difficulty_levels():
    """Get all available difficulty levels."""
    return {
        "difficulties": [
            {
                "value": "beginner",
                "name": "Beginner",
                "description": "Perfect for new users, quick and simple experiments"
            },
            {
                "value": "intermediate", 
                "name": "Intermediate",
                "description": "For users with some experience, balanced complexity"
            },
            {
                "value": "advanced",
                "name": "Advanced", 
                "description": "Complex experiments for experienced researchers"
            }
        ]
    }


@templates_router.get("/{template_id}", response_model=TemplateDetailResponse)
async def get_template(template_id: str):
    """Get detailed information about a specific template."""
    template = template_manager.get_template(template_id)
    
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    
    return TemplateDetailResponse(
        id=template.id,
        name=template.name,
        description=template.description,
        category=template.category.value,
        estimated_duration=template.estimated_duration,
        difficulty_level=template.difficulty_level,
        tags=template.tags,
        version=template.version,
        created_by=template.created_by,
        config=template.config
    )


@templates_router.post("/{template_id}/create-experiment")
async def create_experiment_from_template(
    template_id: str,
    request: CreateExperimentFromTemplateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create a new experiment from a template."""
    try:
        experiment = template_manager.create_experiment_from_template(
            template_id=template_id,
            experiment_name=request.experiment_name,
            user_id=current_user.get("username"),
            custom_config=request.custom_config
        )
        
        return {
            "message": "Experiment created successfully from template",
            "experiment": experiment
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create experiment: {str(e)}")


@templates_router.post("/custom", response_model=TemplateResponse)
async def create_custom_template(
    request: CustomTemplateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create a custom template (requires authentication)."""
    
    # Generate template ID
    template_id = f"custom_{current_user.get('username', 'user')}_{request.name.lower().replace(' ', '_')}"
    
    # Validate category
    try:
        category_enum = ExperimentType(request.category)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid category")
    
    # Create template
    template = ExperimentTemplate(
        id=template_id,
        name=request.name,
        description=request.description,
        category=category_enum,
        config=request.config,
        estimated_duration=request.estimated_duration,
        difficulty_level=request.difficulty_level,
        tags=request.tags,
        created_by=current_user.get("username", "user"),
        version="1.0"
    )
    
    # Add to manager
    if not template_manager.add_custom_template(template):
        raise HTTPException(status_code=409, detail="Template with this ID already exists")
    
    return TemplateResponse(
        id=template.id,
        name=template.name,
        description=template.description,
        category=template.category.value,
        estimated_duration=template.estimated_duration,
        difficulty_level=template.difficulty_level,
        tags=template.tags,
        version=template.version,
        created_by=template.created_by
    )


@templates_router.get("/{template_id}/export")
async def export_template(template_id: str):
    """Export a template as JSON."""
    exported = template_manager.export_template(template_id)
    
    if not exported:
        raise HTTPException(status_code=404, detail="Template not found")
    
    return {
        "template_id": template_id,
        "export_data": exported,
        "format": "json"
    }


@templates_router.post("/import")
async def import_template(
    template_data: dict,
    current_user: dict = Depends(get_current_user)
):
    """Import a template from JSON data."""
    try:
        import json
        template_json = json.dumps(template_data)
        
        if template_manager.import_template(template_json):
            return {"message": "Template imported successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to import template")
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Import failed: {str(e)}")


def _get_category_description(category: ExperimentType) -> str:
    """Get description for experiment category."""
    descriptions = {
        ExperimentType.QUICK_START: "Fast experiments perfect for getting started",
        ExperimentType.PERFORMANCE_TEST: "High-performance benchmarks and stress tests",
        ExperimentType.LONG_EVOLUTION: "Extended evolution studies for research",
        ExperimentType.COMPARISON_STUDY: "Compare different algorithms or parameters",
        ExperimentType.PARAMETER_SWEEP: "Systematic parameter space exploration",
        ExperimentType.BASELINE_BENCHMARK: "Standard benchmarks for comparison"
    }
    return descriptions.get(category, "Custom experiment type")
