"""
Phase 4C: Version Control Routes
FastAPI routes for experiment version control features.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Dict, Any, Optional
import logging

from .version_control import version_control, MergeStrategy
from ..core.auth import get_current_user

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/version-control", tags=["version-control"])


@router.get("/experiments/{experiment_id}/versions", summary="Get Experiment Versions")
async def get_experiment_versions(
    experiment_id: str,
    branch: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=100),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get all versions of an experiment.
    
    Args:
        experiment_id: ID of the experiment
        branch: Optional branch filter
        limit: Maximum number of versions to return
        
    Returns:
        List of experiment versions with metadata
    """
    try:
        # Get all versions for this experiment
        experiment_versions = [
            v for v in version_control.versions.values()
            if v.experiment_id == experiment_id
        ]
        
        # Filter by branch if specified
        if branch:
            branch_key = f"{experiment_id}_{branch}"
            if branch_key in version_control.branches:
                branch_obj = version_control.branches[branch_key]
                experiment_versions = [
                    v for v in experiment_versions
                    if v.version_id in branch_obj.versions
                ]
        
        # Sort by creation date (newest first)
        experiment_versions.sort(key=lambda x: x.metadata.created_at, reverse=True)
        
        # Apply limit
        experiment_versions = experiment_versions[:limit]
          # Convert to API format
        versions_data: list[Dict[str, Any]] = []
        for version in experiment_versions:
            versions_data.append({
                "version_id": version.version_id,
                "version_number": version.version_number,
                "type": version.type.value,
                "author": version.metadata.author,
                "created_at": version.metadata.created_at.isoformat(),
                "message": version.metadata.message,
                "tags": version.metadata.tags,
                "parent_versions": version.metadata.parent_versions,
                "child_versions": version.metadata.child_versions,
                "is_branch": version.metadata.is_branch,
                "is_merge": version.metadata.is_merge,
                "checksum": version.checksum,
                "has_results": version.results is not None
            })
        
        return {
            "experiment_id": experiment_id,
            "versions": versions_data,
            "total": len(versions_data)
        }
        
    except Exception as e:
        logger.error(f"Error getting experiment versions: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve versions")


@router.get("/experiments/{experiment_id}/version/{version_id}", summary="Get Specific Version")
async def get_experiment_version(
    experiment_id: str,
    version_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get a specific version of an experiment.
    
    Args:
        experiment_id: ID of the experiment
        version_id: ID of the version
        
    Returns:
        Complete version data including configuration
    """
    try:
        if version_id not in version_control.versions:
            raise HTTPException(status_code=404, detail="Version not found")
        
        version = version_control.versions[version_id]
        
        if version.experiment_id != experiment_id:
            raise HTTPException(status_code=400, detail="Version does not belong to this experiment")
        
        return {
            "version_id": version.version_id,
            "experiment_id": version.experiment_id,
            "version_number": version.version_number,
            "type": version.type.value,
            "config": version.config,
            "results": version.results,
            "metadata": {
                "author": version.metadata.author,
                "created_at": version.metadata.created_at.isoformat(),
                "message": version.metadata.message,
                "tags": version.metadata.tags,
                "parent_versions": version.metadata.parent_versions,
                "child_versions": version.metadata.child_versions,
                "is_branch": version.metadata.is_branch,
                "is_merge": version.metadata.is_merge,
                "merge_strategy": version.metadata.merge_strategy.value if version.metadata.merge_strategy else None,
                "conflicts_resolved": version.metadata.conflicts_resolved
            },
            "checksum": version.checksum
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting version {version_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve version")


@router.post("/experiments/{experiment_id}/versions", summary="Create New Version")
async def create_experiment_version(
    experiment_id: str,
    version_data: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Create a new version of an experiment.
    
    Args:
        experiment_id: ID of the experiment
        version_data: Version configuration and metadata
        
    Returns:
        Created version information
    """
    try:
        config = version_data.get('config', {})
        message = version_data.get('message', '')
        parent_version = version_data.get('parent_version')
        branch_name = version_data.get('branch', 'main')
        author = current_user.get('username', 'unknown')
        
        if not config:
            raise HTTPException(status_code=400, detail="Configuration is required")
        
        # Check if this is the first version
        existing_versions = [
            v for v in version_control.versions.values()
            if v.experiment_id == experiment_id
        ]
        
        if not existing_versions:
            # Create initial version
            version = version_control.create_initial_version(
                experiment_id=experiment_id,
                config=config,
                author=author,
                message=message or "Initial version"
            )
        else:
            # Create new version
            version = version_control.create_version(
                experiment_id=experiment_id,
                config=config,
                author=author,
                message=message,
                parent_version=parent_version,
                branch_name=branch_name
            )
        
        logger.info(f"Created version {version.version_id} for experiment {experiment_id}")
        
        return {
            "version_id": version.version_id,
            "version_number": version.version_number,
            "type": version.type.value,
            "author": version.metadata.author,
            "created_at": version.metadata.created_at.isoformat(),
            "message": version.metadata.message,
            "checksum": version.checksum
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating version: {e}")
        raise HTTPException(status_code=500, detail="Failed to create version")


@router.get("/experiments/{experiment_id}/branches", summary="Get Experiment Branches")
async def get_experiment_branches(
    experiment_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get all branches for an experiment.
    
    Args:
        experiment_id: ID of the experiment
        
    Returns:
        List of branches with metadata
    """
    try:
        branches = version_control.get_experiment_branches(experiment_id)
        
        branches_data: list[Dict[str, Any]] = []
        for branch in branches:
            # Get version count and latest activity
            version_count = len(branch.versions)
            latest_version = None
            if branch.head_version_id in version_control.versions:
                latest_version = version_control.versions[branch.head_version_id]
            
            branches_data.append({
                "branch_name": branch.branch_name,
                "base_version_id": branch.base_version_id,
                "head_version_id": branch.head_version_id,
                "created_by": branch.created_by,
                "created_at": branch.created_at.isoformat(),
                "description": branch.description,
                "is_active": branch.is_active,
                "version_count": version_count,
                "latest_activity": latest_version.metadata.created_at.isoformat() if latest_version else branch.created_at.isoformat()
            })
        
        return {
            "experiment_id": experiment_id,
            "branches": branches_data,
            "total": len(branches_data)
        }
        
    except Exception as e:
        logger.error(f"Error getting branches: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve branches")


@router.post("/experiments/{experiment_id}/branches", summary="Create New Branch")
async def create_experiment_branch(
    experiment_id: str,
    branch_data: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Create a new branch for an experiment.
    
    Args:
        experiment_id: ID of the experiment
        branch_data: Branch configuration
        
    Returns:
        Created branch information
    """
    try:
        branch_name = branch_data.get('branch_name')
        base_version = branch_data.get('base_version')
        description = branch_data.get('description', '')
        author = current_user.get('username', 'unknown')
        
        if not branch_name:
            raise HTTPException(status_code=400, detail="Branch name is required")
        
        branch = version_control.create_branch(
            experiment_id=experiment_id,
            branch_name=branch_name,
            base_version=base_version,
            author=author,
            description=description
        )
        
        logger.info(f"Created branch {branch_name} for experiment {experiment_id}")
        
        return {
            "branch_name": branch.branch_name,
            "base_version_id": branch.base_version_id,
            "head_version_id": branch.head_version_id,
            "created_by": branch.created_by,
            "created_at": branch.created_at.isoformat(),
            "description": branch.description
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating branch: {e}")
        raise HTTPException(status_code=500, detail="Failed to create branch")


@router.get("/versions/{version1_id}/diff/{version2_id}", summary="Compare Versions")
async def compare_versions(
    version1_id: str,
    version2_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Compare two experiment versions.
    
    Args:
        version1_id: First version ID
        version2_id: Second version ID
        
    Returns:
        Differences between the versions
    """
    try:
        diff = version_control.get_version_diff(version1_id, version2_id)
        
        return {
            "version1_id": version1_id,
            "version2_id": version2_id,
            "added_parameters": diff.added_parameters,
            "removed_parameters": diff.removed_parameters,
            "modified_parameters": diff.modified_parameters,
            "metadata_changes": diff.metadata_changes
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error comparing versions: {e}")
        raise HTTPException(status_code=500, detail="Failed to compare versions")


@router.post("/merge-requests", summary="Create Merge Request")
async def create_merge_request(
    merge_data: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Create a merge request between two versions.
    
    Args:
        merge_data: Merge request configuration
        
    Returns:
        Created merge request information
    """
    try:
        source_version_id = merge_data.get('source_version_id')
        target_version_id = merge_data.get('target_version_id')
        strategy_str = merge_data.get('strategy', 'merge_parameters')
        message = merge_data.get('message', '')
        auto_merge = merge_data.get('auto_merge', False)
        author = current_user.get('username', 'unknown')
        
        if not source_version_id or not target_version_id:
            raise HTTPException(
                status_code=400, 
                detail="source_version_id and target_version_id are required"
            )
        
        try:
            strategy = MergeStrategy(strategy_str)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid merge strategy: {strategy_str}")
        
        merge_request = version_control.create_merge_request(
            source_version_id=source_version_id,
            target_version_id=target_version_id,
            strategy=strategy,
            author=author,
            message=message,
            auto_merge=auto_merge
        )
        
        logger.info(f"Created merge request {merge_request.id}")
          # Convert conflicts to API format
        conflicts_data: list[Dict[str, Any]] = []
        for conflict in merge_request.conflicts:
            conflicts_data.append({
                "parameter_path": conflict.parameter_path,
                "base_value": conflict.base_value,
                "source_value": conflict.source_value,
                "target_value": conflict.target_value,
                "resolution": conflict.resolution,
                "resolved": conflict.resolved
            })
        
        return {
            "merge_request_id": merge_request.id,
            "source_version_id": merge_request.source_version_id,
            "target_version_id": merge_request.target_version_id,
            "strategy": merge_request.strategy.value,
            "author": merge_request.author,
            "created_at": merge_request.created_at.isoformat(),
            "message": merge_request.message,
            "status": merge_request.status,
            "auto_merge": merge_request.auto_merge,
            "conflicts": conflicts_data,
            "conflicts_count": len(conflicts_data)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating merge request: {e}")
        raise HTTPException(status_code=500, detail="Failed to create merge request")


@router.post("/merge-requests/{merge_request_id}/execute", summary="Execute Merge Request")
async def execute_merge_request(
    merge_request_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Execute a merge request.
    
    Args:
        merge_request_id: ID of the merge request
        
    Returns:
        Information about the merged version
    """
    try:
        merged_version = version_control.execute_merge(merge_request_id)
        
        logger.info(f"Executed merge request {merge_request_id}")
        
        return {
            "merged_version_id": merged_version.version_id,
            "version_number": merged_version.version_number,
            "author": merged_version.metadata.author,
            "created_at": merged_version.metadata.created_at.isoformat(),
            "message": merged_version.metadata.message,
            "parent_versions": merged_version.metadata.parent_versions
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error executing merge request: {e}")
        raise HTTPException(status_code=500, detail="Failed to execute merge request")


@router.post("/experiments/{experiment_id}/rollback", summary="Rollback Experiment")
async def rollback_experiment(
    experiment_id: str,
    rollback_data: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Rollback an experiment to a previous version.
    
    Args:
        experiment_id: ID of the experiment
        rollback_data: Rollback configuration
        
    Returns:
        Information about the rollback version
    """
    try:
        target_version_id = rollback_data.get('target_version_id')
        message = rollback_data.get('message', 'Rollback')
        author = current_user.get('username', 'unknown')
        
        if not target_version_id:
            raise HTTPException(status_code=400, detail="target_version_id is required")
        
        rollback_version = version_control.rollback_to_version(
            experiment_id=experiment_id,
            target_version_id=target_version_id,
            author=author,
            message=message
        )
        
        logger.info(f"Rolled back experiment {experiment_id} to version {target_version_id}")
        
        return {
            "rollback_version_id": rollback_version.version_id,
            "version_number": rollback_version.version_number,
            "target_version_id": target_version_id,
            "author": rollback_version.metadata.author,
            "created_at": rollback_version.metadata.created_at.isoformat(),
            "message": rollback_version.metadata.message
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rolling back experiment: {e}")
        raise HTTPException(status_code=500, detail="Failed to rollback experiment")


@router.get("/experiments/{experiment_id}/tree", summary="Get Version Tree")
async def get_version_tree(
    experiment_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get the version tree for an experiment.
    
    Args:
        experiment_id: ID of the experiment
        
    Returns:
        Version tree with relationships
    """
    try:
        tree = version_control.get_version_tree(experiment_id)
        
        # Convert to API format
        tree_data = {}
        for version_id, node in tree.items():
            version = node['version']
            tree_data[version_id] = {
                "version_number": version.version_number,
                "type": version.type.value,
                "author": version.metadata.author,
                "created_at": version.metadata.created_at.isoformat(),
                "message": version.metadata.message,
                "children": node['children'],
                "parents": node['parents'],
                "is_branch": version.metadata.is_branch,
                "is_merge": version.metadata.is_merge
            }
        
        return {
            "experiment_id": experiment_id,
            "version_tree": tree_data
        }
        
    except Exception as e:
        logger.error(f"Error getting version tree: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve version tree")


@router.get("/stats", summary="Get Version Control Statistics")
async def get_version_control_stats(
    experiment_id: Optional[str] = Query(None),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get version control statistics.
    
    Args:
        experiment_id: Optional experiment ID filter
        
    Returns:
        Version control statistics
    """
    try:
        stats = version_control.get_version_stats(experiment_id)
        
        return {
            "experiment_id": experiment_id,
            "statistics": stats
        }
        
    except Exception as e:
        logger.error(f"Error getting version control stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")


@router.get("/health", summary="Version Control Health Check")
async def version_control_health_check() -> Dict[str, Any]:
    """
    Check the health of the version control system.
    
    Returns:
        Health status and basic metrics
    """
    try:
        total_versions = len(version_control.versions)
        total_branches = len(version_control.branches)
        total_merge_requests = len(version_control.merge_requests)
        
        return {
            "status": "healthy",
            "service": "version_control",
            "metrics": {
                "total_versions": total_versions,
                "total_branches": total_branches,
                "total_merge_requests": total_merge_requests,
                "total_experiments": len(version_control.experiment_heads)
            }
        }
        
    except Exception as e:
        logger.error(f"Error checking version control health: {e}")
        return {
            "status": "unhealthy",
            "service": "version_control",
            "error": str(e)
        }
