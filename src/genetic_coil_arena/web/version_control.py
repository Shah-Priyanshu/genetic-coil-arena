"""
Phase 4C: Experiment Version Control
Version control system for experiments with branching, merging, and rollback.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from enum import Enum
import logging
import json
import hashlib

logger = logging.getLogger(__name__)


class VersionType(Enum):
    """Types of experiment versions."""
    INITIAL = "initial"
    BRANCH = "branch"
    MERGE = "merge"
    ROLLBACK = "rollback"
    FORK = "fork"


class MergeStrategy(Enum):
    """Strategies for merging experiment versions."""
    OVERWRITE = "overwrite"
    MERGE_PARAMETERS = "merge_parameters"
    KEEP_BOTH = "keep_both"
    MANUAL = "manual"


@dataclass
class ExperimentDiff:
    """Difference between two experiment versions."""
    added_parameters: Dict[str, Any] = field(default_factory=dict)
    removed_parameters: Dict[str, Any] = field(default_factory=dict)
    modified_parameters: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metadata_changes: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class VersionMetadata:
    """Metadata for an experiment version."""
    author: str
    created_at: datetime = field(default_factory=datetime.now)
    message: str = ""
    tags: List[str] = field(default_factory=list)
    parent_versions: List[str] = field(default_factory=list)
    child_versions: List[str] = field(default_factory=list)
    is_branch: bool = False
    is_merge: bool = False
    merge_strategy: Optional[MergeStrategy] = None
    conflicts_resolved: bool = True


@dataclass
class ExperimentVersion:
    """A version of an experiment."""
    version_id: str
    experiment_id: str
    version_number: str  # e.g., "1.0", "1.1", "2.0-branch"
    type: VersionType
    config: Dict[str, Any]
    results: Optional[Dict[str, Any]] = None
    metadata: VersionMetadata = field(default_factory=lambda: VersionMetadata(author="system"))
    checksum: str = ""
    
    def __post_init__(self):
        """Calculate checksum after initialization."""
        if not self.checksum:
            self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calculate checksum of the experiment configuration."""
        config_str = json.dumps(self.config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]


@dataclass
class VersionBranch:
    """A branch in the version tree."""
    branch_name: str
    base_version_id: str
    head_version_id: str
    created_by: str
    created_at: datetime = field(default_factory=datetime.now)
    description: str = ""
    is_active: bool = True
    versions: List[str] = field(default_factory=list)


@dataclass
class MergeConflict:
    """A merge conflict between versions."""
    parameter_path: str
    base_value: Any
    source_value: Any
    target_value: Any
    resolution: Optional[Any] = None
    resolved: bool = False


@dataclass
class MergeRequest:
    """A request to merge two experiment versions."""
    id: str
    source_version_id: str
    target_version_id: str
    strategy: MergeStrategy
    author: str
    created_at: datetime = field(default_factory=datetime.now)
    message: str = ""
    conflicts: List[MergeConflict] = field(default_factory=list)
    status: str = "pending"  # pending, approved, rejected, merged
    auto_merge: bool = False


class ExperimentVersionControl:
    """Version control system for experiments."""
    
    def __init__(self):
        self.versions: Dict[str, ExperimentVersion] = {}
        self.branches: Dict[str, VersionBranch] = {}
        self.merge_requests: Dict[str, MergeRequest] = {}
        self.experiment_heads: Dict[str, str] = {}  # experiment_id -> latest version_id
        self.version_tree: Dict[str, List[str]] = {}  # parent -> children
    
    def create_initial_version(self, experiment_id: str, config: Dict[str, Any], 
                             author: str, message: str = "Initial version") -> ExperimentVersion:
        """Create the initial version of an experiment."""
        version_id = f"{experiment_id}_v1.0"
        
        metadata = VersionMetadata(
            author=author,
            message=message,
            tags=["initial", "v1.0"]
        )
        
        version = ExperimentVersion(
            version_id=version_id,
            experiment_id=experiment_id,
            version_number="1.0",
            type=VersionType.INITIAL,
            config=config.copy(),
            metadata=metadata
        )
        
        self.versions[version_id] = version
        self.experiment_heads[experiment_id] = version_id
        self.version_tree[version_id] = []
        
        # Create main branch
        main_branch = VersionBranch(
            branch_name="main",
            base_version_id=version_id,
            head_version_id=version_id,
            created_by=author,
            description="Main development branch",
            versions=[version_id]
        )
        
        branch_key = f"{experiment_id}_main"
        self.branches[branch_key] = main_branch
        
        logger.info(f"Created initial version {version_id} for experiment {experiment_id}")
        return version
    
    def create_version(self, experiment_id: str, config: Dict[str, Any], 
                      author: str, message: str = "", parent_version: Optional[str] = None,
                      branch_name: str = "main") -> ExperimentVersion:
        """Create a new version of an experiment."""
        
        # Get parent version
        if not parent_version:
            parent_version = self.experiment_heads.get(experiment_id)
            
        if not parent_version or parent_version not in self.versions:
            raise ValueError(f"Parent version {parent_version} not found")
        
        parent = self.versions[parent_version]
        
        # Generate version number
        if branch_name == "main":
            # Increment major version for main branch
            latest_main = self._get_latest_main_version(experiment_id)
            if latest_main:
                major = int(latest_main.version_number.split('.')[0])
                version_number = f"{major + 1}.0"
            else:
                version_number = "2.0"
        else:
            # Use branch naming for non-main branches
            version_number = f"{parent.version_number}-{branch_name}"
        
        version_id = f"{experiment_id}_v{version_number}"
        
        # Create metadata
        metadata = VersionMetadata(
            author=author,
            message=message,
            parent_versions=[parent_version],
            is_branch=(branch_name != "main")
        )
        
        # Create version
        version = ExperimentVersion(
            version_id=version_id,
            experiment_id=experiment_id,
            version_number=version_number,
            type=VersionType.BRANCH if branch_name != "main" else VersionType.INITIAL,
            config=config.copy(),
            metadata=metadata
        )
        
        # Update relationships
        self.versions[version_id] = version
        
        if parent_version in self.version_tree:
            self.version_tree[parent_version].append(version_id)
        self.version_tree[version_id] = []
        
        # Update parent's children
        if parent_version in self.versions:
            self.versions[parent_version].metadata.child_versions.append(version_id)
        
        # Update branch
        branch_key = f"{experiment_id}_{branch_name}"
        if branch_key in self.branches:
            self.branches[branch_key].head_version_id = version_id
            self.branches[branch_key].versions.append(version_id)
        else:
            # Create new branch
            new_branch = VersionBranch(
                branch_name=branch_name,
                base_version_id=parent_version,
                head_version_id=version_id,
                created_by=author,
                description=f"Branch {branch_name}",
                versions=[version_id]
            )
            self.branches[branch_key] = new_branch
        
        # Update experiment head if this is main branch
        if branch_name == "main":
            self.experiment_heads[experiment_id] = version_id
        
        logger.info(f"Created version {version_id} on branch {branch_name}")
        return version
    
    def create_branch(self, experiment_id: str, branch_name: str, 
                     base_version: Optional[str] = None, author: str = "",
                     description: str = "") -> VersionBranch:
        """Create a new branch from a base version."""
        
        # Get base version
        if not base_version:
            base_version = self.experiment_heads.get(experiment_id)
            
        if not base_version or base_version not in self.versions:
            raise ValueError(f"Base version {base_version} not found")
        
        branch_key = f"{experiment_id}_{branch_name}"
        if branch_key in self.branches:
            raise ValueError(f"Branch {branch_name} already exists")
        
        # Create branch
        branch = VersionBranch(
            branch_name=branch_name,
            base_version_id=base_version,
            head_version_id=base_version,
            created_by=author,
            description=description,
            versions=[base_version]
        )
        
        self.branches[branch_key] = branch
        
        logger.info(f"Created branch {branch_name} for experiment {experiment_id}")
        return branch
    
    def get_version_diff(self, version1_id: str, version2_id: str) -> ExperimentDiff:
        """Calculate differences between two versions."""
        
        if version1_id not in self.versions or version2_id not in self.versions:
            raise ValueError("One or both versions not found")
        
        v1 = self.versions[version1_id]
        v2 = self.versions[version2_id]
        
        diff = ExperimentDiff()
        
        # Compare configurations
        v1_params = set(v1.config.keys())
        v2_params = set(v2.config.keys())
        
        # Added parameters
        for param in v2_params - v1_params:
            diff.added_parameters[param] = v2.config[param]
        
        # Removed parameters
        for param in v1_params - v2_params:
            diff.removed_parameters[param] = v1.config[param]
        
        # Modified parameters
        for param in v1_params & v2_params:
            if v1.config[param] != v2.config[param]:
                diff.modified_parameters[param] = {
                    "old": v1.config[param],
                    "new": v2.config[param]
                }
        
        # Metadata changes
        if v1.metadata.author != v2.metadata.author:
            diff.metadata_changes["author"] = {
                "old": v1.metadata.author,
                "new": v2.metadata.author
            }
        
        return diff
    
    def create_merge_request(self, source_version_id: str, target_version_id: str,
                           strategy: MergeStrategy, author: str, message: str = "",
                           auto_merge: bool = False) -> MergeRequest:
        """Create a merge request between two versions."""
        
        if source_version_id not in self.versions or target_version_id not in self.versions:
            raise ValueError("Source or target version not found")
        
        # Generate merge request ID
        merge_id = f"merge_{source_version_id}_to_{target_version_id}_{datetime.now().timestamp()}"
        
        # Detect conflicts
        conflicts = self._detect_merge_conflicts(source_version_id, target_version_id)
        
        merge_request = MergeRequest(
            id=merge_id,
            source_version_id=source_version_id,
            target_version_id=target_version_id,
            strategy=strategy,
            author=author,
            message=message,
            conflicts=conflicts,
            auto_merge=auto_merge and len(conflicts) == 0
        )
        
        self.merge_requests[merge_id] = merge_request
        
        # Auto-merge if requested and no conflicts
        if auto_merge and len(conflicts) == 0:
            self.execute_merge(merge_id)
        
        logger.info(f"Created merge request {merge_id} with {len(conflicts)} conflicts")
        return merge_request
    
    def _detect_merge_conflicts(self, source_version_id: str, 
                               target_version_id: str) -> List[MergeConflict]:
        """Detect conflicts between two versions."""
        conflicts = []
        
        source = self.versions[source_version_id]
        target = self.versions[target_version_id]
        
        # Find common ancestor
        common_ancestor_id = self._find_common_ancestor(source_version_id, target_version_id)
        if not common_ancestor_id:
            # No common ancestor, treat as full conflict resolution needed
            return conflicts
        
        base = self.versions[common_ancestor_id]
        
        # Check for conflicting changes
        for param in source.config:
            if param in target.config and param in base.config:
                # All three versions have this parameter
                if (source.config[param] != base.config[param] and 
                    target.config[param] != base.config[param] and
                    source.config[param] != target.config[param]):
                    # Both versions modified the same parameter differently
                    conflict = MergeConflict(
                        parameter_path=param,
                        base_value=base.config[param],
                        source_value=source.config[param],
                        target_value=target.config[param]
                    )
                    conflicts.append(conflict)
        
        return conflicts
    
    def _find_common_ancestor(self, version1_id: str, version2_id: str) -> Optional[str]:
        """Find the common ancestor of two versions."""
        # Simple implementation - find first common parent
        # TODO: Implement proper graph traversal for complex cases
        
        v1_ancestors = self._get_ancestors(version1_id)
        v2_ancestors = self._get_ancestors(version2_id)
        
        # Find intersection
        common = v1_ancestors & v2_ancestors
        if not common:
            return None
        
        # Return the most recent common ancestor
        # For now, just return the first one found
        return list(common)[0]
    
    def _get_ancestors(self, version_id: str) -> Set[str]:
        """Get all ancestors of a version."""
        ancestors = set()
        
        def traverse(vid: str):
            if vid in self.versions:
                version = self.versions[vid]
                for parent in version.metadata.parent_versions:
                    ancestors.add(parent)
                    traverse(parent)
        
        traverse(version_id)
        return ancestors
    
    def execute_merge(self, merge_request_id: str) -> ExperimentVersion:
        """Execute a merge request."""
        
        if merge_request_id not in self.merge_requests:
            raise ValueError(f"Merge request {merge_request_id} not found")
        
        merge_req = self.merge_requests[merge_request_id]
        
        # Check if all conflicts are resolved
        unresolved = [c for c in merge_req.conflicts if not c.resolved]
        if unresolved and merge_req.strategy == MergeStrategy.MANUAL:
            raise ValueError(f"Cannot merge: {len(unresolved)} unresolved conflicts")
        
        source = self.versions[merge_req.source_version_id]
        target = self.versions[merge_req.target_version_id]
        
        # Create merged configuration
        merged_config = self._merge_configurations(source, target, merge_req)
        
        # Create new version
        merged_version = self.create_version(
            experiment_id=source.experiment_id,
            config=merged_config,
            author=merge_req.author,
            message=f"Merge: {merge_req.message}",
            parent_version=merge_req.target_version_id
        )
        
        # Update metadata
        merged_version.type = VersionType.MERGE
        merged_version.metadata.is_merge = True
        merged_version.metadata.merge_strategy = merge_req.strategy
        merged_version.metadata.parent_versions = [
            merge_req.source_version_id, 
            merge_req.target_version_id
        ]
        
        # Update merge request status
        merge_req.status = "merged"
        
        logger.info(f"Executed merge request {merge_request_id}")
        return merged_version
    
    def _merge_configurations(self, source: ExperimentVersion, 
                            target: ExperimentVersion, 
                            merge_req: MergeRequest) -> Dict[str, Any]:
        """Merge configurations based on strategy."""
        
        if merge_req.strategy == MergeStrategy.OVERWRITE:
            return source.config.copy()
        
        elif merge_req.strategy == MergeStrategy.MERGE_PARAMETERS:
            merged = target.config.copy()
            merged.update(source.config)
            return merged
        
        elif merge_req.strategy == MergeStrategy.MANUAL:
            merged = target.config.copy()
            
            # Apply resolved conflicts
            for conflict in merge_req.conflicts:
                if conflict.resolved and conflict.resolution is not None:
                    merged[conflict.parameter_path] = conflict.resolution
            
            return merged
        
        elif merge_req.strategy == MergeStrategy.KEEP_BOTH:
            # Create a hybrid configuration
            merged = target.config.copy()
            
            # Add source parameters with prefix
            for key, value in source.config.items():
                if key in merged and merged[key] != value:
                    merged[f"source_{key}"] = value
                    merged[f"target_{key}"] = merged[key]
                    del merged[key]
                else:
                    merged[key] = value
            
            return merged
        
        else:
            raise ValueError(f"Unknown merge strategy: {merge_req.strategy}")
    
    def rollback_to_version(self, experiment_id: str, target_version_id: str,
                           author: str, message: str = "") -> ExperimentVersion:
        """Rollback an experiment to a previous version."""
        
        if target_version_id not in self.versions:
            raise ValueError(f"Target version {target_version_id} not found")
        
        target = self.versions[target_version_id]
        if target.experiment_id != experiment_id:
            raise ValueError("Target version is not for this experiment")
        
        # Create rollback version
        rollback_version = self.create_version(
            experiment_id=experiment_id,
            config=target.config.copy(),
            author=author,
            message=f"Rollback to {target_version_id}: {message}"
        )
        
        rollback_version.type = VersionType.ROLLBACK
        rollback_version.metadata.tags.append(f"rollback-to-{target_version_id}")
        
        logger.info(f"Rolled back experiment {experiment_id} to version {target_version_id}")
        return rollback_version
    
    def _get_latest_main_version(self, experiment_id: str) -> Optional[ExperimentVersion]:
        """Get the latest version on the main branch."""
        main_branch_key = f"{experiment_id}_main"
        if main_branch_key not in self.branches:
            return None
        
        branch = self.branches[main_branch_key]
        head_id = branch.head_version_id
        
        return self.versions.get(head_id)
    
    def get_version_tree(self, experiment_id: str) -> Dict[str, Any]:
        """Get the version tree for an experiment."""
        experiment_versions = {
            vid: v for vid, v in self.versions.items() 
            if v.experiment_id == experiment_id
        }
        
        # Build tree structure
        tree = {}
        for version_id, version in experiment_versions.items():
            tree[version_id] = {
                "version": version,
                "children": self.version_tree.get(version_id, []),
                "parents": version.metadata.parent_versions
            }
        
        return tree
    
    def get_experiment_branches(self, experiment_id: str) -> List[VersionBranch]:
        """Get all branches for an experiment."""
        return [
            branch for branch_key, branch in self.branches.items()
            if branch_key.startswith(f"{experiment_id}_")
        ]
    
    def get_version_stats(self, experiment_id: Optional[str] = None) -> Dict[str, Any]:
        """Get version control statistics."""
        versions = list(self.versions.values())
        if experiment_id:
            versions = [v for v in versions if v.experiment_id == experiment_id]
        
        total_versions = len(versions)
        branches = len([b for b in self.branches.values() 
                       if not experiment_id or b.branch_name.startswith(experiment_id)])
        merges = len([v for v in versions if v.type == VersionType.MERGE])
        rollbacks = len([v for v in versions if v.type == VersionType.ROLLBACK])
        
        return {
            "total_versions": total_versions,
            "total_branches": branches,
            "total_merges": merges,
            "total_rollbacks": rollbacks,
            "pending_merge_requests": len([mr for mr in self.merge_requests.values() 
                                         if mr.status == "pending"])
        }


# Global version control instance
version_control = ExperimentVersionControl()
