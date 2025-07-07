"""
Phase 4B: 3D Visualization Engine
Interactive 3D visualizations for experiment data using Three.js integration.
"""

import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class Point3D:
    """3D point representation."""
    x: float
    y: float
    z: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FitnessLandscape3D:
    """3D fitness landscape data structure."""
    points: List[Point3D]
    experiment_id: str
    generation: int
    timestamp: datetime
    bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # min, max for each axis
    color_mapping: str = "fitness"  # "fitness", "generation", "diversity"


@dataclass
class PopulationCloud3D:
    """3D population visualization data."""
    individuals: List[Point3D]
    experiment_id: str
    generation: int
    elite_indices: List[int] = field(default_factory=list)
    diversity_score: float = 0.0
    best_fitness: float = 0.0


class Visualization3DEngine:
    """Advanced 3D visualization engine for evolutionary data."""
    
    def __init__(self):
        self.landscapes: Dict[str, List[FitnessLandscape3D]] = {}
        self.population_clouds: Dict[str, List[PopulationCloud3D]] = {}
        self.animation_frames: Dict[str, List[Dict[str, Any]]] = {}
    
    def generate_fitness_landscape(self, experiment_id: str, fitness_data: List[List[float]], 
                                 generation: int) -> FitnessLandscape3D:
        """Generate 3D fitness landscape from experiment data."""
        try:
            # Convert fitness data to 3D points
            points = []
            
            # Create a grid-based landscape representation
            if len(fitness_data) > 0:
                # Use PCA-like dimensionality reduction for visualization
                population_size = len(fitness_data[0]) if fitness_data else 0
                
                for gen_idx, gen_fitness in enumerate(fitness_data[-20:]):  # Last 20 generations
                    for ind_idx, fitness in enumerate(gen_fitness):
                        # Map to 3D space using mathematical transformations
                        x = ind_idx / max(1, population_size - 1)  # Normalize individual index
                        y = gen_idx / max(1, len(fitness_data[-20:]) - 1)  # Normalize generation
                        z = fitness  # Fitness as height
                        
                        points.append(Point3D(
                            x=x,
                            y=y, 
                            z=z,
                            metadata={
                                "generation": generation - len(fitness_data[-20:]) + gen_idx,
                                "individual": ind_idx,
                                "fitness": fitness
                            }
                        ))
            
            # Calculate bounds
            if points:
                bounds = {
                    "x": (min(p.x for p in points), max(p.x for p in points)),
                    "y": (min(p.y for p in points), max(p.y for p in points)),
                    "z": (min(p.z for p in points), max(p.z for p in points))
                }
            else:
                bounds = {"x": (0, 1), "y": (0, 1), "z": (0, 1)}
            
            landscape = FitnessLandscape3D(
                points=points,
                experiment_id=experiment_id,
                generation=generation,
                timestamp=datetime.now(),
                bounds=bounds
            )
            
            # Store landscape
            if experiment_id not in self.landscapes:
                self.landscapes[experiment_id] = []
            self.landscapes[experiment_id].append(landscape)
            
            # Keep only last 100 landscapes per experiment
            if len(self.landscapes[experiment_id]) > 100:
                self.landscapes[experiment_id] = self.landscapes[experiment_id][-100:]
            
            return landscape
            
        except Exception as e:
            logger.error(f"Failed to generate fitness landscape: {e}")
            return FitnessLandscape3D(
                points=[],
                experiment_id=experiment_id,
                generation=generation,
                timestamp=datetime.now()
            )
    
    def generate_population_cloud(self, experiment_id: str, population_data: List[Dict[str, Any]], 
                                generation: int) -> PopulationCloud3D:
        """Generate 3D population cloud visualization."""
        try:
            individuals = []
            elite_indices = []
            
            # Sort by fitness to identify elites
            sorted_pop = sorted(enumerate(population_data), key=lambda x: x[1].get('fitness', 0), reverse=True)
            elite_count = max(1, len(population_data) // 10)  # Top 10%
            
            for idx, (original_idx, individual) in enumerate(sorted_pop):
                # Use individual's genetic parameters for 3D positioning
                genome = individual.get('genome', [])
                fitness = individual.get('fitness', 0)
                
                # Map genome to 3D space (simplified PCA-like mapping)
                if len(genome) >= 3:
                    x = genome[0] if isinstance(genome[0], (int, float)) else 0
                    y = genome[1] if isinstance(genome[1], (int, float)) else 0
                    z = fitness
                elif len(genome) >= 2:
                    x = genome[0] if isinstance(genome[0], (int, float)) else 0
                    y = genome[1] if isinstance(genome[1], (int, float)) else 0
                    z = fitness
                else:
                    # Fallback: use index-based positioning
                    x = original_idx / max(1, len(population_data) - 1)
                    y = fitness
                    z = individual.get('diversity_contribution', 0.5)
                
                individuals.append(Point3D(
                    x=float(x),
                    y=float(y),
                    z=float(z),
                    metadata={
                        "individual_id": original_idx,
                        "fitness": fitness,
                        "generation": generation,
                        "is_elite": idx < elite_count
                    }
                ))
                
                if idx < elite_count:
                    elite_indices.append(len(individuals) - 1)
            
            # Calculate diversity score
            diversity_score = 0.0
            if len(individuals) > 1:
                # Simple diversity measure based on position spread
                positions = [(p.x, p.y, p.z) for p in individuals]
                center_x = sum(p[0] for p in positions) / len(positions)
                center_y = sum(p[1] for p in positions) / len(positions)
                center_z = sum(p[2] for p in positions) / len(positions)
                
                total_distance = sum(
                    ((p[0] - center_x)**2 + (p[1] - center_y)**2 + (p[2] - center_z)**2)**0.5
                    for p in positions
                )
                diversity_score = total_distance / len(positions)
            
            best_fitness = max((ind.get('fitness', 0) for ind in population_data), default=0)
            
            cloud = PopulationCloud3D(
                individuals=individuals,
                experiment_id=experiment_id,
                generation=generation,
                elite_indices=elite_indices,
                diversity_score=diversity_score,
                best_fitness=best_fitness
            )
            
            # Store cloud
            if experiment_id not in self.population_clouds:
                self.population_clouds[experiment_id] = []
            self.population_clouds[experiment_id].append(cloud)
            
            # Keep only last 50 clouds per experiment
            if len(self.population_clouds[experiment_id]) > 50:
                self.population_clouds[experiment_id] = self.population_clouds[experiment_id][-50:]
            
            return cloud
            
        except Exception as e:
            logger.error(f"Failed to generate population cloud: {e}")
            return PopulationCloud3D(
                individuals=[],
                experiment_id=experiment_id,
                generation=generation
            )
    
    def create_evolution_animation(self, experiment_id: str, frame_count: int = 50) -> List[Dict[str, Any]]:
        """Create animation frames showing evolution over time."""
        try:
            if experiment_id not in self.population_clouds:
                return []
            
            clouds = self.population_clouds[experiment_id]
            if not clouds:
                return []
            
            # Sample clouds for animation frames
            total_clouds = len(clouds)
            if total_clouds <= frame_count:
                selected_clouds = clouds
            else:
                # Sample evenly across available clouds
                indices = [int(i * total_clouds / frame_count) for i in range(frame_count)]
                selected_clouds = [clouds[i] for i in indices]
            
            animation_frames = []
            for cloud in selected_clouds:
                frame = {
                    "generation": cloud.generation,
                    "timestamp": cloud.timestamp.isoformat() if hasattr(cloud, 'timestamp') else datetime.now().isoformat(),
                    "individuals": [
                        {
                            "x": ind.x,
                            "y": ind.y,
                            "z": ind.z,
                            "fitness": ind.metadata.get("fitness", 0),
                            "is_elite": ind.metadata.get("is_elite", False)
                        }
                        for ind in cloud.individuals
                    ],
                    "diversity_score": cloud.diversity_score,
                    "best_fitness": cloud.best_fitness,
                    "elite_count": len(cloud.elite_indices)
                }
                animation_frames.append(frame)
            
            # Store animation
            self.animation_frames[experiment_id] = animation_frames
            
            return animation_frames
            
        except Exception as e:
            logger.error(f"Failed to create evolution animation: {e}")
            return []
    
    def export_for_threejs(self, experiment_id: str, visualization_type: str = "landscape") -> Dict[str, Any]:
        """Export visualization data in Three.js compatible format."""
        try:
            if visualization_type == "landscape":
                if experiment_id not in self.landscapes or not self.landscapes[experiment_id]:
                    return {"error": "No landscape data available"}
                
                latest_landscape = self.landscapes[experiment_id][-1]
                
                return {
                    "type": "fitness_landscape",
                    "experiment_id": experiment_id,
                    "generation": latest_landscape.generation,
                    "points": [
                        {
                            "x": point.x,
                            "y": point.y,
                            "z": point.z,
                            "metadata": point.metadata
                        }
                        for point in latest_landscape.points
                    ],
                    "bounds": latest_landscape.bounds,
                    "color_mapping": latest_landscape.color_mapping,
                    "timestamp": latest_landscape.timestamp.isoformat()
                }
            
            elif visualization_type == "population":
                if experiment_id not in self.population_clouds or not self.population_clouds[experiment_id]:
                    return {"error": "No population data available"}
                
                latest_cloud = self.population_clouds[experiment_id][-1]
                
                return {
                    "type": "population_cloud",
                    "experiment_id": experiment_id,
                    "generation": latest_cloud.generation,
                    "individuals": [
                        {
                            "x": ind.x,
                            "y": ind.y,
                            "z": ind.z,
                            "metadata": ind.metadata
                        }
                        for ind in latest_cloud.individuals
                    ],
                    "elite_indices": latest_cloud.elite_indices,
                    "diversity_score": latest_cloud.diversity_score,
                    "best_fitness": latest_cloud.best_fitness
                }
            
            elif visualization_type == "animation":
                if experiment_id not in self.animation_frames:
                    # Generate animation if not exists
                    self.create_evolution_animation(experiment_id)
                
                return {
                    "type": "evolution_animation",
                    "experiment_id": experiment_id,
                    "frames": self.animation_frames.get(experiment_id, []),
                    "frame_count": len(self.animation_frames.get(experiment_id, [])),
                    "duration_ms": len(self.animation_frames.get(experiment_id, [])) * 200  # 200ms per frame
                }
            
            else:
                return {"error": f"Unknown visualization type: {visualization_type}"}
                
        except Exception as e:
            logger.error(f"Failed to export for Three.js: {e}")
            return {"error": str(e)}
    
    def get_experiment_visualizations(self, experiment_id: str) -> Dict[str, Any]:
        """Get all available visualizations for an experiment."""
        return {
            "experiment_id": experiment_id,
            "available_visualizations": {
                "fitness_landscape": experiment_id in self.landscapes and len(self.landscapes[experiment_id]) > 0,
                "population_cloud": experiment_id in self.population_clouds and len(self.population_clouds[experiment_id]) > 0,
                "evolution_animation": experiment_id in self.animation_frames and len(self.animation_frames[experiment_id]) > 0
            },
            "data_counts": {
                "landscapes": len(self.landscapes.get(experiment_id, [])),
                "population_clouds": len(self.population_clouds.get(experiment_id, [])),
                "animation_frames": len(self.animation_frames.get(experiment_id, []))
            },
            "last_updated": datetime.now().isoformat()
        }
    
    def clear_experiment_data(self, experiment_id: str):
        """Clear all visualization data for an experiment."""
        if experiment_id in self.landscapes:
            del self.landscapes[experiment_id]
        if experiment_id in self.population_clouds:
            del self.population_clouds[experiment_id]
        if experiment_id in self.animation_frames:
            del self.animation_frames[experiment_id]
        
        logger.info(f"Cleared visualization data for experiment {experiment_id}")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get visualization system statistics."""
        total_landscapes = sum(len(landscapes) for landscapes in self.landscapes.values())
        total_clouds = sum(len(clouds) for clouds in self.population_clouds.values())
        total_animations = len(self.animation_frames)
        
        return {
            "total_experiments": len(set(list(self.landscapes.keys()) + list(self.population_clouds.keys()))),            "total_landscapes": total_landscapes,
            "total_population_clouds": total_clouds,
            "total_animations": total_animations,
            "memory_usage_estimate_mb": (total_landscapes * 0.1 + total_clouds * 0.2 + total_animations * 1.0),
            "experiments_with_data": list(set(list(self.landscapes.keys()) + list(self.population_clouds.keys())))
        }
    
    def create_fitness_landscape_3d(self, experiment_id: str, generation: int, 
                                   data: List[Any], resolution: int = 50) -> FitnessLandscape3D:
        """Create 3D fitness landscape from database results."""
        try:
            points = []
            
            # Extract fitness data from database results
            fitness_values = [row[2] for row in data]  # fitness is 3rd column
            
            if fitness_values:
                # Create grid-based landscape
                grid_size = int(resolution ** 0.5)  # Square grid
                
                for i in range(grid_size):
                    for j in range(grid_size):
                        # Map grid position to landscape
                        x = i / max(1, grid_size - 1)
                        y = j / max(1, grid_size - 1)
                        
                        # Interpolate fitness value for this grid position
                        if len(fitness_values) > 0:
                            # Simple interpolation based on position
                            index = int((i * grid_size + j) % len(fitness_values))
                            z = fitness_values[index]
                        else:
                            z = 0.0
                        
                        points.append(Point3D(
                            x=float(x),
                            y=float(y),
                            z=float(z),
                            metadata={
                                "grid_x": i,
                                "grid_y": j,
                                "generation": generation,
                                "fitness": z
                            }
                        ))
            
            # Calculate bounds
            if points:
                bounds = {
                    "x": (min(p.x for p in points), max(p.x for p in points)),
                    "y": (min(p.y for p in points), max(p.y for p in points)),
                    "z": (min(p.z for p in points), max(p.z for p in points))
                }
            else:
                bounds = {"x": (0, 1), "y": (0, 1), "z": (0, 1)}
            
            landscape = FitnessLandscape3D(
                points=points,
                experiment_id=experiment_id,
                generation=generation,
                timestamp=datetime.now(),
                bounds=bounds
            )
            
            return landscape
            
        except Exception as e:
            logger.error(f"Error creating 3D fitness landscape: {e}")
            return FitnessLandscape3D(
                points=[],
                experiment_id=experiment_id,
                generation=generation,
                timestamp=datetime.now()
            )
    
    def create_population_cloud_3d(self, experiment_id: str, generation: int,
                                 population_data: List[Any], show_elite: bool = True) -> PopulationCloud3D:
        """Create 3D population cloud from database results."""
        try:
            individuals = []
            elite_indices = []
            
            # Extract data from database results
            population = []
            for i, row in enumerate(population_data):
                individual_data = row[0]  # individual_data column
                fitness = row[1]  # fitness column
                
                # Parse individual data (could be JSON or other format)
                try:
                    if isinstance(individual_data, str):
                        import json
                        genome = json.loads(individual_data)
                    else:
                        genome = individual_data if isinstance(individual_data, list) else [individual_data]
                except:
                    genome = [i]  # Fallback to index
                
                population.append({
                    "genome": genome,
                    "fitness": fitness,
                    "index": i
                })
            
            # Sort by fitness for elite identification
            sorted_pop = sorted(population, key=lambda x: x["fitness"], reverse=True)
            elite_count = max(1, len(population) // 10) if show_elite else 0
            
            for idx, individual in enumerate(sorted_pop):
                genome = individual["genome"]
                fitness = individual["fitness"]
                
                # Map to 3D space
                if len(genome) >= 3:
                    x = float(genome[0]) if isinstance(genome[0], (int, float)) else 0.0
                    y = float(genome[1]) if isinstance(genome[1], (int, float)) else 0.0
                    z = float(fitness)
                elif len(genome) >= 2:
                    x = float(genome[0]) if isinstance(genome[0], (int, float)) else 0.0
                    y = float(genome[1]) if isinstance(genome[1], (int, float)) else 0.0
                    z = float(fitness)
                else:
                    # Fallback positioning
                    x = individual["index"] / max(1, len(population) - 1)
                    y = float(fitness)
                    z = 0.5
                
                individuals.append(Point3D(
                    x=x,
                    y=y,
                    z=z,
                    metadata={
                        "individual_id": individual["index"],
                        "fitness": fitness,
                        "generation": generation,
                        "is_elite": idx < elite_count
                    }
                ))
                
                if idx < elite_count:
                    elite_indices.append(len(individuals) - 1)
            
            # Calculate diversity
            diversity_score = 0.0
            if len(individuals) > 1:
                positions = [(p.x, p.y, p.z) for p in individuals]
                center_x = sum(p[0] for p in positions) / len(positions)
                center_y = sum(p[1] for p in positions) / len(positions)
                center_z = sum(p[2] for p in positions) / len(positions)
                
                total_distance = sum(
                    ((p[0] - center_x)**2 + (p[1] - center_y)**2 + (p[2] - center_z)**2)**0.5
                    for p in positions
                )
                diversity_score = total_distance / len(positions)
            
            best_fitness = max((ind["fitness"] for ind in population), default=0)
            
            return PopulationCloud3D(
                individuals=individuals,
                experiment_id=experiment_id,
                generation=generation,
                elite_indices=elite_indices,
                diversity_score=diversity_score,
                best_fitness=best_fitness
            )
            
        except Exception as e:
            logger.error(f"Error creating population cloud 3D: {e}")
            return PopulationCloud3D(
                individuals=[],
                experiment_id=experiment_id,
                generation=generation
            )
    
    def create_evolution_trajectory_3d(self, experiment_id: str, trajectory_data: List[Any]) -> Dict[str, Any]:
        """Create 3D evolution trajectory from database results."""
        try:
            trajectory_points = []
            
            for row in trajectory_data:
                generation = row[0]
                avg_fitness = row[1]
                best_fitness = row[2]
                worst_fitness = row[3]
                
                # Create trajectory points
                trajectory_points.append({
                    "generation": generation,
                    "x": generation,
                    "y": avg_fitness,
                    "z": best_fitness,
                    "metadata": {
                        "avg_fitness": avg_fitness,
                        "best_fitness": best_fitness,
                        "worst_fitness": worst_fitness
                    }
                })
            
            return {
                "experiment_id": experiment_id,
                "type": "evolution_trajectory",
                "points": trajectory_points,
                "bounds": {
                    "x": (min(p["x"] for p in trajectory_points), max(p["x"] for p in trajectory_points)),
                    "y": (min(p["y"] for p in trajectory_points), max(p["y"] for p in trajectory_points)),
                    "z": (min(p["z"] for p in trajectory_points), max(p["z"] for p in trajectory_points))
                } if trajectory_points else {"x": (0, 1), "y": (0, 1), "z": (0, 1)},
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating evolution trajectory: {e}")
            return {
                "experiment_id": experiment_id,
                "type": "evolution_trajectory",
                "points": [],
                "error": str(e)
            }
    
    def export_to_threejs(self, landscape: FitnessLandscape3D) -> Dict[str, Any]:
        """Export fitness landscape to Three.js format."""
        try:
            return {
                "type": "fitness_landscape",
                "geometry": {
                    "vertices": [
                        {"x": point.x, "y": point.y, "z": point.z}
                        for point in landscape.points
                    ],
                    "faces": self._generate_faces(landscape.points),
                    "colors": self._generate_colors(landscape.points, landscape.color_mapping)
                },
                "bounds": landscape.bounds,
                "metadata": {
                    "experiment_id": landscape.experiment_id,
                    "generation": landscape.generation,
                    "point_count": len(landscape.points),
                    "color_mapping": landscape.color_mapping
                }
            }
        except Exception as e:
            logger.error(f"Error exporting to Three.js: {e}")
            return {"error": str(e)}
    
    def export_population_to_threejs(self, cloud: PopulationCloud3D) -> Dict[str, Any]:
        """Export population cloud to Three.js format."""
        try:
            return {
                "type": "population_cloud",
                "points": [
                    {
                        "position": {"x": ind.x, "y": ind.y, "z": ind.z},
                        "metadata": ind.metadata,
                        "color": self._get_individual_color(ind)
                    }
                    for ind in cloud.individuals
                ],
                "elite_indices": cloud.elite_indices,
                "stats": {
                    "diversity_score": cloud.diversity_score,
                    "best_fitness": cloud.best_fitness,
                    "population_size": len(cloud.individuals),
                    "elite_count": len(cloud.elite_indices)
                },
                "metadata": {
                    "experiment_id": cloud.experiment_id,
                    "generation": cloud.generation
                }
            }
        except Exception as e:
            logger.error(f"Error exporting population to Three.js: {e}")
            return {"error": str(e)}
    
    def _generate_faces(self, points: List[Point3D]) -> List[Dict[str, Any]]:
        """Generate triangular faces for mesh rendering."""
        faces = []
        # Simple grid-based face generation
        grid_size = int(len(points) ** 0.5)
        
        for i in range(grid_size - 1):
            for j in range(grid_size - 1):
                # Calculate indices for quad
                idx1 = i * grid_size + j
                idx2 = i * grid_size + (j + 1)
                idx3 = (i + 1) * grid_size + j
                idx4 = (i + 1) * grid_size + (j + 1)
                
                # Create two triangles from quad
                if all(idx < len(points) for idx in [idx1, idx2, idx3, idx4]):
                    faces.extend([
                        {"a": idx1, "b": idx2, "c": idx3},
                        {"a": idx2, "b": idx4, "c": idx3}
                    ])
        
        return faces
    
    def _generate_colors(self, points: List[Point3D], color_mapping: str) -> List[str]:
        """Generate colors for points based on mapping type."""
        colors = []
        
        if color_mapping == "fitness":
            # Map fitness to color gradient (red = low, green = high)
            fitness_values = [p.z for p in points]
            if fitness_values:
                min_fitness = min(fitness_values)
                max_fitness = max(fitness_values)
                fitness_range = max_fitness - min_fitness if max_fitness != min_fitness else 1
                
                for point in points:
                    normalized = (point.z - min_fitness) / fitness_range
                    # Create RGB color (red to green gradient)
                    red = int(255 * (1 - normalized))
                    green = int(255 * normalized)
                    colors.append(f"rgb({red}, {green}, 0)")
            else:
                colors = ["rgb(128, 128, 128)"] * len(points)
        else:
            # Default color
            colors = ["rgb(100, 150, 255)"] * len(points)
        
        return colors
    
    def _get_individual_color(self, individual: Point3D) -> str:
        """Get color for individual based on properties."""
        if individual.metadata.get("is_elite", False):
            return "rgb(255, 215, 0)"  # Gold for elite
        else:
            # Color based on fitness
            fitness = individual.metadata.get("fitness", 0)
            normalized = min(1.0, max(0.0, fitness / 100))  # Assume max fitness ~100
            blue = int(255 * (1 - normalized))
            red = int(255 * normalized)
            return f"rgb({red}, 0, {blue})"
