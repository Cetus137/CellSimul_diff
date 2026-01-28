"""
Complete inference pipeline for generating synthetic cell images.

This module provides end-to-end functionality for:
1. Generating realistic cell centres
2. Creating conditioning maps
3. Generating images from trained diffusion model
"""

import torch
import numpy as np
import yaml
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import logging
import sys

sys.path.append(str(Path(__file__).parent.parent))

from sampling.generate_centres import (
    generate_random_centres_poisson,
    generate_random_centres_simple,
    generate_centres_from_training_distribution
)
from sampling.sample_from_centres import load_model, sample_from_centres
from preprocessing.generate_condition_maps import generate_conditioning_maps
from utils.normalization import to_zero_one

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CellSynthesizer:
    """
    End-to-end pipeline for synthetic cell image generation.
    
    Usage:
        >>> synthesizer = CellSynthesizer(checkpoint_path="checkpoints/best.pt")
        >>> images, centres_list = synthesizer.generate_batch(
        ...     num_samples=10,
        ...     method='poisson'
        ... )
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        model_config: str = "configs/model.yaml",
        data_config: str = "configs/data_hpc.yaml",
        device: str = "cuda"
    ):
        """
        Initialize the synthesizer with a trained model.
        
        Args:
            checkpoint_path: Path to model checkpoint
            model_config: Path to model configuration
            data_config: Path to data configuration (for preprocessing params)
            device: Device to run inference on
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        if device == "cuda" and self.device == "cpu":
            logger.warning("CUDA requested but not available, using CPU")
        
        # Load configurations
        with open(model_config, 'r') as f:
            self.model_config = yaml.safe_load(f)
        
        with open(data_config, 'r') as f:
            self.data_config = yaml.safe_load(f)
        
        # Load model
        logger.info(f"Loading model from {checkpoint_path}...")
        self.model = load_model(checkpoint_path, model_config, self.device)
        self.model.eval()
        
        # Get preprocessing parameters
        self.image_size = self.data_config['preprocessing']['patch_size']
        self.heatmap_sigma = self.data_config['preprocessing']['centre_heatmap_sigma']
        
        logger.info(f"Synthesizer initialized on {self.device}")
        logger.info(f"Image size: {self.image_size}x{self.image_size}")
    
    def generate_centres(
        self,
        method: str = 'poisson',
        num_cells: Optional[int] = None,
        density: Optional[float] = None,
        min_distance: float = 12.0,
        mean_cells: float = 20.0,
        std_cells: float = 8.0,
        centres_file: Optional[str] = None,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate cell centres using specified method.
        
        Args:
            method: Generation method ('simple', 'poisson', 'training_dist')
            num_cells: Number of cells (for 'simple' method)
            density: Cell density per pixel (for 'poisson' method)
            min_distance: Minimum distance between centres
            mean_cells: Mean cell count (for 'training_dist' method)
            std_cells: Std dev of cell count (for 'training_dist' method)
            seed: Random seed
        
        Returns:
            centres: Array of shape (N, 2) with (y, x) coordinates
        """
        image_shape = (self.image_size, self.image_size)
        
        if method == 'simple':
            if num_cells is None:
                num_cells = 20
            return generate_random_centres_simple(
                image_shape=image_shape,
                num_cells=num_cells,
                border_margin=20,
                seed=seed
            )
        
        elif method == 'poisson':
            if density is None:
                density = 0.0003  # ~20 cells in 256x256
            return generate_random_centres_poisson(
                image_shape=image_shape,
                density=density,
                min_distance=min_distance,
                border_margin=20,
                seed=seed
            )
        
        elif method == 'training_dist':
            return generate_centres_from_training_distribution(
                image_shape=image_shape,
                mean_cells=mean_cells,
                std_cells=std_cells,
                mean_min_dist=min_distance,
                border_margin=20,
                seed=seed
            )
        
        elif method == 'from_file':
            if centres_file is None:
                raise ValueError("centres_file must be provided when method='from_file'")
            centres = np.load(centres_file)
            logger.info(f"Loaded {len(centres)} centres from {centres_file}")
            return centres
        
        else:
            raise ValueError(f"Unknown method: {method}. Use 'simple', 'poisson', 'training_dist', or 'from_file'")
    
    def generate_conditioning(
        self,
        centres: np.ndarray
    ) -> np.ndarray:
        """
        Generate conditioning maps from centres.
        
        Args:
            centres: Cell centres of shape (N, 2)
        
        Returns:
            conditioning: Conditioning maps of shape (3, H, W)
        """
        image_shape = (self.image_size, self.image_size)
        
        conditioning = generate_conditioning_maps(
            centres,
            image_shape,
            heatmap_sigma=self.heatmap_sigma,
            boundary_sigma=2.0,
            boundary_method='entropy',
            distance_percentile=95.0
        )
        
        return conditioning
    
    def generate_image(
        self,
        centres: np.ndarray,
        use_cfg: bool = True,
        guidance_scale: float = 3.0,
        return_conditioning: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Generate a single image from cell centres.
        
        Args:
            centres: Cell centres of shape (N, 2)
            use_cfg: Use classifier-free guidance
            guidance_scale: CFG guidance scale
            return_conditioning: Also return conditioning maps
        
        Returns:
            result: Dictionary with keys:
                - 'image': Generated image (H, W) in [0, 1]
                - 'centres': Input centres (N, 2)
                - 'conditioning': Conditioning maps (3, H, W) if requested
        """
        image_shape = (self.image_size, self.image_size)
        
        # Generate image
        image = sample_from_centres(
            model=self.model,
            centres=centres,
            image_shape=image_shape,
            heatmap_sigma=self.heatmap_sigma,
            boundary_sigma=2.0,
            use_cfg=use_cfg,
            guidance_scale=guidance_scale,
            device=self.device
        )
        
        # Convert from [-1, 1] to [0, 1] for visualization
        image = to_zero_one(image, data_min=-1.0, data_max=1.0)
        
        result = {
            'image': image,
            'centres': centres
        }
        
        if return_conditioning:
            conditioning = self.generate_conditioning(centres)
            result['conditioning'] = conditioning
        
        return result
    
    def generate_image_correlated(
        self,
        centres: np.ndarray,
        x_T: torch.Tensor,
        seed_shared: int,
        seed_unique: int,
        rho: float = 0.95,
        temporal_smoothness: float = 0.0,
        unique_noise_prev: Optional[torch.Tensor] = None,
        guidance_scale: float = 0.0,
        return_conditioning: bool = False
    ) -> Dict:
        """
        Generate image with temporally correlated and smoothly evolving noise.
        
        Args:
            centres: Cell centres of shape (N, 2)
            x_T: Initial noise tensor (reused for correlation)
            seed_shared: Seed for shared noise (same across frames)
            seed_unique: Seed for unique noise (different per frame)
            rho: Spatial correlation coefficient (0-1)
            temporal_smoothness: AR(1) coefficient for smooth evolution (0-1)
            unique_noise_prev: Previous frame's unique noise state
            guidance_scale: CFG guidance strength
            return_conditioning: Also return conditioning maps
        
        Returns:
            result: Dictionary with 'image', 'centres', 'unique_noise_state', optionally 'conditioning'
        """
        # Generate conditioning
        conditioning = self.generate_conditioning(centres)
        conditioning_tensor = torch.from_numpy(conditioning[None]).float().to(self.device)
        
        # Generate with correlated and temporally smooth noise
        with torch.no_grad():
            generated, unique_noise_state = self.model.sample_correlated(
                conditioning=conditioning_tensor,
                x_T=x_T,
                seed_shared=seed_shared,
                seed_unique=seed_unique,
                rho=rho,
                temporal_smoothness=temporal_smoothness,
                unique_noise_prev=unique_noise_prev,
                guidance_scale=guidance_scale,
                clip_denoised=True
            )
        
        # Convert to numpy and normalize
        image = generated[0, 0].cpu().numpy()
        image = to_zero_one(image, data_min=-1.0, data_max=1.0)
        
        result = {
            'image': image,
            'centres': centres,
            'unique_noise_state': unique_noise_state
        }
        
        if return_conditioning:
            result['conditioning'] = conditioning
        
        return result
    
    def generate_batch(
        self,
        num_samples: int = 10,
        method: str = 'poisson',
        use_cfg: bool = True,
        guidance_scale: float = 3.0,
        seed: Optional[int] = None,
        **centre_kwargs
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Generate multiple synthetic images.
        
        Args:
            num_samples: Number of images to generate
            method: Centre generation method
            use_cfg: Use classifier-free guidance
            guidance_scale: CFG guidance scale
            seed: Base random seed (incremented for each sample)
            **centre_kwargs: Additional arguments for centre generation
        
        Returns:
            images: List of generated images, each (H, W) in [0, 1]
            centres_list: List of corresponding centres, each (N, 2)
        """
        images = []
        centres_list = []
        
        logger.info(f"Generating {num_samples} synthetic images...")
        
        for i in range(num_samples):
            # Generate centres with different seed for each sample
            sample_seed = None if seed is None else seed + i
            centres = self.generate_centres(
                method=method,
                seed=sample_seed,
                **centre_kwargs
            )
            
            # Generate image
            result = self.generate_image(
                centres=centres,
                use_cfg=use_cfg,
                guidance_scale=guidance_scale
            )
            
            images.append(result['image'])
            centres_list.append(centres)
            
            if (i + 1) % 10 == 0:
                logger.info(f"  Generated {i + 1}/{num_samples} images")
        
        logger.info(f"✓ Generated {num_samples} images")
        return images, centres_list
    
    def save_samples(
        self,
        images: List[np.ndarray],
        centres_list: List[np.ndarray],
        output_dir: str,
        prefix: str = "synthetic",
        save_centres: bool = True,
        save_visualization: bool = True
    ):
        """
        Save generated images and centres to disk.
        
        Args:
            images: List of images (H, W) in [0, 1]
            centres_list: List of centres (N, 2)
            output_dir: Output directory
            prefix: Filename prefix
            save_centres: Also save centres as .npy files
            save_visualization: Also save PNG with centre overlays
        """
        import tifffile
        import matplotlib.pyplot as plt
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for i, (image, centres) in enumerate(zip(images, centres_list)):
            # Save image as TIFF (convert to uint16 for compatibility)
            image_uint16 = (image * 65535).astype(np.uint16)
            image_file = output_path / f"{prefix}_{i:04d}.tif"
            tifffile.imwrite(image_file, image_uint16)
            
            # Save centres
            if save_centres:
                centres_file = output_path / f"{prefix}_{i:04d}_centres.npy"
                np.save(centres_file, centres)
            
            # Save visualization with centres overlay
            if save_visualization:
                fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
                ax.imshow(image, cmap='gray', vmin=0, vmax=1)
                
                # Plot centres as red crosses
                if len(centres) > 0:
                    ax.scatter(centres[:, 1], centres[:, 0], 
                             c='red', marker='x', s=50, linewidths=2, 
                             label=f'{len(centres)} cells')
                    ax.legend(loc='upper right', framealpha=0.8)
                
                ax.set_title(f'{prefix}_{i:04d}', fontsize=12)
                ax.axis('off')
                plt.tight_layout()
                
                vis_file = output_path / f"{prefix}_{i:04d}_overlay.png"
                plt.savefig(vis_file, bbox_inches='tight', pad_inches=0.1)
                plt.close(fig)
        
        logger.info(f"Saved {len(images)} samples to {output_dir}")


# Example usage
if __name__ == "__main__":
    # Initialize synthesizer
    synthesizer = CellSynthesizer(
        checkpoint_path="checkpoints/best.pt",
        device="cuda"
    )
    
    # Generate a batch of images
    images, centres = synthesizer.generate_batch(
        num_samples=10,
        method='poisson',
        density=0.0003,
        min_distance=12.0,
        use_cfg=True,
        guidance_scale=3.0,
        seed=42
    )
    
    # Save results
    synthesizer.save_samples(
        images,
        centres,
        output_dir="generated_samples",
        prefix="cell"
    )
    
    print(f"Generated {len(images)} images")
    print(f"Image shape: {images[0].shape}")
    print(f"Cell counts: {[len(c) for c in centres]}")
