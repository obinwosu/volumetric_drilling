"""Screen-space water rendering effects for real-time photorealism.

Implements cheap but high-impact screen-space techniques:
- Thickness rendering for proper refraction
- Contact shadows at water-bone interface  
- Edge detection for meniscus enhancement
- Temporal accumulation for quality

These effects are post-process, so they're very fast (<2ms typically)
but dramatically improve water appearance.

Photorealism Strategy for 30-40 FPS:
1. Screen-space thickness (~0.5ms) - enables proper refraction
2. Contact shadows (~0.3ms) - grounds water visually
3. Edge enhancement (~0.2ms) - sharpen meniscus
4. Temporal AA (~0.1ms) - smooth out artifacts

Total overhead: ~1-2ms, preserves real-time performance.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


class ScreenSpaceWaterEffects:
    """
    Fast screen-space effects for photorealistic water at 30-40 FPS.
    
    These are post-process effects that run after the main RTX rendering
    pass, adding visual quality with minimal performance cost.
    """
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.thickness_buffer = None
        self.last_frame = None
        self.temporal_blend = 0.15  # Blend with previous frame
        
    def compute_thickness_map(self, 
                             depth_front: np.ndarray,
                             depth_back: np.ndarray) -> np.ndarray:
        """
        Compute thickness from front and back depth buffers.
        
        This is THE KEY technique for realistic refraction - it lets
        the material know how much water light passes through.
        
        Args:
            depth_front: Front-face depth buffer (closest water surface)
            depth_back: Back-face depth buffer (water-bone interface)
            
        Returns:
            Thickness in millimeters at each pixel
        """
        # Thickness is difference between back and front depth
        thickness = np.maximum(depth_back - depth_front, 0.0)
        
        # Convert depth units to millimeters (assuming depth in meters)
        thickness_mm = thickness * 1000.0
        
        # Clamp to reasonable range
        thickness_mm = np.clip(thickness_mm, 0.0, 10.0)
        
        # Gaussian blur for smoothness
        thickness_mm = self._bilateral_filter(thickness_mm, depth_front)
        
        return thickness_mm
    
    def compute_contact_shadows(self,
                                water_depth: np.ndarray,
                                bone_depth: np.ndarray,
                                normal: np.ndarray) -> np.ndarray:
        """
        Compute soft shadows where water contacts bone.
        
        This is ambient occlusion specifically at the water-bone interface,
        making the water look "grounded" and connected to the surface.
        
        Returns:
            Shadow factor 0-1 (0 = full shadow, 1 = no shadow)
        """
        # Distance to bone
        distance = bone_depth - water_depth
        
        # Close to bone = more shadow
        shadow = 1.0 - np.exp(-distance * 10.0)
        
        # Modulate by surface normal (facing down = more shadow)
        if normal is not None:
            facing_down = np.maximum(-normal[:, :, 2], 0.0)  # Z component
            shadow *= (0.7 + 0.3 * facing_down)
        
        return np.clip(shadow, 0.3, 1.0)
    
    def enhance_edges(self, 
                     color: np.ndarray,
                     depth: np.ndarray,
                     thickness: np.ndarray) -> np.ndarray:
        """
        Enhance edges for sharper meniscus appearance.
        
        Detects edges in thickness map and brightens them slightly
        to emphasize surface tension and meniscus.
        """
        # Sobel edge detection on thickness
        edges = self._sobel_edges(thickness)
        
        # Brighten edges
        edge_highlight = np.stack([edges] * 3, axis=-1)
        enhanced = color + edge_highlight * 0.15
        
        return np.clip(enhanced, 0.0, 1.0)
    
    def temporal_accumulation(self,
                            current_frame: np.ndarray) -> np.ndarray:
        """
        Blend with previous frames for temporal anti-aliasing.
        
        This reduces flickering and noise, especially important for
        caustics and specular highlights.
        """
        if self.last_frame is None:
            self.last_frame = current_frame.copy()
            return current_frame
        
        # Exponential moving average
        blended = (
            (1 - self.temporal_blend) * current_frame +
            self.temporal_blend * self.last_frame
        )
        
        self.last_frame = blended.copy()
        return blended
    
    def _bilateral_filter(self,
                         thickness: np.ndarray,
                         depth: np.ndarray,
                         kernel_size: int = 5) -> np.ndarray:
        """
        Edge-preserving blur for smooth thickness without losing detail.
        """
        # Simple implementation - would use optimized version in production
        from scipy.ndimage import gaussian_filter
        
        # Smooth thickness while preserving depth edges
        smoothed = gaussian_filter(thickness, sigma=2.0)
        
        # Detect depth edges
        depth_grad = np.gradient(depth)
        edge_strength = np.sqrt(depth_grad[0]**2 + depth_grad[1]**2)
        edge_mask = edge_strength > 0.01
        
        # Use original thickness at edges, smoothed elsewhere
        result = np.where(edge_mask, thickness, smoothed)
        
        return result
    
    def _sobel_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Sobel edge detection.
        """
        from scipy.ndimage import sobel
        
        edges_x = sobel(image, axis=0)
        edges_y = sobel(image, axis=1)
        edges = np.sqrt(edges_x**2 + edges_y**2)
        
        # Normalize
        if edges.max() > 0:
            edges = edges / edges.max()
        
        return edges


class AdaptiveQualityController:
    """
    Dynamically adjust quality to maintain target FPS.
    
    If frame time goes above budget, reduce quality. If we have headroom,
    increase quality. This ensures 30-40 FPS even in complex scenes.
    """
    
    def __init__(self, target_fps: float = 35.0):
        self.target_fps = target_fps
        self.target_frame_ms = 1000.0 / target_fps
        self.recent_frame_times = []
        self.max_history = 30
        
        # Quality levels
        self.quality_level = 2  # 0=low, 1=medium, 2=high, 3=ultra
        self.quality_settings = {
            0: {  # Low - 40+ FPS
                'caustics': False,
                'spp': 2,
                'max_bounces': 4,
                'mesh_resolution': 32,
                'temporal_samples': 1,
            },
            1: {  # Medium - 35-40 FPS
                'caustics': True,
                'spp': 4,
                'max_bounces': 5,
                'mesh_resolution': 48,
                'temporal_samples': 2,
            },
            2: {  # High - 30-35 FPS (default)
                'caustics': True,
                'spp': 6,
                'max_bounces': 6,
                'mesh_resolution': 64,
                'temporal_samples': 3,
            },
            3: {  # Ultra - 25-30 FPS
                'caustics': True,
                'spp': 8,
                'max_bounces': 8,
                'mesh_resolution': 96,
                'temporal_samples': 4,
            },
        }
    
    def update(self, frame_time_ms: float) -> dict:
        """
        Update quality based on recent frame times.
        
        Returns current quality settings.
        """
        self.recent_frame_times.append(frame_time_ms)
        if len(self.recent_frame_times) > self.max_history:
            self.recent_frame_times.pop(0)
        
        # Average recent frame time
        avg_frame_time = np.mean(self.recent_frame_times)
        
        # Adjust quality if needed
        if avg_frame_time > self.target_frame_ms * 1.15:
            # Too slow, reduce quality
            if self.quality_level > 0:
                self.quality_level -= 1
                print(f"Reducing quality to level {self.quality_level} "
                      f"(avg frame time: {avg_frame_time:.1f}ms)", flush=True)
        
        elif avg_frame_time < self.target_frame_ms * 0.85:
            # Fast enough, increase quality
            if self.quality_level < 3:
                self.quality_level += 1
                print(f"Increasing quality to level {self.quality_level} "
                      f"(avg frame time: {avg_frame_time:.1f}ms)", flush=True)
        
        return self.quality_settings[self.quality_level]
    
    def get_current_settings(self) -> dict:
        """Get current quality settings without updating."""
        return self.quality_settings[self.quality_level]
