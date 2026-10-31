"""RTX rendering configuration for photorealistic water and caustics.

This module provides functions to configure NVIDIA RTX path tracing settings
for enhanced water appearance, including caustics, transmission bounces, and
other advanced rendering features.
"""

from __future__ import annotations


def setup_caustics(viewer, enabled: bool = True):
    """
    Enable RTX caustics for realistic water refraction effects on bone.
    
    Caustics are the bright patterns created when light refracts through water
    and focuses on the bone surface beneath. This is computationally expensive
    but provides significant realism for surgical irrigation visualization.
    
    Args:
        viewer: RTX viewer instance with carb.settings access
        enabled: Whether to enable caustics rendering
    
    Returns:
        True if settings were applied successfully, False otherwise
    """
    try:
        import carb
        
        settings = carb.settings.get_settings()
        
        if enabled:
            # Enable caustics in path tracing
            settings.set("/rtx/pathtracing/caustics", True)
            
            # Increase samples for cleaner caustics (4-8 recommended)
            current_spp = settings.get("/rtx/pathtracing/totalSpp")
            if current_spp is None or current_spp < 4:
                settings.set("/rtx/pathtracing/totalSpp", 6)
            
            # Increase max bounces to allow light through water
            settings.set("/rtx/pathtracing/maxBounces", 6)
            settings.set("/rtx/pathtracing/maxSpecularAndTransmissionBounces", 8)
            
            # Optional: Photon mapping for faster caustics (if supported)
            # This is more efficient than pure path tracing for caustics
            try:
                settings.set("/rtx/pathtracing/causticPhotonMap/enable", True)
                settings.set("/rtx/pathtracing/causticPhotonMap/photonCount", 50000)
            except Exception:
                # Photon mapping may not be available in all RTX configurations
                pass
            
            print("RTX caustics enabled: realistic water refraction effects active", flush=True)
        else:
            settings.set("/rtx/pathtracing/caustics", False)
            print("RTX caustics disabled", flush=True)
        
        return True
        
    except Exception as exc:
        print(f"Failed to configure RTX caustics: {exc}", flush=True)
        return False


def setup_transmission_quality(viewer, quality: str = "high"):
    """
    Configure transmission quality for water rendering.
    
    Args:
        viewer: RTX viewer instance
        quality: Quality level - "low", "medium", "high", or "ultra"
    
    Returns:
        True if settings were applied successfully
    """
    try:
        import carb
        
        settings = carb.settings.get_settings()
        
        quality_configs = {
            "low": {
                "totalSpp": 2,
                "maxBounces": 4,
                "maxSpecularAndTransmissionBounces": 4,
            },
            "medium": {
                "totalSpp": 4,
                "maxBounces": 5,
                "maxSpecularAndTransmissionBounces": 6,
            },
            "high": {
                "totalSpp": 6,
                "maxBounces": 6,
                "maxSpecularAndTransmissionBounces": 8,
            },
            "ultra": {
                "totalSpp": 8,
                "maxBounces": 8,
                "maxSpecularAndTransmissionBounces": 10,
            },
        }
        
        config = quality_configs.get(quality.lower(), quality_configs["high"])
        
        for key, value in config.items():
            settings.set(f"/rtx/pathtracing/{key}", value)
        
        print(f"RTX transmission quality: {quality} ({config['totalSpp']} spp, "
              f"{config['maxSpecularAndTransmissionBounces']} transmission bounces)", 
              flush=True)
        
        return True
        
    except Exception as exc:
        print(f"Failed to configure RTX transmission quality: {exc}", flush=True)
        return False


def setup_water_rendering(viewer, enable_caustics: bool = True, quality: str = "high"):
    """
    Configure all RTX settings for photorealistic water rendering.
    
    This is a convenience function that applies all recommended settings for
    realistic surgical irrigation visualization.
    
    Args:
        viewer: RTX viewer instance
        enable_caustics: Whether to enable caustics (refraction patterns)
        quality: Rendering quality level
    
    Returns:
        Dict with status of each configuration step
    """
    results = {
        "caustics": setup_caustics(viewer, enabled=enable_caustics),
        "transmission": setup_transmission_quality(viewer, quality=quality),
    }
    
    if all(results.values()):
        print("✓ RTX water rendering configured successfully", flush=True)
    else:
        print("⚠ Some RTX settings could not be applied", flush=True)
    
    return results


def get_current_rtx_config(viewer) -> dict:
    """
    Query current RTX rendering configuration.
    
    Args:
        viewer: RTX viewer instance
    
    Returns:
        Dictionary of current RTX settings
    """
    try:
        import carb
        
        settings = carb.settings.get_settings()
        
        return {
            "caustics": settings.get("/rtx/pathtracing/caustics"),
            "totalSpp": settings.get("/rtx/pathtracing/totalSpp"),
            "maxBounces": settings.get("/rtx/pathtracing/maxBounces"),
            "maxTransmissionBounces": settings.get("/rtx/pathtracing/maxSpecularAndTransmissionBounces"),
            "photonMapping": settings.get("/rtx/pathtracing/causticPhotonMap/enable"),
        }
        
    except Exception as exc:
        print(f"Failed to query RTX configuration: {exc}", flush=True)
        return {}
