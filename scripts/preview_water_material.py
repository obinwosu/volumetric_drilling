#!/usr/bin/env python3
"""
Visual preview of enhanced water material parameters across different depths.

Generates comparison images showing material properties for thin films, 
medium pools, and deep pools based on the Tier 1 implementation.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Import our enhanced water functions
import sys
sys.path.insert(0, 'scripts')
from rtx_view import rtx_materials

def visualize_material_params(ax, params, title, depth_mm):
    """Visualize key material parameters as a property card."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.2, title, ha='center', va='top', fontsize=14, fontweight='bold')
    ax.text(5, 8.6, f"Water Depth: {depth_mm:.2f} mm", ha='center', va='top', fontsize=10, color='#666')
    
    y = 7.8
    dy = 0.6
    
    properties = [
        ("Thin-Walled", "Yes" if params.get('thin_walled') else "No"),
        ("Specular Roughness", f"{params.get('specular_reflection_roughness', 0):.3f}"),
        ("Coat Weight", f"{params.get('coat_weight', 0):.2f}"),
        ("Coat Roughness", f"{params.get('coat_roughness', 0):.3f}"),
        ("Diffuse Weight", f"{params.get('diffuse_reflection_weight', 0):.2f}"),
        ("IOR", f"{params.get('specular_reflection_ior', 1.333):.3f}"),
    ]
    
    if not params.get('thin_walled'):
        vol_abs = params.get('enable_volume_absorption', False)
        properties.append(("Volume Absorption", "Enabled" if vol_abs else "Disabled"))
        if vol_abs:
            properties.append(("Absorption Dist", f"{params.get('volume_absorption_distance', 0):.2f} mm"))
    
    for label, value in properties:
        ax.text(1, y, label + ":", ha='left', va='top', fontsize=9, fontweight='bold')
        ax.text(9, y, value, ha='right', va='top', fontsize=9, family='monospace')
        y -= dy

def create_water_visualization(ax, params, depth_mm):
    """Create a simplified visual representation of water appearance."""
    # Background (bone)
    bone_color = np.array([255, 249, 219]) / 255.0
    ax.add_patch(mpatches.Rectangle((0, 0), 1, 0.4, facecolor=bone_color, edgecolor='none'))
    
    # Water layer
    water_base = np.array([0.95, 0.97, 1.0])
    
    # Apply absorption for volumetric water
    if not params.get('thin_walled') and params.get('enable_volume_absorption'):
        absorption_color = np.array(params['volume_absorption_color'])
        distance = params.get('volume_absorption_distance', 3.0)
        # Simple absorption model
        absorption_factor = np.exp(-depth_mm / distance)
        water_color = water_base * absorption_factor + (1 - absorption_factor) * absorption_color
    else:
        water_color = water_base
    
    water_height = min(0.6, 0.2 + depth_mm * 0.05)
    ax.add_patch(mpatches.Rectangle((0, 0.4), 1, water_height, 
                                     facecolor=water_color, 
                                     alpha=0.85,
                                     edgecolor='white',
                                     linewidth=2))
    
    # Coat/meniscus highlight (if coat weight is significant)
    coat_weight = params.get('coat_weight', 0)
    if coat_weight > 0.2:
        # Bright highlight at top edge
        highlight_height = 0.02
        highlight_alpha = coat_weight * 0.6
        ax.add_patch(mpatches.Rectangle((0, 0.4 + water_height - highlight_height), 
                                        1, highlight_height,
                                        facecolor='white',
                                        alpha=highlight_alpha,
                                        edgecolor='none'))
    
    # Specular highlight (bright spot)
    roughness = params.get('specular_reflection_roughness', 0.04)
    if roughness < 0.015:
        # Very sharp highlight
        highlight_size = 0.1
        highlight_alpha = 0.9
    elif roughness < 0.025:
        highlight_size = 0.15
        highlight_alpha = 0.7
    else:
        highlight_size = 0.2
        highlight_alpha = 0.5
    
    circle = mpatches.Circle((0.7, 0.4 + water_height * 0.7), 
                             highlight_size * (0.5 / roughness)**0.5,
                             facecolor='white',
                             alpha=highlight_alpha,
                             edgecolor='none')
    ax.add_patch(circle)
    
    # Caustics on bone (if volumetric)
    if not params.get('thin_walled') and depth_mm > 0.5:
        # Simple caustic pattern
        for i in range(3):
            x = 0.2 + i * 0.25
            width = 0.08
            caustic = mpatches.Ellipse((x, 0.2), width, 0.05,
                                       facecolor='white',
                                       alpha=0.3,
                                       edgecolor='none')
            ax.add_patch(caustic)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f"{depth_mm:.1f} mm depth", fontsize=10, pad=5)

def main():
    # Test depths
    depths = [0.2, 0.5, 1.0, 2.5]
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Enhanced Water Material: Depth-Dependent Appearance (Tier 1)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Add subtitle explaining the comparison
    fig.text(0.5, 0.94, 
             'Left: Material Properties | Right: Visual Appearance (Simplified)',
             ha='center', fontsize=11, style='italic', color='#666')
    
    for i, depth in enumerate(depths):
        # Get parameters for this depth
        params = rtx_materials.water_params_from_depth(depth)
        
        # Material properties (left column)
        ax_props = plt.subplot(4, 2, i*2 + 1)
        regime = "Thin Film" if depth < 0.3 else ("Medium Pool" if depth < 1.5 else "Deep Pool")
        visualize_material_params(ax_props, params, regime, depth)
        
        # Visual representation (right column)
        ax_visual = plt.subplot(4, 2, i*2 + 2)
        create_water_visualization(ax_visual, params, depth)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    # Save
    output_path = '/tmp/enhanced_water_preview.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Preview saved to: {output_path}")
    
    # Also create a comparison image
    fig2, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig2.suptitle('Before vs After: Water Material Comparison', fontsize=14, fontweight='bold')
    
    # Before (basic water)
    ax_before = axes[0]
    old_params = {
        "thin_walled": True,
        "specular_reflection_roughness": 0.04,
        "coat_weight": 0.0,
        "diffuse_reflection_weight": 0.06,
        "enable_volume_absorption": False,
    }
    visualize_material_params(ax_before, old_params, "BEFORE: Basic Water", 1.0)
    
    # After (enhanced water)
    ax_after = axes[1]
    new_params = rtx_materials.water_params_from_depth(1.0)
    visualize_material_params(ax_after, new_params, "AFTER: Enhanced Water (1mm)", 1.0)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    comparison_path = '/tmp/water_comparison.png'
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Comparison saved to: {comparison_path}")
    
    return output_path, comparison_path

if __name__ == '__main__':
    preview, comparison = main()
    print("\n✓ Visual preview generated successfully!")
    print(f"  1. Depth progression: {preview}")
    print(f"  2. Before/after: {comparison}")
