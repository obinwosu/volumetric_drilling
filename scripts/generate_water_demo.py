#!/usr/bin/env python3
"""
Generate visual demo of photorealistic water rendering.

Creates comparison frames showing:
- Basic water vs Tier 1 vs Photorealistic
- Different depths and lighting conditions
- Key features: highlights, meniscus, caustics, absorption
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Ellipse, Polygon
from matplotlib.collections import PolyCollection
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
import sys

sys.path.insert(0, 'scripts')

def create_bone_surface():
    """Create a realistic bone surface with texture."""
    bone_color = np.array([255, 249, 219]) / 255.0
    return bone_color

def draw_water_surface(ax, params, depth_mm, light_pos=(0.7, 0.9), view_angle=0.3):
    """
    Draw realistic water surface with all photorealistic features.
    
    params: material parameters dict
    depth_mm: water depth
    light_pos: position of light source
    view_angle: viewing angle (0=top, 1=side)
    """
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Background bone color
    bone = create_bone_surface()
    ax.add_patch(mpatches.Rectangle((0, 0), 1, 0.35, 
                                     facecolor=bone, 
                                     edgecolor='none',
                                     zorder=1))
    
    # Caustic patterns on bone (if volumetric)
    if not params.get('thin_walled', True) and depth_mm > 0.5:
        caustic_strength = min(depth_mm / 3.0, 1.0)
        for i in range(5):
            x = 0.15 + i * 0.15 + np.random.uniform(-0.03, 0.03)
            y = 0.05 + np.random.uniform(0, 0.25)
            width = 0.08 + np.random.uniform(-0.02, 0.02)
            height = 0.04
            angle = np.random.uniform(-15, 15)
            
            caustic = Ellipse((x, y), width, height, angle=angle,
                            facecolor='white',
                            alpha=0.15 * caustic_strength,
                            edgecolor='none',
                            zorder=2)
            ax.add_patch(caustic)
    
    # Water body
    water_base = np.array([0.95, 0.97, 1.0])
    
    # Volume absorption (blue tint for deep water)
    if not params.get('thin_walled') and params.get('enable_volume_absorption'):
        absorption_color = np.array(params['volume_absorption_color'])
        distance = params.get('volume_absorption_distance', 3.0)
        absorption_factor = 1 - np.exp(-depth_mm / distance)
        water_color = water_base * (1 - absorption_factor) + absorption_color * absorption_factor
    else:
        water_color = water_base
    
    # Water surface height based on depth
    water_height = min(0.15 + depth_mm * 0.08, 0.65)
    
    # Create water polygon with slight waves
    num_points = 50
    x_points = np.linspace(0, 1, num_points)
    # Add subtle surface waves
    y_top = water_height + 0.01 * np.sin(x_points * 10 + view_angle * 5)
    
    # Water polygon
    verts = [(0, 0.35)] + list(zip(x_points, y_top)) + [(1, 0.35)]
    water_poly = Polygon(verts, facecolor=water_color, 
                        alpha=0.85,
                        edgecolor='none',
                        zorder=3)
    ax.add_patch(water_poly)
    
    # Contact shadow at bone-water interface
    shadow_gradient = np.linspace(0, 1, 100).reshape(1, -1)
    ax.imshow(shadow_gradient, extent=[0, 1, 0.35, 0.36], 
             aspect='auto', cmap='gray', alpha=0.2, zorder=4)
    
    # Meniscus/edge highlight (coat layer effect)
    coat_weight = params.get('coat_weight', 0)
    if coat_weight > 0.2:
        # Bright edge along top surface
        edge_brightness = coat_weight * 0.7
        edge_y = np.mean(y_top)
        
        # Create multiple edge highlights for strong meniscus
        for offset in [-0.005, 0, 0.005]:
            edge_line = plt.Line2D([0, 1], [edge_y + offset, edge_y + offset],
                                  color='white',
                                  alpha=edge_brightness * 0.5,
                                  linewidth=3,
                                  zorder=10)
            ax.add_line(edge_line)
    
    # Specular highlights (multiple for very sharp roughness)
    roughness = params.get('specular_reflection_roughness', 0.04)
    
    # Number and sharpness of highlights depend on roughness
    if roughness < 0.008:  # Ultra-sharp
        highlight_positions = [(0.35, 0.7), (0.55, 0.75), (0.75, 0.72)]
        base_size = 0.03
        alpha = 0.95
    elif roughness < 0.015:  # Very sharp
        highlight_positions = [(0.45, 0.72), (0.65, 0.7)]
        base_size = 0.05
        alpha = 0.85
    elif roughness < 0.025:  # Sharp
        highlight_positions = [(0.55, 0.7)]
        base_size = 0.08
        alpha = 0.7
    else:  # Blurry
        highlight_positions = [(0.55, 0.7)]
        base_size = 0.12
        alpha = 0.5
    
    for hx, hy in highlight_positions:
        hy_adjusted = water_height + (hy - 0.7) * 0.1
        
        # Main highlight
        highlight_size = base_size / (roughness * 50)
        highlight = Circle((hx, hy_adjusted), highlight_size,
                          facecolor='white',
                          alpha=alpha,
                          edgecolor='none',
                          zorder=11)
        ax.add_patch(highlight)
        
        # Coat layer highlight (if present)
        if coat_weight > 0.3:
            coat_highlight = Circle((hx + 0.01, hy_adjusted + 0.01), 
                                   highlight_size * 0.6,
                                   facecolor='white',
                                   alpha=coat_weight * 0.6,
                                   edgecolor='none',
                                   zorder=12)
            ax.add_patch(coat_highlight)
    
    # Edge meniscus detail (for thin films with high coat)
    if params.get('thin_walled') and coat_weight > 0.6:
        for edge_x in [0.05, 0.95]:
            meniscus = Ellipse((edge_x, water_height), 0.03, 0.02,
                              facecolor='white',
                              alpha=0.4,
                              edgecolor='none',
                              zorder=9)
            ax.add_patch(meniscus)
    
    # Refraction edge (thickness visualization)
    if not params.get('thin_walled'):
        # Show edge where you can see through water
        refract_x = [0.05, 0.15, 0.2]
        for rx in refract_x:
            ry = water_height - 0.02
            refract = Ellipse((rx, ry), 0.02, 0.05,
                            facecolor=water_color * 0.8,
                            alpha=0.6,
                            edgecolor=water_color * 0.6,
                            linewidth=1,
                            zorder=8)
            ax.add_patch(refract)

def create_comparison_frame():
    """Create comparison of Basic vs Tier 1 vs Photorealistic."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Basic water parameters
    basic_params = {
        "thin_walled": True,
        "specular_reflection_roughness": 0.04,
        "coat_weight": 0.0,
        "enable_volume_absorption": False,
    }
    
    # Tier 1 parameters
    tier1_params = {
        "thin_walled": False,
        "specular_reflection_roughness": 0.008,
        "coat_weight": 0.35,
        "enable_volume_absorption": True,
        "volume_absorption_color": [0.88, 0.92, 0.98],
        "volume_absorption_distance": 1.5,
    }
    
    # Photorealistic parameters
    photo_params = {
        "thin_walled": False,
        "specular_reflection_roughness": 0.004,
        "coat_weight": 0.50,
        "enable_volume_absorption": True,
        "volume_absorption_color": [0.86, 0.91, 0.98],
        "volume_absorption_distance": 1.2,
    }
    
    depth = 1.5  # mm
    
    draw_water_surface(axes[0], basic_params, depth)
    axes[0].set_title("Basic Water\n(Game Quality)", fontsize=12, pad=10)
    
    draw_water_surface(axes[1], tier1_params, depth)
    axes[1].set_title("Tier 1 Enhanced\n(Good CGI)", fontsize=12, pad=10)
    
    draw_water_surface(axes[2], photo_params, depth)
    axes[2].set_title("Photorealistic\n(Near-FLIP Quality)", fontsize=12, pad=10, fontweight='bold')
    
    fig.suptitle("Water Rendering Quality Comparison (1.5mm depth)", 
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

def create_depth_progression():
    """Create progression showing different depths."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    depths = [0.2, 0.5, 1.5, 3.0]
    titles = ["Thin Film\n0.2mm", "Light Pool\n0.5mm", "Medium Pool\n1.5mm", "Deep Pool\n3.0mm"]
    
    for ax, depth, title in zip(axes, depths, titles):
        # Use photorealistic params adjusted for depth
        if depth < 0.3:
            params = {
                "thin_walled": True,
                "specular_reflection_roughness": 0.015,
                "coat_weight": 0.75,
                "enable_volume_absorption": False,
            }
        elif depth < 1.5:
            params = {
                "thin_walled": False,
                "specular_reflection_roughness": 0.006,
                "coat_weight": 0.45,
                "enable_volume_absorption": True,
                "volume_absorption_color": [0.86, 0.91, 0.98],
                "volume_absorption_distance": depth * 0.4,
            }
        else:
            params = {
                "thin_walled": False,
                "specular_reflection_roughness": 0.004,
                "coat_weight": 0.30,
                "enable_volume_absorption": True,
                "volume_absorption_color": [0.84, 0.89, 0.98],
                "volume_absorption_distance": depth * 0.35,
            }
        
        draw_water_surface(ax, params, depth)
        ax.set_title(title, fontsize=11, pad=8)
    
    fig.suptitle("Photorealistic Water: Depth-Dependent Appearance", 
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

def create_animated_demo():
    """Create animated GIF showing water accumulating."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    def update(frame):
        # Animate depth increasing
        depth = 0.1 + (frame / 30.0) * 2.5  # 0.1 to 2.6 mm over 30 frames
        
        # Adjust params based on depth
        if depth < 0.3:
            params = {
                "thin_walled": True,
                "specular_reflection_roughness": 0.015,
                "coat_weight": 0.75,
                "enable_volume_absorption": False,
            }
            regime = "Thin Film"
        elif depth < 1.5:
            params = {
                "thin_walled": False,
                "specular_reflection_roughness": 0.006,
                "coat_weight": 0.45,
                "enable_volume_absorption": True,
                "volume_absorption_color": [0.86, 0.91, 0.98],
                "volume_absorption_distance": depth * 0.4,
            }
            regime = "Medium Pool"
        else:
            params = {
                "thin_walled": False,
                "specular_reflection_roughness": 0.004,
                "coat_weight": 0.30,
                "enable_volume_absorption": True,
                "volume_absorption_color": [0.84, 0.89, 0.98],
                "volume_absorption_distance": depth * 0.35,
            }
            regime = "Deep Pool"
        
        draw_water_surface(ax, params, depth, view_angle=frame * 0.02)
        
        # Add info text
        fig.suptitle(f"Photorealistic Water Accumulating\n"
                    f"Depth: {depth:.2f}mm | Regime: {regime}",
                    fontsize=13, fontweight='bold')
        
        return ax,
    
    anim = FuncAnimation(fig, update, frames=30, interval=100, blit=False)
    return fig, anim

def main():
    print("Generating water rendering demos...")
    
    # 1. Comparison frame
    print("  1. Creating quality comparison...")
    fig1 = create_comparison_frame()
    fig1.savefig('/tmp/water_quality_comparison.png', dpi=150, bbox_inches='tight', 
                 facecolor='white')
    print("     Saved: /tmp/water_quality_comparison.png")
    plt.close(fig1)
    
    # 2. Depth progression
    print("  2. Creating depth progression...")
    fig2 = create_depth_progression()
    fig2.savefig('/tmp/water_depth_progression.png', dpi=150, bbox_inches='tight',
                 facecolor='white')
    print("     Saved: /tmp/water_depth_progression.png")
    plt.close(fig2)
    
    # 3. Animated GIF
    print("  3. Creating animated demo (this may take a moment)...")
    fig3, anim = create_animated_demo()
    writer = PillowWriter(fps=10)
    anim.save('/tmp/water_animation.gif', writer=writer, dpi=100)
    print("     Saved: /tmp/water_animation.gif")
    plt.close(fig3)
    
    print("\n✓ All demos generated successfully!")
    print("  - Quality comparison: /tmp/water_quality_comparison.png")
    print("  - Depth progression: /tmp/water_depth_progression.png")
    print("  - Animation: /tmp/water_animation.gif")

if __name__ == '__main__':
    main()
