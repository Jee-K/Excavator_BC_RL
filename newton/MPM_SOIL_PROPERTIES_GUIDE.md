# MPM Soil Properties Guide

## Overview

Newton's Material Point Method (MPM) solver uses continuum mechanics to simulate granular materials like soil. The key properties control how the soil behaves under stress, compression, and shear.

## Core MPM Material Properties

### 1. **Young's Modulus (E)** - Stiffness/Elasticity
Controls how much the soil resists deformation under stress.

- **Units**: Pascals (Pa) or MPa (1 MPa = 1,000,000 Pa)
- **Range**: 0.1 MPa to 50 MPa for soils
- **Effect**:
  - Lower values = softer, more deformable soil
  - Higher values = stiffer, harder soil

**Examples:**
- `E = 0.5e6` (500 kPa) - Very soft soil, mud
- `E = 1.0e6` (1 MPa) - Soft soil, loose sand
- `E = 2.0e6` (2 MPa) - Medium soil, wet soil (current default)
- `E = 5.0e6` (5 MPa) - Firm soil, compacted clay
- `E = 10.0e6` (10 MPa) - Very firm soil, dense clay
- `E = 50.0e6` (50 MPa) - Rock-like hardness

### 2. **Poisson's Ratio (ν)** - Volume Change Under Stress
Controls how much the soil changes volume when compressed.

- **Units**: Dimensionless (0 to 0.5)
- **Range**: 0.25 to 0.5 for soils
- **Effect**:
  - 0 = material changes volume freely (compressible)
  - 0.5 = incompressible (like rubber or saturated soil)

**Examples:**
- `ν = 0.25` - Highly compressible, dry granular material
- `ν = 0.30` - Typical for dry sand
- `ν = 0.35` - Typical for moist soil (current default)
- `ν = 0.40` - Wet clay, less compressible
- `ν = 0.45` - Nearly incompressible, saturated soil
- `ν = 0.49` - Almost incompressible (caution: can cause numerical issues at 0.5)

### 3. **Density (ρ)** - Mass per Volume
Controls the mass and weight of the soil.

- **Units**: kg/m³
- **Range**: 1200 to 2200 kg/m³ for soils
- **Effect**:
  - Lower values = lighter soil, easier to move
  - Higher values = heavier soil, more inertia

**Examples:**
- `ρ = 1200` - Very loose, dry sand
- `ρ = 1400` - Loose sand, peat soil
- `ρ = 1600` - Medium sand (typical dry sand)
- `ρ = 1800` - Wet sand, moist soil (current default)
- `ρ = 2000` - Wet clay, compacted soil
- `ρ = 2200` - Very dense clay, wet gravel

### 4. **Friction Angle (φ)** - Internal Friction
Controls how much the soil resists shearing (sliding).

- **Units**: Degrees (°)
- **Range**: 20° to 45° for soils
- **In code**: Converted to friction coefficient `μ = tan(φ)`
- **Effect**:
  - Lower angles = smooth, low friction (like wet clay)
  - Higher angles = rough, high friction (like sand)

**Examples:**
- `φ = 20°` (`μ = 0.36`) - Very smooth, wet clay
- `φ = 25°` (`μ = 0.47`) - Soft clay
- `φ = 30°` (`μ = 0.58`) - Typical soil (current default)
- `φ = 35°` (`μ = 0.70`) - Dense sand
- `φ = 40°` (`μ = 0.84`) - Coarse sand, gravel
- `φ = 45°` (`μ = 1.00`) - Very dense gravel

### 5. **Cohesion (c)** - Stickiness/Binding
Controls how much the soil particles stick together.

- **Units**: Pascals (Pa) or kPa (1 kPa = 1,000 Pa)
- **Range**: 0 to 100 kPa for soils
- **Effect**:
  - 0 = no cohesion, free-flowing (dry sand)
  - Higher values = sticky, clumpy (wet soil, clay)

**Examples:**
- `c = 0` - Completely dry sand, no cohesion
- `c = 1000` (1 kPa) - Slightly moist sand
- `c = 5000` (5 kPa) - Wet soil (current default)
- `c = 10000` (10 kPa) - Sticky clay
- `c = 25000` (25 kPa) - Very cohesive clay
- `c = 50000` (50 kPa) - Extremely sticky clay
- `c = 100000` (100 kPa) - Clay with high binding

## Preset Soil Types

### Dry Sand (Free-Flowing)
```python
youngs_modulus = 1.0e6    # 1 MPa - soft
poissons_ratio = 0.30     # Compressible
density = 1600.0          # Light
friction_angle = 35.0     # High friction
cohesion = 0.0            # No stickiness
```

### Wet Sand (Slightly Cohesive)
```python
youngs_modulus = 1.5e6    # 1.5 MPa
poissons_ratio = 0.33     # Slightly less compressible
density = 1800.0          # Heavier
friction_angle = 30.0     # Medium friction
cohesion = 2000.0         # 2 kPa - slight cohesion
```

### Soft Soil (Current Default)
```python
youngs_modulus = 2.0e6    # 2 MPa
poissons_ratio = 0.35     # Typical
density = 1800.0          # Typical wet
friction_angle = 30.0     # Medium
cohesion = 5000.0         # 5 kPa - moderate cohesion
```

### Firm Clay (Sticky)
```python
youngs_modulus = 5.0e6    # 5 MPa - firm
poissons_ratio = 0.40     # Less compressible
density = 2000.0          # Dense
friction_angle = 25.0     # Lower friction
cohesion = 15000.0        # 15 kPa - very sticky
```

### Dense Compacted Soil
```python
youngs_modulus = 10.0e6   # 10 MPa - hard
poissons_ratio = 0.38     # Compacted
density = 2100.0          # Very dense
friction_angle = 38.0     # High friction
cohesion = 8000.0         # 8 kPa - some cohesion
```

### Heavy Wet Clay
```python
youngs_modulus = 8.0e6    # 8 MPa
poissons_ratio = 0.45     # Nearly incompressible
density = 2200.0          # Very heavy
friction_angle = 22.0     # Low friction
cohesion = 25000.0        # 25 kPa - extremely sticky
```

### Loose Gravel (Non-cohesive)
```python
youngs_modulus = 3.0e6    # 3 MPa
poissons_ratio = 0.28     # Highly compressible
density = 1700.0          # Medium weight
friction_angle = 40.0     # Very high friction
cohesion = 0.0            # No cohesion
```

## How to Change Soil Properties in Your Code

In `excavator_mpm_soil.py`, modify the `create_mpm_soil()` function around line 172-177:

```python
# Soil material properties (lines 172-177)
youngs_modulus = 2.0e6    # Change this
poissons_ratio = 0.35     # Change this
density = 1800.0          # Change this
friction_angle = 30.0     # Change this
cohesion = 5000.0         # Change this
```

## Interaction Effects

### Excavation Difficulty
- **Easy to dig**: Low E, low density, low cohesion (dry sand)
- **Hard to dig**: High E, high density, high cohesion (clay)

### Pile Formation
- **Spreading piles**: Low cohesion, high friction (dry sand spreads flat)
- **Steep piles**: High cohesion, low friction (wet clay makes steep piles)

### Bucket Resistance
- **Low resistance**: Low E, low density (soft soil, easy to scoop)
- **High resistance**: High E, high density (hard soil, excavator must work harder)

### Soil Sticking to Bucket
- **No sticking**: Zero cohesion (dry sand falls off)
- **Heavy sticking**: High cohesion (wet clay clings to bucket)

## Performance Considerations

### Particle Count
More particles = more realistic but slower simulation.

Current: **307,197 particles** (4m × 4m × 0.8m, voxel_size=0.05m, 3 particles/cell)

To adjust:
```python
# In main() function
voxel_size = 0.05          # Smaller = more particles, slower
particles_per_cell = 3     # More = smoother, slower
```

**Examples:**
- `voxel_size=0.08, ppc=2`: ~120,000 particles (faster, less detailed)
- `voxel_size=0.05, ppc=3`: ~307,000 particles (current, balanced)
- `voxel_size=0.03, ppc=4`: ~1,200,000 particles (very detailed, much slower)

### Solver Options
In `__init__` around line 94-101:

```python
mpm_options.tolerance = 1.0e-5      # Lower = more accurate, slower
mpm_options.max_iterations = 50     # More = more accurate, slower
```

## Troubleshooting

### Soil too soft/mushy
- Increase Young's modulus (E)
- Increase cohesion (c)

### Soil too stiff/rigid
- Decrease Young's modulus (E)
- Check if Poisson's ratio is too high

### Soil falls through ground
- Check ground plane position
- Increase solver iterations
- Reduce time step

### Numerical instability (explosion, NaN)
- Reduce Poisson's ratio (keep below 0.49)
- Reduce time step (increase sim_substeps)
- Check extreme property values

## References

- Newton MPM Documentation: https://newton-physics.github.io/newton/
- Material Point Method: https://en.wikipedia.org/wiki/Material_point_method
- Soil Mechanics: Typical values from geotechnical engineering

## Quick Reference Table

| Soil Type | E (MPa) | ν | ρ (kg/m³) | φ (°) | c (kPa) |
|-----------|---------|---|-----------|-------|---------|
| Dry sand | 1.0 | 0.30 | 1600 | 35 | 0 |
| Wet sand | 1.5 | 0.33 | 1800 | 30 | 2 |
| Soft soil | 2.0 | 0.35 | 1800 | 30 | 5 |
| Firm clay | 5.0 | 0.40 | 2000 | 25 | 15 |
| Dense soil | 10.0 | 0.38 | 2100 | 38 | 8 |
| Wet clay | 8.0 | 0.45 | 2200 | 22 | 25 |
| Gravel | 3.0 | 0.28 | 1700 | 40 | 0 |
