# Excavator + MPM Soil Simulation Guide

## Quick Start

Run MPM examples easily:
```bash
cd ~/Desktop/newton

# Simple helper script
./run_mpm.sh mpm_granular      # Sand/soil particles
./run_mpm.sh mpm_anymal         # Robot on terrain (MOST RELEVANT!)
./run_mpm.sh mpm_twoway_coupling # Rigid body interactions
./run_mpm.sh mpm_multi_material  # Multiple soil types
```

## MPM Examples Explained

### 1. `mpm_granular` - Basic Granular Materials
- **What it does**: Simulates sand/soil particles falling and piling
- **Physics**: MPM solver with material properties (friction, cohesion)
- **Use case**: Understanding how MPM particles behave
- **File**: `newton/examples/mpm/example_mpm_granular.py`

### 2. `mpm_anymal` - Robot + MPM Terrain ⭐ KEY EXAMPLE
- **What it does**: ANYmal quadruped walks on MPM sand terrain
- **Physics**: Two-way coupling between rigid body robot and MPM soil
- **Use case**: **This is your excavator equivalent!**
- **File**: `newton/examples/mpm/example_mpm_anymal.py`
- **Why relevant**: Shows exactly how to couple an articulated robot with MPM granular materials

### 3. `mpm_twoway_coupling` - Rigid Bodies + MPM
- **What it does**: Rigid objects interact with MPM materials
- **Physics**: Forces from MPM affect rigid bodies and vice versa
- **Use case**: Understanding bucket-soil interaction forces
- **File**: `newton/examples/mpm/example_mpm_twoway_coupling.py`

### 4. `mpm_multi_material` - Different Soil Types
- **What it does**: Multiple MPM materials (sand, soil, etc.) with different properties
- **Physics**: Each material has unique cohesion, friction, elasticity
- **Use case**: Simulating different soil conditions
- **File**: `newton/examples/mpm/example_mpm_multi_material.py`

## Adapting for Excavator

### Step 1: Study the ANYmal Example
The `mpm_anymal` example is your starting point because it shows:
- Loading a URDF robot (line 70-79)
- Creating MPM granular terrain
- Two-way coupling between robot and terrain
- Running simulation with control policy

### Step 2: Replace ANYmal with Your Excavator

Key changes needed in `example_mpm_anymal.py`:

```python
# Instead of ANYmal URDF
asset_path = newton.utils.download_asset("anybotics_anymal_c")
stage_path = str(asset_path / "urdf" / "anymal.urdf")

# Use your excavator URDF
stage_path = "/path/to/your/excavator.urdf"
```

### Step 3: Customize MPM Soil Properties

MPM materials have these key properties:
- **Young's modulus** (E): Soil stiffness
- **Poisson's ratio** (ν): Volume change under stress
- **Density** (ρ): Soil density (kg/m³)
- **Friction** (μ): Internal friction angle
- **Cohesion** (c): Soil cohesion for wet/sticky soil

Example soil types:
```python
# Dry sand
E = 1e6, ν = 0.3, ρ = 1600, μ = 0.5, c = 0

# Wet soil (sticky)
E = 2e6, ν = 0.35, ρ = 1800, μ = 0.7, c = 5000

# Clay
E = 5e6, ν = 0.4, ρ = 2000, μ = 0.4, c = 10000
```

### Step 4: Control Your Excavator

You can control joints either:
1. **Direct position control** (like ANYmal example uses)
2. **Velocity control**
3. **Torque control** (for force feedback from soil)

## Next Steps

1. **Run the examples** to see MPM in action
2. **Read the source code** of `mpm_anymal.py` to understand the structure
3. **Modify** to load your excavator URDF instead of ANYmal
4. **Tune** MPM parameters for realistic soil behavior
5. **Add control** logic for excavation motions

## Key Newton MPM Features

- ✅ GPU-accelerated (much faster than PhysX particles)
- ✅ True continuum mechanics (not rigid particles)
- ✅ Material properties (cohesion, friction, plasticity)
- ✅ Two-way coupling with rigid bodies
- ✅ Differentiable (can train RL policies)
- ✅ Scalable to large terrains

## Helpful Resources

- **Newton Docs**: https://newton-physics.github.io/newton/
- **Newton GitHub**: https://github.com/newton-physics/newton
- **Example Files**: `~/Desktop/newton/newton/examples/mpm/`
- **MPM Test**: `~/Desktop/newton/newton/tests/test_implicit_mpm.py`

## Comparison: Newton MPM vs PhysX Particles

| Feature | Newton MPM | PhysX Particles |
|---------|-----------|----------------|
| Physics Model | Continuum (MPM) | Rigid spheres |
| Particle Count | 100,000+ | ~10,000 (slow) |
| Soil Behavior | Realistic | Approximate |
| GPU Acceleration | ✅ Warp | ✅ CUDA |
| Cohesion/Stickiness | ✅ Built-in | ❌ Via friction only |
| Plasticity | ✅ Yes | ❌ No |
| Differentiable | ✅ Yes | ❌ No |
| Integration | Newton only | Isaac Sim |

## Summary

You now have access to **true MPM soil simulation** via Newton! The `mpm_anymal` example shows exactly how to couple a robot with MPM terrain, which is what you need for excavator simulation.

Start with running the examples, then adapt the ANYmal code to load your excavator URDF.
