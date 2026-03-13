# Excavator MPM Simulation + RL Training - Project Context
*Export date: 2026-02-12*
*Use this file to resume the project in a new chat.*

---

## Project Goal

Train a reinforcement learning policy for an excavator to perform autonomous digging using:
1. **Newton Physics Engine** with MPM (Material Point Method) soil simulation
2. **Behavioral Cloning** from YouTube expert operator videos
3. **PPO RL finetuning** in Newton simulation with soil-moved reward

---

## System Setup

- **OS**: Ubuntu (Linux 6.8.0-59-generic)
- **GPU**: NVIDIA RTX 6000 Ada Generation (47 GiB, sm_89)
- **Python package manager**: `~/snap/code/220/.local/bin/uv`
- **Newton install**: `/home/caee/Desktop/newton/` (with `.venv`)
- **Run command**: `cd ~/Desktop/newton && ~/snap/code/220/.local/bin/uv run python <script>.py`

---

## Key Files

| File | Purpose |
|------|---------|
| `/home/caee/Desktop/newton/excavator_mpm_soil.py` | **MAIN FILE** - Excavator + MPM soil simulation |
| `/home/caee/Desktop/newton/MPM_SOIL_PROPERTIES_GUIDE.md` | Complete guide to soil property options |
| `/home/caee/Desktop/newton/EXCAVATOR_MPM_GUIDE.md` | Overview guide for MPM examples |
| `/home/caee/Desktop/newton/run_mpm.sh` | Helper script to run Newton MPM examples |
| `/home/caee/Desktop/urdf_integration/RL/excavatorURDF/robot_fixed_cleaned.urdf` | Excavator URDF (17 joints) |

### Newton MPM Examples (reference)
```bash
cd ~/Desktop/newton
./run_mpm.sh mpm_anymal          # ANYmal robot on MPM terrain (KEY REFERENCE)
./run_mpm.sh mpm_granular        # Basic sand/soil
./run_mpm.sh mpm_twoway_coupling # Rigid body + MPM
./run_mpm.sh mpm_multi_material  # Multiple soil types
```

**ANYmal example source**: `newton/examples/mpm/example_mpm_anymal.py`
- Shows correct control gain setup: `builder.joint_target_ke[i] = 150`
- Shows RL observation/reward structure

---

## Current Simulation State

### What Works
- Newton MPM soil simulation running with 307,197 particles
- Excavator URDF loaded (17 joints)
- Two solvers running:
  - `SolverMuJoCo` for excavator rigid body dynamics
  - `SolverImplicitMPM` for soil particles
- Viewer displaying simulation
- Excavator positioned at side of soil pile `(0.0, 3.0, 0.5)`

### Soil Properties (currently set to high values by user)
```python
youngs_modulus = 20.0e6    # Pa - stiff
poissons_ratio = 0.4
density = 1800.0           # kg/m³
friction_angle = 40.0      # degrees
cohesion = 50000.0         # Pa - very sticky
```
*See MPM_SOIL_PROPERTIES_GUIDE.md for full options table*

### Open Issue: Weak Joint Tracking
Joints barely follow position control targets despite multiple fixes tried:
- Debug shows target pos changing but current pos barely moves
- Tried: `joint_target_ke=5000`, `joint_target_kd=500` (per-joint on builder)
- Tried: `njmax=200` to handle contact constraints
- Root cause: likely URDF joint limits, heavy inertia, or soil resistance overwhelming control

**Next thing to try**: Check URDF joint types/limits, or try torque control instead of position control

---

## Code Structure: excavator_mpm_soil.py

```python
class ExcavatorMPMExample:
    def __init__(self, viewer, voxel_size=0.05, particles_per_cell=3):
        # 1. Build ModelBuilder (up_axis=Z)
        # 2. Set joint config (armature, limit_ke/kd)
        # 3. Load excavator URDF at (0.0, 3.0, 0.5)
        # 4. Set per-joint control gains (ke=5000, kd=500)
        # 5. Create MPM soil (307k particles)
        # 6. Add ground plane
        # 7. Finalize model
        # 8. Create SolverMuJoCo (njmax=200) for excavator
        # 9. Create SolverImplicitMPM for soil
        # 10. setup_collider with zeros body_mass (kinematic)
        # 11. Create states and control

    def create_mpm_soil(self, builder, voxel_size, particles_per_cell):
        # 4m x 4m x 0.8m soil volume
        # Adds particles with add_particle()
        # Sets builder.mpm_E, mpm_nu, mpm_mu, mpm_lambda

    def apply_control(self):
        # Currently: sinusoidal joint motions
        # joints[0] = swing, [1] = boom, [2] = arm, [3] = bucket
        # Uses: self.control.joint_target_pos.numpy() + .assign()
        # Debug output every 2 seconds

    def simulate_excavator(self):
        # self.solver.step(state_0, state_1, control, contacts=None, dt)

    def simulate_sand(self):
        # self.mpm_solver.step() + project_outside()
```

### Critical API Notes
- **Warp arrays**: Use `.numpy()` to convert, NOT `wp.to_numpy()` (doesn't exist)
- **Control**: `self.control.joint_target_pos.numpy()` then `.assign()`
- **Control gains**: Set on `builder.joint_target_ke[i]` BEFORE `builder.finalize()`
- **Joint limits**: Use `self.model.joint_limit_lower.numpy()` (convert whole array first)
- **MPM solver**: `SolverImplicitMPM.Options()` (not `SolverImplicitMPMOptions`)

---

## RL Training Plan (Decided)

### Approach: BC + RL Finetuning (Hybrid)

**Data Source**: 2D monocular YouTube videos of expert excavator operators

**Why this works**: Excavator arm motion is mostly planar → 2D video is sufficient to extract joint angles

#### Stage 1: Video → Joint Angles
- **Tool**: SAM2 (Meta) for segment tracking OR custom keypoint detection
- **Practical start**: Label 200 frames with 4 keypoints → train YOLO-Pose → extract angles
- **Output**: `{states: [joint_pos], actions: [joint_vel], task_labels: ['dig','dump','swing']}`

#### Stage 2: Behavioral Cloning
- **Algorithm**: LSTM-BC or ACT (Action Chunking Transformer)
- **Input**: joint positions (17) + task context
- **Output**: joint velocity commands (4 main joints)
- **Training**: MSE loss on expert trajectories

#### Stage 3: RL Finetuning in Newton
- **Algorithm**: PPO with BC warmstart + KL regularization
- **Reward**: soil particles scooped per cycle - energy penalty
- **Environment**: Your existing `excavator_mpm_soil.py` adapted as RL env

### Hierarchical Action Structure
```
High-level (every 2s): choose skill {DIG, DUMP, MOVE, RETURN}
Low-level (every 0.1s): execute skill → joint commands
```

### Reward Function
```python
reward = (
    soil_in_bucket * 100.0      # Main: maximize soil
    - energy_used * 0.1          # Penalize waste
    - cycle_time * 1.0           # Encourage speed
    + dump_bonus * 500.0         # Complete full cycle
    - soil_lost * 50.0           # Don't spill
    - joint_jerk * 0.5           # Smooth motion
)
```

---

## Important Errors Fixed (Don't Repeat)

| Error | Wrong | Correct |
|-------|-------|---------|
| MPM Options | `newton.solvers.SolverImplicitMPMOptions()` | `SolverImplicitMPM.Options()` |
| Warp numpy | `wp.to_numpy(array)` | `array.numpy()` |
| Control attribute | `self.control.joint_act` | `self.control.joint_target_pos` |
| Viewer shutdown | `viewer.shutdown()` | Remove (doesn't exist) |
| Warp array indexing | `model.joint_type[i]` | `model.joint_type.numpy()[i]` |
| Control gains location | `JointDofConfig(target_ke=...)` | `builder.joint_target_ke[i] = ...` |

---

## Next Steps (Priority Order)

1. **Fix excavator movement** - Investigate URDF joint types/limits, try torque control
2. **Build RL environment** - Wrap current simulation as `gym.Env`
3. **Add soil reward** - Count MPM particles inside bucket bounding box
4. **Video processing pipeline** - Extract joint angles from YouTube videos
5. **BC training** - Train LSTM policy on extracted demonstrations
6. **PPO finetuning** - Optimize soil efficiency in Newton

---

## Newton MPM Technical Notes

- MPM solver only works with `SolverImplicitMPM`, NOT with MuJoCo solver
- MPM soil properties set on `builder` before `finalize()`:
  - `builder.mpm_E` = Young's modulus
  - `builder.mpm_nu` = Poisson's ratio
  - `builder.mpm_mu` = tan(friction_angle)
  - `builder.mpm_lambda` = cohesion
- Two-way coupling: `mpm_solver.setup_collider(body_mass=...)`
  - `wp.zeros_like(model.body_mass)` = kinematic (excavator pushes soil, not pushed back)
  - `model.body_mass` = dynamic (soil pushes back on excavator)
- CUDA graph capture only works for `grid_type="fixed"`, NOT `"sparse"`

---

## References

- Newton docs: https://newton-physics.github.io/newton/
- Newton GitHub: https://github.com/newton-physics/newton
- ANYmal MPM RL example: `~/Desktop/newton/newton/examples/mpm/example_mpm_anymal.py`
- MPM test: `~/Desktop/newton/newton/tests/test_implicit_mpm.py`
