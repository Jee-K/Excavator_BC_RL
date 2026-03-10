#!/usr/bin/env python3
"""
Excavator MPM Soil Simulation
Excavator digging in realistic MPM granular soil

Based on Newton's ANYmal MPM example, adapted for excavator.

Usage:
    cd ~/Desktop/newton
    ~/snap/code/220/.local/bin/uv run python excavator_mpm_soil.py
"""

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverImplicitMPM


class ExcavatorMPMExample:
    def __init__(self, viewer, voxel_size=0.03, particles_per_cell=3):
        # Simulation parameters
        self.fps = 10
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 20
        self.sim_dt = self.frame_dt / self.sim_substeps
        # ??? motivation to have more mpm substeps

        self.viewer = viewer
        self.device = wp.get_device()

        # Build the scene
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)

        # Set default joint and shape properties for excavator
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.1,
            limit_ke=1.0e4,
            limit_kd=1.0e2,
        )
        builder.default_shape_cfg.ke = 1.0e6  # 10x stiffer to prevent MPM particle penetration during fast motion
        builder.default_shape_cfg.kd = 1.0e4  # Increased damping proportionally
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.7

        # Register MPM custom attributes before adding particles
        SolverImplicitMPM.register_custom_attributes(builder)

        # Load excavator URDF
        # Use relative path from this script's directory

        # Load excavator URDF
        excavator_urdf = "./excavatorURDF/excavator_boxy.urdf"
        print(f"Loading excavator from: {excavator_urdf}")
        
        # Position excavator at the side of soil pile so it can reach in
        # Soil is centered at origin, 4m x 4m, so excavator at x=-3.0 is at the edge
        # Set position control gains for each excavator joint (like ANYmal example)
        # Using much higher gains for heavy excavator
        # Correctly determine control slot range for gains (skip root DOFs)
        control_start = len(builder.joint_target_ke)
        # After add_urdf, record control end
        builder.add_urdf(
            excavator_urdf,
            xform=wp.transform(wp.vec3(0.0, 3.0, 0.5), wp.quat_identity()),
            floating=True,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            ignore_inertial_definitions=False,
        )
        control_end = len(builder.joint_target_ke)
        print(f"[INFO] Control range for excavator: {control_start} → {control_end}")

        # Apply gains only to actual control slots (not floating base)
        # these values intended to be quite high
        for i in range(control_start, control_end):
            builder.joint_target_ke[i] = 5000.0 # we could choose to make these *extremely* high and rely on urdf values
            builder.joint_target_kd[i] = 500.0
        builder.joint_target_ke[8] = 15000.0 # artificially increase boom gain, since it seems to be struggling

        # Create MPM soil terrain
        print("Creating MPM soil terrain...")
        self.create_mpm_soil(builder, voxel_size, particles_per_cell)

        # Add ground plane for collision
        builder.add_shape_plane(
            body=-1,
            cfg=newton.ModelBuilder.ShapeConfig(mu=0.5, density=0.0),
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            width=20.0,
            length=20.0,
        )

        # Add dump bucket (goal area)
        # Excavator is at (0, 3, 0.5) facing soil at origin.
        # Bucket is to the side of the excavator so it can swing and dump.
        self.bucket_center = np.array([-4.0, 3.0, 0.0])
        self.bucket_inner_half = np.array([0.75, 1.5, 0.8])  # 1.5m x 1.5m x 1.6m inner
        self.add_dump_bucket(builder)

        # Finalize model
        self.model = builder.finalize()
        self.model.gravity.assign(wp.array([wp.vec3(0.0, 0.0, -9.81)], dtype=wp.vec3))

        # Print joint information
        print(f"\nExcavator has {self.model.joint_count} joints:")

        # Convert arrays to numpy for indexing
        joint_types = self.model.joint_type.numpy() if hasattr(self.model, 'joint_type') else None
        joint_lower = self.model.joint_limit_lower.numpy() if hasattr(self.model, 'joint_limit_lower') else None
        joint_upper = self.model.joint_limit_upper.numpy() if hasattr(self.model, 'joint_limit_upper') else None

        print(joint_types, joint_lower, joint_upper)

        joint_names = []
        # TODO: fix all this
        for i in range(self.model.joint_count):
            # works for this setup, at least
            name = self.model.joint_name[i] if hasattr(self.model, 'joint_name') else f"joint_{i}"
            joint_names.append(name)

            # Get joint info
            jtype = joint_types[i] if joint_types is not None else -1
            lower = joint_lower[i] if joint_lower is not None else 0.0
            upper = joint_upper[i] if joint_upper is not None else 0.0

            print(f"  [{i}] {name:20s} type={jtype} limits=[{lower:.3f}, {upper:.3f}]")

        # MPM solver options
        mpm_options = SolverImplicitMPM.Options()
        mpm_options.voxel_size = voxel_size
        mpm_options.tolerance = 1.0e-5
        mpm_options.transfer_scheme = "pic" # ???
        mpm_options.grid_type = "sparse"  # "sparse" or "fixed"
        mpm_options.strain_basis = "P0"
        mpm_options.max_iterations = 50
        mpm_options.critical_fraction = 0.0

        # Initialize MuJoCo solver for excavator rigid body dynamics
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            ls_iterations=50,
            njmax=200,  # Increased to handle contact constraints
        )

        # Initialize MPM solver for soil
        self.mpm_solver = SolverImplicitMPM(self.model, mpm_options)

        # Configure collider: treat excavator bodies as kinematic
        self.mpm_solver.setup_collider(
            body_mass=wp.zeros_like(self.model.body_mass),
        )

        # Simulation state
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        # Initialize forward kinematics
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        # Control setup
        self.control = self.model.control()
        self._joint_target_host = self.control.joint_target_pos.numpy()

        print(f"\nControl gains (set per joint on builder):")
        print(f"  joint_target_ke: 5000.0")
        print(f"  joint_target_kd: 500.0")

        # Score tracking
        self.particles_in_bucket = 0
        self.total_particles = self.model.particle_count
        self.score_print_interval = 5.0  # print score every 5 sim seconds
        self.last_score_print = 0.0

        # Set viewer
        self.viewer.set_model(self.model)
        self.viewer.show_particles = True

        # CUDA graph capture for speed
        self.capture()

        print("\n" + "="*60)
        print("EXCAVATOR MPM SIMULATION")
        print("="*60)
        print("Controls:")
        print("  Press SPACE to start/stop")
        print("  Excavator joints can be controlled via code (see apply_control)")
        print(f"\nGoal: Move soil into the dump bucket at ({self.bucket_center[0]:.0f}, {self.bucket_center[1]:.0f}, 0)")
        print(f"  Bucket size: {self.bucket_inner_half[0]*2:.1f}m x {self.bucket_inner_half[1]*2:.1f}m")
        print(f"  Total soil particles: {self.total_particles:,}")
        print("="*60)

    def add_dump_bucket(self, builder):
        """Add a static dump bucket (goal area) to the scene.

        Bucket is an open-top box made of 5 box shapes (bottom + 4 walls).
        Positioned to the left of the excavator so it can swing and dump.
        """
        bx, by, bz = self.bucket_center
        iw, il, idepth = self.bucket_inner_half  # inner half-extents
        T = 0.1  # wall thickness

        cfg = newton.ModelBuilder.ShapeConfig(
            ke=1.0e5, kd=1.0e3, kf=1.0e3, mu=0.6, density=0.0
        )

        # Bottom plate
        builder.add_shape_box(
            body=-1, cfg=cfg,
            xform=wp.transform(wp.vec3(bx, by, bz + T * 0.5), wp.quat_identity()),
            hx=iw + T, hy=il + T, hz=T * 0.5,
        )
        # Front wall  (negative Y)
        builder.add_shape_box(
            body=-1, cfg=cfg,
            xform=wp.transform(wp.vec3(bx, by - il - T * 0.5, bz + T + idepth * 0.5), wp.quat_identity()),
            hx=iw + T, hy=T * 0.5, hz=idepth * 0.5,
        )
        # Back wall (positive Y)
        builder.add_shape_box(
            body=-1, cfg=cfg,
            xform=wp.transform(wp.vec3(bx, by + il + T * 0.5, bz + T + idepth * 0.5), wp.quat_identity()),
            hx=iw + T, hy=T * 0.5, hz=idepth * 0.5,
        )
        # Left wall (negative X)
        builder.add_shape_box(
            body=-1, cfg=cfg,
            xform=wp.transform(wp.vec3(bx - iw - T * 0.5, by, bz + T + idepth * 0.5), wp.quat_identity()),
            hx=T * 0.5, hy=il, hz=idepth * 0.5,
        )
        # Right wall (positive X)
        builder.add_shape_box(
            body=-1, cfg=cfg,
            xform=wp.transform(wp.vec3(bx + iw + T * 0.5, by, bz + T + idepth * 0.5), wp.quat_identity()),
            hx=T * 0.5, hy=il, hz=idepth * 0.5,
        )

        print(f"\nDump bucket added:")
        print(f"  Center: ({bx}, {by}, {bz})")
        print(f"  Inner size: {iw*2:.1f}m x {il*2:.1f}m x {idepth*2:.1f}m")
        print(f"  Goal: swing excavator left and dump soil into bucket")

    def count_particles_in_bucket(self):
        """Count MPM particles currently inside the dump bucket volume."""
        if self.model.particle_count == 0:
            return 0

        positions = self.state_0.particle_q.numpy()  # shape: [N, 3]

        bx, by, bz = self.bucket_center
        iw, il, idepth = self.bucket_inner_half
        T = 0.1

        # Inner bounding box of the bucket
        x_min, x_max = bx - iw, bx + iw
        y_min, y_max = by - il, by + il
        z_min, z_max = bz + T, bz + T + idepth * 2

        inside = (
            (positions[:, 0] >= x_min) & (positions[:, 0] <= x_max) &
            (positions[:, 1] >= y_min) & (positions[:, 1] <= y_max) &
            (positions[:, 2] >= z_min) & (positions[:, 2] <= z_max)
        )
        return int(np.count_nonzero(inside))

    def create_mpm_soil(self, builder, voxel_size, particles_per_cell):
        soil_width = 2.0
        soil_length = 2.0
        soil_height = 0.8

        nx = int(round(soil_width / voxel_size))
        ny = int(round(soil_length / voxel_size))
        nz = int(round(soil_height / voxel_size))

        density = 1800.0
        particle_mass = density * (voxel_size ** 3) / particles_per_cell

        # Build voxel origins in a vectorized way
        xs = -soil_width / 2 + np.arange(nx, dtype=np.float32) * voxel_size
        ys = -soil_length / 2 + np.arange(ny, dtype=np.float32) * voxel_size
        zs = 0.05 + np.arange(nz, dtype=np.float32) * voxel_size

        grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
        n_cells = grid.shape[0]

        # Vectorized jitter for all particles
        rng = np.random.default_rng(42)
        jitter = rng.random((n_cells, particles_per_cell, 3), dtype=np.float32) * voxel_size
        positions = (grid[:, None, :] + jitter).reshape(-1, 3)

        zero_vel = wp.vec3(0.0, 0.0, 0.0)
        for p in positions:
            builder.add_particle(
                pos=wp.vec3(float(p[0]), float(p[1]), float(p[2])),
                vel=zero_vel,
                mass=particle_mass,
            )

        # Elastic material
        E = 20e6
        nu = 0.4
        builder.mpm_E = E
        builder.mpm_nu = nu
        builder.mpm_mu = E / (2 * (1 + nu))
        builder.mpm_lambda = E * nu / ((1 + nu) * (1 - 2 * nu))

        print(f"Soil properties:")
        # print(f"  Young's modulus: {youngs_modulus/1e6:.1f} MPa")
        # print(f"  Poisson's ratio: {poissons_ratio}")
        print(f"  Density: {density} kg/m³")
        # print(f"  Friction angle: {friction_angle}°")
        # print(f"  Cohesion: {cohesion/1000:.1f} kPa")

    def capture(self):
        """Capture CUDA graphs for performance"""
        self.excavator_graph = None
        self.sand_graph = None

        if wp.get_device().is_cuda:
            # Capture excavator simulation
            with wp.ScopedCapture() as capture:
                self.simulate_excavator()
            self.excavator_graph = capture.graph

            # Capture sand simulation (if using fixed grid)
            if self.mpm_solver.grid_type == "fixed":
                with wp.ScopedCapture() as capture:
                    self.simulate_sand()
                self.sand_graph = capture.graph

    def simulate_excavator(self):
        """Simulate excavator rigid body dynamics"""
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, contacts=None, dt=self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def simulate_sand(self):
        """Simulate MPM sand physics"""
        mpm_bonus_factor = 2 # giving it a bonus for particle simulation
        mpm_dt = self.sim_dt / mpm_bonus_factor
        for _ in range(self.sim_substeps * mpm_bonus_factor): 
            self.mpm_solver.step(self.state_0, self.state_1, None, None, mpm_dt)
            self.mpm_solver.project_outside(self.state_1, self.state_1, mpm_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def apply_control(self):
        """Apply control to excavator joints

        Modify this function to control the excavator.
        Sets target positions for joint position control.
        """
        current_pos = self.state_0.joint_q.numpy()
        pos = self._joint_target_host

        if pos.shape[0] < 4:
            return

        t = self.sim_time
        # in robot_fixed_alternate
        # grounded versions, +1 for float:
        # 0 - base, left then right
        # 1 - front left-right rotator, right then left
        # 2 - back arm, down then up # you really have to exceed the real limit to achieve this, something like -3.5 is necessary to get a hover
        # 3 - middle arm, outwards then inwards # 0.8 usually hovers at full extension
        # 4 - bucket, tight then open

        # in robot_fixed_alternate and ungrounded
        # 0 - x
        # 1 - y
        # 2 - z
        # 3-6 - quaternion related partwise-rotation nonsense
        # 7 - base
        # 8 - front l/r rotator
        # 9 - back arm
        # 10 - middle arm
        # 11 - bucket

        # new scheme of loading joints for alt ungrounded
        # 6 - base rotator?
        # 7 - front left-right rotator
        # 8 - back arm
        # 9 - middle arm
        # 10 - bucket

        # pos[2] = .5
        pos[6] = 0
        pos[7] = 0
        pos[8] = -.5-np.sin(t / 5)
        pos[9] = -np.sin(t / 3)
        pos[10] = 5
        # pos[10] = 10
        # pos[6] = t
        # pos[6] = 10 * np.sin(t / 2)

        self.control.joint_target_pos.assign(pos)

        # Debug output every 2 seconds
        if int(t) % 2 == 0 and t - int(t) < self.frame_dt:
            
            print(pos, current_pos)
            print(f"\n[t={t:.1f}s] Control Debug:")
            
            print(f"  Target pos:          [{', '.join([format(float(x), '6.3f') for x in pos])}]")
            print(f"  Current pos: [{', '.join([format(float(x), '6.3f') for x in current_pos])}]")

    def step(self):
        """Step the simulation forward"""
        # Apply control
        self.apply_control()

        # Simulate excavator
        if self.excavator_graph:
            wp.capture_launch(self.excavator_graph)
        else:
            self.simulate_excavator()

        # Simulate sand
        if self.sand_graph:
            wp.capture_launch(self.sand_graph)
        else:
            self.simulate_sand()

        # Update time
        self.sim_time += self.frame_dt

        # Count and report particles in bucket
        if self.sim_time - self.last_score_print >= self.score_print_interval:
            self.particles_in_bucket = self.count_particles_in_bucket()
            pct = 100.0 * self.particles_in_bucket / max(self.total_particles, 1)
            print(f"\n[t={self.sim_time:.1f}s] SCORE: {self.particles_in_bucket:,} / {self.total_particles:,} "
                  f"particles in bucket ({pct:.1f}%)")
            self.last_score_print = self.sim_time

        # Update viewer
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


def main():
    """Main entry point"""
    # Initialize viewer with Newton's standard args
    viewer, args = newton.examples.init()

    # Use default MPM parameters
    voxel_size = .03 # .05
    particles_per_cell = 10 # 3

    # Create simulation
    example = ExcavatorMPMExample(
        viewer,
        voxel_size=voxel_size,
        particles_per_cell=particles_per_cell
    )

    # Run simulation loop
    try:
        while viewer.is_running():
            example.step()
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")


if __name__ == "__main__":
    main()
