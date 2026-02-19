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
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps

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
        excavator_urdf = "/home/caee/Desktop/urdf_integration/RL/excavatorURDF/robot_fixed_cleaned.urdf"
        print(f"Loading excavator from: {excavator_urdf}")
        
        # Position excavator at the side of soil pile so it can reach in
        # Soil is centered at origin, 4m x 4m, so excavator at x=-3.0 is at the edge
        articulation_start = builder.joint_count
        builder.add_urdf(
            excavator_urdf,
            xform=wp.transform(
                wp.vec3(0.0, 3.0, 0.5),  # Side of soil pile, at edge
                wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.0)  # Facing soil
            ),
            floating=False,  # Excavator is grounded
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            ignore_inertial_definitions=False,
        )

        # Set position control gains for each excavator joint (like ANYmal example)
        # Using much higher gains for heavy excavator
        for i in range(articulation_start, builder.joint_count):
            builder.joint_target_ke[i] = 5000.0  # Position control stiffness (10x higher for excavator)
            builder.joint_target_kd[i] = 500.0   # Position control damping

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
        # Bucket is to the LEFT side of the excavator so it can swing and dump.
        self.bucket_center = np.array([-4.0, 3.0, 0.0])
        self.bucket_inner_half = np.array([0.75, 1.5, 0.8])  # 1.5m x 1.5m x 1.6m inner
        self.add_dump_bucket(builder)

        # Finalize model
        self.model = builder.finalize()
        self.model.ground = True

        # Print joint information
        print(f"\nExcavator has {self.model.joint_count} joints:")

        # Convert arrays to numpy for indexing
        joint_types = self.model.joint_type.numpy() if hasattr(self.model, 'joint_type') else None
        joint_lower = self.model.joint_limit_lower.numpy() if hasattr(self.model, 'joint_limit_lower') else None
        joint_upper = self.model.joint_limit_upper.numpy() if hasattr(self.model, 'joint_limit_upper') else None

        joint_names = []
        for i in range(self.model.joint_count):
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
        mpm_options.transfer_scheme = "pic"
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
        return int(np.sum(inside))

    def create_mpm_soil(self, builder, voxel_size, particles_per_cell):
        """Create MPM soil particles"""
        # Soil region parameters
        soil_width = 4.0   # 4m wide
        soil_length = 4.0  # 4m long
        soil_height = 0.8  # 80cm deep

        # Calculate number of particles
        volume = soil_width * soil_length * soil_height
        voxel_volume = voxel_size ** 3
        num_voxels = int(volume / voxel_volume)
        num_particles = num_voxels * particles_per_cell

        print(f"Creating {num_particles:,} soil particles...")
        print(f"  Soil dimensions: {soil_width}m x {soil_length}m x {soil_height}m")
        print(f"  Voxel size: {voxel_size}m")
        print(f"  Particles per voxel: {particles_per_cell}")

        # Generate particle positions
        positions = []
        for i in range(num_particles):
            # Random position within soil volume
            x = np.random.uniform(-soil_width/2, soil_width/2)
            y = np.random.uniform(-soil_length/2, soil_length/2)
            z = np.random.uniform(0.05, soil_height)  # Slight offset from ground
            positions.append([x, y, z])

        positions = np.array(positions, dtype=np.float32)

        # Soil material properties (wet, cohesive soil)
        youngs_modulus = 20.0e6    # Pa (soft soil)
        poissons_ratio = 0.4     # Typical for soil
        density = 1800.0          # kg/m³ (wet soil)
        friction_angle = 40.0     # degrees (internal friction)
        cohesion = 50000.0         # Pa (wet soil cohesion)

        # Add particles to model
        for i, pos in enumerate(positions):
            builder.add_particle(
                pos=wp.vec3(pos[0], pos[1], pos[2]),
                vel=wp.vec3(0.0, 0.0, 0.0),
                mass=density * (voxel_size ** 3) / particles_per_cell
            )

        # Set MPM material properties
        builder.mpm_E = youngs_modulus
        builder.mpm_nu = poissons_ratio
        builder.mpm_mu = np.tan(np.radians(friction_angle))
        builder.mpm_lambda = cohesion

        print(f"Soil properties:")
        print(f"  Young's modulus: {youngs_modulus/1e6:.1f} MPa")
        print(f"  Poisson's ratio: {poissons_ratio}")
        print(f"  Density: {density} kg/m³")
        print(f"  Friction angle: {friction_angle}°")
        print(f"  Cohesion: {cohesion/1000:.1f} kPa")

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
        for _ in range(self.sim_substeps):
            self.mpm_solver.step(self.state_0, self.state_1, None, None, self.sim_dt)
            self.mpm_solver.project_outside(self.state_1, self.state_1, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def apply_control(self):
        """Apply control to excavator joints

        Modify this function to control the excavator.
        Sets target positions for joint position control.
        """
        t = self.sim_time

        # Sinusoidal joint motions for demonstration
        if self.model.joint_count >= 4:
            positions = self.control.joint_target_pos.numpy()
            current_pos = self.state_0.joint_q.numpy()

            # Slow, gentle motions to see excavator interact with soil
            positions[0] = 0.15 * np.sin(t * 0.4)   # Body rotation (swing)
            positions[1] = 0.25 * np.sin(t * 0.25)  # Upper boom (up/down)
            positions[2] = 0.3 * np.sin(t * 0.3)    # Lower boom (extend/retract)
            positions[3] = 0.2 * np.sin(t * 0.35)   # Bucket (curl/dump)

            self.control.joint_target_pos.assign(positions)

            # Debug output every 2 seconds
            if int(t) % 2 == 0 and t - int(t) < self.frame_dt:
                print(f"\n[t={t:.1f}s] Control Debug:")
                print(f"  Target pos [0-3]: [{positions[0]:.3f}, {positions[1]:.3f}, {positions[2]:.3f}, {positions[3]:.3f}]")
                print(f"  Current pos [0-3]: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}, {current_pos[3]:.3f}]")

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
    voxel_size = 0.05
    particles_per_cell = 3

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
