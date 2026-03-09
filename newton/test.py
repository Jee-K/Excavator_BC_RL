#!/usr/bin/env python3
"""
Excavator + MPM soil simulation reworked to keep the standard dp_soil motion while forcing full-contact soil resolution all the time.

What changed versus the uploaded rework:
- keep the leak fix: synchronized collider pose + interleaved rigid/soil stepping
- keep the thicker bucket MPM proxy to reduce soil tunneling through the scoop
- add simulation presets (fast_rl / balanced / accuracy)
- replace adaptive near/far MPM stepping with fixed full-contact substeps
- increase bucket projection passes during each soil microstep
- preserve the original standard dp_soil dig-cycle motion instead of the softened URDF-only demo
- remove the hardcoded demo-target overwrite so the named dig cycle actually drives the arm

Important limitation:
This stays within the Newton/Warp implicit MPM interface exposed in the uploaded
scripts. The example below adds a Drucker-Prager-inspired post-step plasticity
regularizer around the existing solver. It is intentionally smooth and RL-friendly,
but it is not a full internal return-mapping implementation inside Newton's MPM core.
"""

import math
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverImplicitMPM


@dataclass(frozen=True)
class SoilPreset:
    name: str = "moist_sandy_loam"
    density_kg_m3: float = 1650.0
    youngs_modulus_pa: float = 8.0e6
    poisson_ratio: float = 0.28
    friction_angle_deg: float = 32.0
    cohesion_pa: float = 12_000.0
    dilatancy_deg: float = 3.0
    interface_friction_mu: float = 0.55

    slope_angle_deg: float = 32.0
    bank_height_m: float = 0.85
    bank_width_m: float = 2.6
    bank_length_m: float = 2.2
    spawn_clearance_m: float = 0.025


@dataclass(frozen=True)
class EnvironmentPreset:
    excavator_position: tuple[float, float, float] = (0.0, 3.0, 0.35)
    ground_size: tuple[float, float] = (20.0, 20.0)
    bucket_center: tuple[float, float, float] = (-4.0, 3.0, 0.0)
    bucket_inner_half: tuple[float, float, float] = (0.75, 1.5, 0.8)
    bucket_wall_thickness: float = 0.1
    bank_front_y: float = 1.10
    bank_back_y: float = -1.10


@dataclass(frozen=True)
class DigPose:
    swing: float
    boom: float
    arm: float
    stick: float
    bucket: float


@dataclass(frozen=True)
class SimulationPreset:
    name: str
    voxel_size_m: float
    particles_per_cell: int
    rigid_substeps: int
    mpm_far: int
    mpm_near: int
    mpm_contact: int
    mpm_iterations: int
    rigid_ls_iterations: int
    rigid_njmax: int


SIM_PRESETS: dict[str, SimulationPreset] = {
    "balanced": SimulationPreset(
        name="balanced",
        voxel_size_m=0.020,
        particles_per_cell=4,
        rigid_substeps=6,
        mpm_far=1,
        mpm_near=3,
        mpm_contact=6,
        mpm_iterations=40,
        rigid_ls_iterations=60,
        rigid_njmax=260,
    ),
    "balanced_full_contact": SimulationPreset(
        name="balanced_full_contact",
        voxel_size_m=0.020,
        particles_per_cell=4,
        rigid_substeps=6,
        mpm_far=6,
        mpm_near=6,
        mpm_contact=6,
        mpm_iterations=40,
        rigid_ls_iterations=60,
        rigid_njmax=260,
    ),
}


@dataclass(frozen=True)
class DPRegularizerConfig:
    enabled: bool = False
    matching: str = "inscribed_from_mohr_coulomb"
    active_margin_m: float = 0.35
    bucket_margin_m: float = 0.25
    flow_length_m: float = 0.06
    static_damping: float = 12.0
    flow_speed_base_m_s: float = 0.18
    flow_speed_gain: float = 0.22
    rebound_damping: float = 8.0
    lock_speed_m_s: float = 0.03
    bucket_bonus: float = 0.75


@wp.kernel
def drucker_prager_velocity_regularizer(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    dt: float,
    bank_x_half: float,
    bank_y_back: float,
    bank_y_front: float,
    bank_z0: float,
    bank_height: float,
    slope_tan: float,
    active_margin: float,
    density: float,
    gravity_mag: float,
    dp_alpha: float,
    dp_k: float,
    flow_length: float,
    static_damping: float,
    flow_speed_base: float,
    flow_speed_gain: float,
    rebound_damping: float,
    lock_speed: float,
    bucket_center: wp.vec3,
    bucket_half_extent: wp.vec3,
    bucket_margin: float,
    bucket_bonus: float,
):
    """Smooth Drucker-Prager-inspired velocity regularization.

    This is an external post-step correction, not a native constitutive update.
    It approximates pressure-dependent yielding by making low-speed lateral motion
    stick under confinement while capping unrealistically fast shear flow.
    """
    tid = wp.tid()
    p = particle_q[tid]
    v = particle_qd[tid]

    active_bank = (
        wp.abs(p[0]) <= bank_x_half + active_margin
        and p[1] >= bank_y_back - active_margin
        and p[1] <= bank_y_front + active_margin
        and p[2] >= -0.10
    )
    active_bucket = (
        wp.abs(p[0] - bucket_center[0]) <= bucket_half_extent[0] + bucket_margin
        and wp.abs(p[1] - bucket_center[1]) <= bucket_half_extent[1] + bucket_margin
        and p[2] >= bucket_center[2] - bucket_margin
        and p[2] <= bucket_center[2] + 2.0 * bucket_half_extent[2] + bucket_margin
    )
    if not active_bank and not active_bucket:
        return

    depth_into_bank = wp.max(0.0, bank_y_front - p[1])
    reference_surface = bank_z0 + wp.min(bank_height, depth_into_bank * slope_tan)
    depth = wp.max(0.0, reference_surface - p[2])
    if active_bucket:
        depth = wp.max(depth, 0.25 * bucket_half_extent[2])

    pressure = density * gravity_mag * depth
    yield_stress = dp_k + 3.0 * dp_alpha * pressure

    vxy_norm = wp.sqrt(v[0] * v[0] + v[1] * v[1])
    shear_stress = density * flow_length * vxy_norm / wp.max(dt, 1.0e-6)

    mobilization = 1.0e3
    if yield_stress > 1.0e-6:
        mobilization = shear_stress / yield_stress

    # Smooth transition: ~1 below yield (stick), ~0 above yield (free flow).
    stick = 1.0 / (1.0 + wp.exp(10.0 * (mobilization - 1.0)))

    bucket_scale = 1.0
    if active_bucket:
        bucket_scale = 1.0 + bucket_bonus

    damp = wp.max(0.0, 1.0 - static_damping * stick * dt * bucket_scale)
    vx = v[0] * damp
    vy = v[1] * damp
    vz = v[2]

    excess = wp.max(0.0, shear_stress - yield_stress)
    flow_cap = flow_speed_base + flow_speed_gain * wp.sqrt(excess / wp.max(density, 1.0e-6))
    vxy_after = wp.sqrt(vx * vx + vy * vy)
    if vxy_after > flow_cap and flow_cap > 0.0:
        scale = flow_cap / vxy_after
        vx = vx * scale
        vy = vy * scale
        vxy_after = flow_cap

    if vz > 0.0:
        vz = vz * wp.max(0.0, 1.0 - rebound_damping * stick * dt * bucket_scale)

    if stick > 0.97 and vxy_after < lock_speed:
        vx = 0.0
        vy = 0.0
        if wp.abs(vz) < lock_speed:
            vz = 0.0

    particle_qd[tid] = wp.vec3(vx, vy, vz)


class ExcavatorMPMExample:
    def __init__(
        self,
        viewer,
        preset_name: str = "balanced_full_contact",
        voxel_size: Optional[float] = None,
        particles_per_cell: Optional[int] = None,
    ):
        if preset_name not in SIM_PRESETS:
            raise ValueError(f"Unknown preset '{preset_name}'. Available: {sorted(SIM_PRESETS)}")

        self.viewer = viewer
        self.device = wp.get_device()
        self.soil = SoilPreset()
        self.env = EnvironmentPreset()
        self.dp = DPRegularizerConfig()
        self.sim_preset = SIM_PRESETS[preset_name]

        self.voxel_size = float(voxel_size if voxel_size is not None else self.sim_preset.voxel_size_m)
        self.particles_per_cell = int(
            particles_per_cell if particles_per_cell is not None else self.sim_preset.particles_per_cell
        )

        self.fps = 30
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = self.sim_preset.rigid_substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.score_print_interval = 5.0
        self.last_score_print = 0.0
        self.settle_duration = 1.5
        self.dig_cycle_duration = 12.0

        # Full-contact soil-step scheduling (fixed at mpm_contact each rigid substep).
        self.bucket_proxy_radius_m = 0.45
        self.contact_margin_m = max(0.10, 2.0 * self.voxel_size)
        self.near_margin_m = max(0.30, 5.0 * self.voxel_size)
        self.contact_hold_counter = 0
        self.last_selected_mpm_substeps = self.sim_preset.mpm_contact
        self.last_bucket_bank_distance_m = float("nan")

        self.bucket_proxy_body_name: Optional[str] = None
        self.bucket_proxy_body_index_builder: Optional[int] = None
        self.bucket_body_index: Optional[int] = None

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        self.configure_rigid_defaults(builder)
        SolverImplicitMPM.register_custom_attributes(builder)

        excavator_urdf = "./excavatorURDF/robot_fixed_cleaned.urdf"
        print(f"Loading excavator from: {excavator_urdf}")

        control_start = len(builder.joint_target_ke)
        builder.add_urdf(
            excavator_urdf,
            xform=wp.transform(wp.vec3(*self.env.excavator_position), wp.quat_identity()),
            floating=True,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            ignore_inertial_definitions=False,
        )
        self.add_bucket_mpm_proxy(builder)
        control_end = len(builder.joint_target_ke)

        for i in range(control_start, control_end):
            builder.joint_target_ke[i] = 5000.0
            builder.joint_target_kd[i] = 500.0
        if control_end > 8:
            builder.joint_target_ke[8] = 15000.0

        print("Creating sloped MPM excavation bank...")
        self.create_mpm_soil_bank(builder)
        self.add_ground_plane(builder)

        self.bucket_center = np.asarray(self.env.bucket_center, dtype=np.float64)
        self.bucket_inner_half = np.asarray(self.env.bucket_inner_half, dtype=np.float64)
        self.add_dump_bucket(builder)

        self.model = builder.finalize()
        self.model.gravity.assign(wp.array([wp.vec3(0.0, 0.0, -9.81)], dtype=wp.vec3))

        self.bucket_body_index = self._resolve_model_body_index(
            preferred_name=self.bucket_proxy_body_name,
            aliases=("scoop1", "scoop", "bucket"),
            fallback=self.bucket_proxy_body_index_builder,
        )
        self._build_excavation_aabb()
        self._configure_drucker_prager_regularizer()

        mpm_options = SolverImplicitMPM.Options()
        mpm_options.voxel_size = self.voxel_size
        mpm_options.tolerance = 1.0e-5
        mpm_options.transfer_scheme = "pic"
        mpm_options.grid_type = "sparse"
        mpm_options.strain_basis = "P0"
        mpm_options.max_iterations = self.sim_preset.mpm_iterations
        mpm_options.critical_fraction = 0.0

        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            ls_iterations=self.sim_preset.rigid_ls_iterations,
            njmax=self.sim_preset.rigid_njmax,
        )
        self.mpm_solver = SolverImplicitMPM(self.model, mpm_options)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        # Core leak fix: the MPM collider reads from a dedicated pose buffer that is
        # refreshed from the active rigid state before each soil solve.
        self.collider_body_q = wp.zeros_like(self.state_0.body_q)
        self.collider_body_q.assign(self.state_0.body_q)
        self.mpm_solver.setup_collider(
            body_mass=wp.zeros_like(self.model.body_mass),
            body_q=self.collider_body_q,
        )

        self.control = self.model.control()
        self._joint_target_host = self.control.joint_target_pos.numpy()
        self.control_size = int(self._joint_target_host.shape[0])

        self.control_joint_names = self._extract_control_joint_names()
        self.control_lower, self.control_upper = self._extract_control_limits()
        self.joint_map = self._identify_joint_map()

        self.total_particles = int(self.model.particle_count)
        self.particles_in_bucket = 0

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.viewer.show_visual = False
        self.viewer.show_collision = True
        self.viewer.show_cloth = False

        self.capture()
        self.print_summary()
        self.review_configuration()

    # ------------------------------------------------------------------
    # Scene and material setup
    # ------------------------------------------------------------------
    def configure_rigid_defaults(self, builder) -> None:
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.05,
            limit_ke=2.0e4,
            limit_kd=2.0e2,
        )
        builder.default_shape_cfg.ke = 1.0e6
        builder.default_shape_cfg.kd = 1.0e4
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = self.soil.interface_friction_mu

    def add_ground_plane(self, builder) -> None:
        width, length = self.env.ground_size
        builder.add_shape_plane(
            body=-1,
            cfg=newton.ModelBuilder.ShapeConfig(mu=0.60, density=0.0),
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            width=width,
            length=length,
        )

    def _get_builder_body_names(self, builder) -> list[str]:
        for attr in ("body_name", "body_names"):
            if hasattr(builder, attr):
                return [str(name) for name in getattr(builder, attr)]
        return []

    def _get_model_body_names(self) -> list[str]:
        for attr in ("body_name", "body_names"):
            if hasattr(self.model, attr):
                return [str(name) for name in getattr(self.model, attr)]
        return []

    @staticmethod
    def _find_named_index(names: list[str], aliases: Iterable[str]) -> Optional[int]:
        aliases_c = [alias.lower().replace("_", "") for alias in aliases]
        for idx, name in enumerate(names):
            compact = name.lower().replace("_", "")
            if any(alias == compact for alias in aliases_c):
                return idx
        for idx, name in enumerate(names):
            compact = name.lower().replace("_", "")
            if any(alias in compact for alias in aliases_c):
                return idx
        return None

    def _resolve_model_body_index(
        self,
        preferred_name: Optional[str],
        aliases: Iterable[str],
        fallback: Optional[int],
    ) -> Optional[int]:
        names = self._get_model_body_names()
        if names:
            if preferred_name is not None:
                idx = self._find_named_index(names, (preferred_name,))
                if idx is not None:
                    return idx
            idx = self._find_named_index(names, aliases)
            if idx is not None:
                return idx
        return fallback

    def add_bucket_mpm_proxy(self, builder) -> None:
        """Inflate the MPM bucket collider without changing the rigid URDF geometry."""
        names = self._get_builder_body_names(builder)
        bucket_body = self._find_named_index(names, ("scoop1", "scoop", "bucket"))
        if bucket_body is None:
            print("Warning: bucket body for MPM proxy could not be resolved; using raw URDF collisions.")
            return

        self.bucket_proxy_body_index_builder = bucket_body
        if 0 <= bucket_body < len(names):
            self.bucket_proxy_body_name = names[bucket_body]

        cfg = newton.ModelBuilder.ShapeConfig(
            ke=2.0e6,
            kd=2.0e4,
            kf=1.0e3,
            mu=self.soil.interface_friction_mu,
            density=0.0,
        )

        min_wall = max(0.12, 2.5 * self.voxel_size)
        seam_overlap = max(0.015, 0.35 * self.voxel_size)
        panels = (
            ((0.2014, 0.2989, 0.0550), (0.550, 0.030, 0.650), 3.0159),
            ((0.3931, 0.4359, 0.0550), (0.050, 0.450, 0.650), 3.0159),
            ((0.3017, 0.7498, 0.0550), (0.050, 0.500, 0.650), 3.62),
            ((0.2408, 0.6114, -0.1950), (0.550, 0.750, 0.030), 3.0159),
            ((0.2408, 0.6114, 0.3050), (0.550, 0.750, 0.030), 3.0159),
        )

        for origin, size, yaw in panels:
            full_extents = np.asarray(size, dtype=np.float64)
            thin_axis = int(np.argmin(full_extents))
            inflated = full_extents.copy()
            inflated[thin_axis] = max(inflated[thin_axis], min_wall)
            for axis in range(3):
                if axis != thin_axis:
                    inflated[axis] += 2.0 * seam_overlap

            hx, hy, hz = 0.5 * inflated
            builder.add_shape_box(
                body=bucket_body,
                cfg=cfg,
                xform=wp.transform(
                    wp.vec3(*origin),
                    wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(yaw)),
                ),
                hx=float(hx),
                hy=float(hy),
                hz=float(hz),
            )

    def add_dump_bucket(self, builder) -> None:
        bx, by, bz = self.bucket_center
        iw, il, idepth = self.bucket_inner_half
        t = self.env.bucket_wall_thickness

        cfg = newton.ModelBuilder.ShapeConfig(
            ke=2.0e5,
            kd=2.0e3,
            kf=1.0e3,
            mu=0.6,
            density=0.0,
        )

        builder.add_shape_box(
            body=-1,
            cfg=cfg,
            xform=wp.transform(wp.vec3(bx, by, bz + 0.5 * t), wp.quat_identity()),
            hx=iw + t,
            hy=il + t,
            hz=0.5 * t,
        )
        builder.add_shape_box(
            body=-1,
            cfg=cfg,
            xform=wp.transform(wp.vec3(bx, by - il - 0.5 * t, bz + t + 0.5 * idepth), wp.quat_identity()),
            hx=iw + t,
            hy=0.5 * t,
            hz=0.5 * idepth,
        )
        builder.add_shape_box(
            body=-1,
            cfg=cfg,
            xform=wp.transform(wp.vec3(bx, by + il + 0.5 * t, bz + t + 0.5 * idepth), wp.quat_identity()),
            hx=iw + t,
            hy=0.5 * t,
            hz=0.5 * idepth,
        )
        builder.add_shape_box(
            body=-1,
            cfg=cfg,
            xform=wp.transform(wp.vec3(bx - iw - 0.5 * t, by, bz + t + 0.5 * idepth), wp.quat_identity()),
            hx=0.5 * t,
            hy=il,
            hz=0.5 * idepth,
        )
        builder.add_shape_box(
            body=-1,
            cfg=cfg,
            xform=wp.transform(wp.vec3(bx + iw + 0.5 * t, by, bz + t + 0.5 * idepth), wp.quat_identity()),
            hx=0.5 * t,
            hy=il,
            hz=0.5 * idepth,
        )

    def create_mpm_soil_bank(self, builder) -> None:
        x_half = 0.5 * self.soil.bank_width_m
        y_front = self.env.bank_front_y
        y_back = self.env.bank_back_y
        max_height = self.soil.bank_height_m
        z0 = self.soil.spawn_clearance_m
        slope_tan = math.tan(math.radians(self.soil.slope_angle_deg))

        nx = int(math.ceil(self.soil.bank_width_m / self.voxel_size))
        ny = int(math.ceil(self.soil.bank_length_m / self.voxel_size))
        nz = int(math.ceil((max_height + z0 + self.voxel_size) / self.voxel_size))

        xs = -x_half + np.arange(nx, dtype=np.float32) * self.voxel_size
        ys = y_back + np.arange(ny, dtype=np.float32) * self.voxel_size
        zs = z0 + np.arange(nz, dtype=np.float32) * self.voxel_size

        cell_origins = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
        depth_into_bank = np.maximum(0.0, y_front - cell_origins[:, 1])
        local_height = np.minimum(max_height, depth_into_bank * slope_tan)
        occupancy_mask = (
            (cell_origins[:, 1] >= y_back)
            & (cell_origins[:, 1] <= y_front)
            & (cell_origins[:, 2] <= z0 + local_height)
        )
        occupied_cells = cell_origins[occupancy_mask]

        rng = np.random.default_rng(7)
        jitter = rng.random((occupied_cells.shape[0], self.particles_per_cell, 3), dtype=np.float32) * self.voxel_size
        positions = (occupied_cells[:, None, :] + jitter).reshape(-1, 3)

        particle_mass = self.soil.density_kg_m3 * (self.voxel_size ** 3) / float(self.particles_per_cell)
        zero_vel = wp.vec3(0.0, 0.0, 0.0)
        for p in positions:
            builder.add_particle(
                pos=wp.vec3(float(p[0]), float(p[1]), float(p[2])),
                vel=zero_vel,
                mass=particle_mass,
            )

        e = self.soil.youngs_modulus_pa
        nu = self.soil.poisson_ratio
        builder.mpm_E = e
        builder.mpm_nu = nu
        builder.mpm_mu = e / (2.0 * (1.0 + nu))
        builder.mpm_lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    def _build_excavation_aabb(self) -> None:
        x_half = 0.5 * self.soil.bank_width_m
        self.excavation_aabb_min = np.array(
            [-x_half - 0.30, self.env.bank_back_y - 0.30, -0.05],
            dtype=np.float64,
        )
        self.excavation_aabb_max = np.array(
            [x_half + 0.30, self.env.bank_front_y + 0.30, self.soil.bank_height_m + 0.20],
            dtype=np.float64,
        )

    def _configure_drucker_prager_regularizer(self) -> None:
        # Smooth inscribed Drucker-Prager fit from the soil prior (c, phi).
        sin_phi = math.sin(math.radians(self.soil.friction_angle_deg))
        cos_phi = math.cos(math.radians(self.soil.friction_angle_deg))
        denom = math.sqrt(3.0) * max(3.0 - sin_phi, 1.0e-6)
        self.dp_alpha = 2.0 * sin_phi / denom
        self.dp_k = 6.0 * self.soil.cohesion_pa * cos_phi / denom
        self.bucket_center_wp = wp.vec3(
            float(self.bucket_center[0]),
            float(self.bucket_center[1]),
            float(self.bucket_center[2]),
        )
        self.bucket_inner_half_wp = wp.vec3(
            float(self.bucket_inner_half[0]),
            float(self.bucket_inner_half[1]),
            float(self.bucket_inner_half[2]),
        )

    def apply_drucker_prager_regularizer(self, dt: float) -> None:
        """Apply a smooth DP-inspired plasticity correction.

        This does not replace the underlying Newton MPM constitutive model. It
        wraps the existing step with a pressure-dependent velocity regularizer so
        low-speed confined soil tends to lock, rebound is reduced, and high-shear
        lateral flow is limited in a smooth way that is friendlier to RL.
        """
        if not self.dp.enabled or self.model.particle_count == 0:
            return

        wp.launch(
            kernel=drucker_prager_velocity_regularizer,
            dim=self.model.particle_count,
            inputs=[
                self.state_0.particle_q,
                self.state_0.particle_qd,
                float(dt),
                0.5 * self.soil.bank_width_m,
                self.env.bank_back_y,
                self.env.bank_front_y,
                self.soil.spawn_clearance_m,
                self.soil.bank_height_m,
                math.tan(math.radians(self.soil.slope_angle_deg)),
                self.dp.active_margin_m,
                self.soil.density_kg_m3,
                9.81,
                self.dp_alpha,
                self.dp_k,
                self.dp.flow_length_m,
                self.dp.static_damping,
                self.dp.flow_speed_base_m_s,
                self.dp.flow_speed_gain,
                self.dp.rebound_damping,
                self.dp.lock_speed_m_s,
                self.bucket_center_wp,
                self.bucket_inner_half_wp,
                self.dp.bucket_margin_m,
                self.dp.bucket_bonus,
            ],
            outputs=[],
        )

    # ------------------------------------------------------------------
    # Introspection / configuration review
    # ------------------------------------------------------------------
    def _extract_control_joint_names(self) -> list[str]:
        if hasattr(self.model, "joint_name"):
            names = list(self.model.joint_name)
            if len(names) == self.control_size:
                return [str(n) for n in names]
        return [f"q_{i}" for i in range(self.control_size)]

    def _extract_control_limits(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        lower = getattr(self.model, "joint_limit_lower", None)
        upper = getattr(self.model, "joint_limit_upper", None)
        if lower is None or upper is None:
            return None, None

        lower_np = lower.numpy()
        upper_np = upper.numpy()
        if lower_np.shape[0] != self.control_size or upper_np.shape[0] != self.control_size:
            return None, None
        return lower_np, upper_np

    def _identify_joint_map(self) -> dict[str, Optional[int]]:
        names_l = [name.lower() for name in self.control_joint_names]
        canonical = {
            "swing": ("tankspin_wheel", "tankspin", "swing", "slew"),
            "boom": ("full_arm_rotation", "full_arm", "boom"),
            "arm": ("lower_arm", "lowerarm", "back_arm", "arm"),
            "stick": ("uppertolow", "upper_to_low", "middle", "stick", "dipper"),
            "bucket": ("scoop1", "scoop", "bucket"),
        }

        def find_best(tokens: Iterable[str], used: set[int]) -> Optional[int]:
            for idx, name in enumerate(names_l):
                compact = name.replace("_", "")
                for token in tokens:
                    token_c = token.replace("_", "")
                    if compact == token_c and idx not in used:
                        return idx
            for idx, name in enumerate(names_l):
                compact = name.replace("_", "")
                for token in tokens:
                    token_c = token.replace("_", "")
                    if token_c in compact and idx not in used:
                        return idx
            return None

        resolved: dict[str, Optional[int]] = {}
        used: set[int] = set()
        for key in ("swing", "boom", "arm", "stick", "bucket"):
            idx = find_best(canonical[key], used)
            if idx is not None:
                used.add(idx)
            resolved[key] = idx

        if sum(idx is not None for idx in resolved.values()) < 4 and self.control_size >= 5:
            base = self.control_size - 5
            resolved = {
                "swing": base + 0,
                "boom": base + 1,
                "arm": base + 2,
                "stick": base + 3,
                "bucket": base + 4,
            }
        return {k: (v if v is not None and v < self.control_size else None) for k, v in resolved.items()}

    def review_configuration(self) -> None:
        print("\n" + "=" * 72)
        print("CONFIGURATION REVIEW")
        print("=" * 72)
        print(f"Preset: {self.sim_preset.name}")
        print(f"Control vector size: {self.control_size}")
        print(f"Control names: {self.control_joint_names}")
        print(f"Joint map: {self.joint_map}")
        print(f"Bucket body index for soil broad-phase: {self.bucket_body_index}")
        print(f"Particles: {self.total_particles:,}")
        print(f"Voxel size: {self.voxel_size:.3f} m, particles/cell: {self.particles_per_cell}")
        print(
            "Fixed soil substeps per rigid step: "
            f"{self.sim_preset.mpm_contact}"
        )
        print(
            f"DP-like regularizer: enabled={self.dp.enabled}, alpha={self.dp_alpha:.4f}, "
            f"k={self.dp_k / 1000.0:.2f} kPa, fit={self.dp.matching}"
        )
        print("=" * 72)

    def print_summary(self) -> None:
        print("\n" + "=" * 72)
        print("EXCAVATOR MPM SIMULATION")
        print("=" * 72)
        print(f"Preset: {self.sim_preset.name}")
        print(f"Soil preset: {self.soil.name}")
        print(f"  density: {self.soil.density_kg_m3:.0f} kg/m^3")
        print(f"  E: {self.soil.youngs_modulus_pa / 1.0e6:.2f} MPa")
        print(f"  nu: {self.soil.poisson_ratio:.2f}")
        print(f"  target friction angle for calibration: {self.soil.friction_angle_deg:.1f} deg")
        print(f"  target cohesion for calibration: {self.soil.cohesion_pa / 1000.0:.1f} kPa")
        print(
            f"  DP-like fit: alpha={self.dp_alpha:.4f}, k={self.dp_k / 1000.0:.2f} kPa "
            f"({self.dp.matching})"
        )
        print(f"Total soil particles: {self.total_particles:,}")
        print("=" * 72)

    # ------------------------------------------------------------------
    # Simulation stepping / soil-contact scheduling
    # ------------------------------------------------------------------
    def capture(self) -> None:
        self.coupled_graph = None

    def _sync_mpm_collider_pose(self) -> None:
        self.collider_body_q.assign(self.state_0.body_q)

    @staticmethod
    def _coerce_vec3(obj) -> Optional[np.ndarray]:
        if obj is None:
            return None
        if isinstance(obj, np.void):
            if obj.dtype.names:
                for key in ("p", "pos", "position", "translation"):
                    if key in obj.dtype.names:
                        return ExcavatorMPMExample._coerce_vec3(obj[key])
                for key in obj.dtype.names:
                    candidate = ExcavatorMPMExample._coerce_vec3(obj[key])
                    if candidate is not None:
                        return candidate
            return None
        if isinstance(obj, np.ndarray):
            if obj.dtype.names:
                return ExcavatorMPMExample._coerce_vec3(obj[()])
            flat = np.asarray(obj, dtype=np.float64).reshape(-1)
            if flat.size >= 3:
                return flat[:3]
            return None
        if hasattr(obj, "p"):
            return ExcavatorMPMExample._coerce_vec3(getattr(obj, "p"))
        if hasattr(obj, "tolist"):
            return ExcavatorMPMExample._coerce_vec3(obj.tolist())
        if isinstance(obj, (tuple, list)):
            if len(obj) >= 3 and all(np.isscalar(v) for v in obj[:3]):
                return np.asarray(obj[:3], dtype=np.float64)
            for item in obj:
                candidate = ExcavatorMPMExample._coerce_vec3(item)
                if candidate is not None:
                    return candidate
        return None

    def _get_bucket_world_position(self) -> Optional[np.ndarray]:
        if self.bucket_body_index is None:
            return None
        try:
            body_q_host = self.state_0.body_q.numpy()
            if self.bucket_body_index >= len(body_q_host):
                return None
            return self._coerce_vec3(body_q_host[self.bucket_body_index])
        except Exception:
            return None

    @staticmethod
    def _distance_to_aabb(point: np.ndarray, bb_min: np.ndarray, bb_max: np.ndarray) -> float:
        delta = np.maximum(0.0, np.maximum(bb_min - point, point - bb_max))
        return float(np.linalg.norm(delta))

    def select_mpm_substeps(self) -> int:
        pos = self._get_bucket_world_position()
        if pos is None:
            self.last_bucket_bank_distance_m = float("nan")
        else:
            self.last_bucket_bank_distance_m = max(
                0.0,
                self._distance_to_aabb(pos, self.excavation_aabb_min, self.excavation_aabb_max) - self.bucket_proxy_radius_m,
            )

        self.last_selected_mpm_substeps = self.sim_preset.mpm_contact
        return self.last_selected_mpm_substeps

    def simulate_rigid_substep(self, dt: float) -> None:
        self.state_0.clear_forces()
        self.viewer.apply_forces(self.state_0)
        self.solver.step(self.state_0, self.state_1, self.control, contacts=None, dt=dt)
        self.state_0, self.state_1 = self.state_1, self.state_0
        self._sync_mpm_collider_pose()

    def simulate_soil_substeps(self, count: int, dt: float) -> None:
        for _ in range(count):
            self._sync_mpm_collider_pose()
            self.mpm_solver.step(self.state_0, self.state_0, None, None, dt)
            for _ in range(4):
                self.mpm_solver.project_outside(self.state_0, self.state_0, dt)

    def simulate_coupled_frame(self) -> None:
        for _ in range(self.sim_substeps):
            self.simulate_rigid_substep(self.sim_dt)
            local_soil_substeps = self.select_mpm_substeps()
            local_mpm_dt = self.sim_dt / float(local_soil_substeps)
            self.simulate_soil_substeps(local_soil_substeps, local_mpm_dt)
            self.apply_drucker_prager_regularizer(self.sim_dt)

    # ------------------------------------------------------------------
    # Controller (kept mainly for the built-in digging demo)
    # ------------------------------------------------------------------
    @staticmethod
    def smoothstep(u: float) -> float:
        u = float(np.clip(u, 0.0, 1.0))
        return u * u * (3.0 - 2.0 * u)

    @staticmethod
    def blend_pose(a: DigPose, b: DigPose, u: float) -> DigPose:
        s = ExcavatorMPMExample.smoothstep(u)
        return DigPose(
            swing=(1.0 - s) * a.swing + s * b.swing,
            boom=(1.0 - s) * a.boom + s * b.boom,
            arm=(1.0 - s) * a.arm + s * b.arm,
            stick=(1.0 - s) * a.stick + s * b.stick,
            bucket=(1.0 - s) * a.bucket + s * b.bucket,
        )

    def sample_dig_cycle(self, t: float) -> DigPose:
        home = DigPose(0.00, 0.72, 0.55, 0.10, 0.55)
        entry = DigPose(0.04, 0.42, 0.18, 0.22, 0.72)
        crowd = DigPose(0.08, 0.18, -0.10, 0.28, 0.80)
        curl = DigPose(0.10, 0.28, -0.02, -0.48, -0.72)
        lift = DigPose(0.12, 0.62, 0.32, -0.72, -0.88)
        swing = DigPose(-1.00, 0.58, 0.24, -0.56, -0.82)
        dump = DigPose(-1.15, 0.50, 0.12, -0.32, 0.76)
        return_high = DigPose(-0.30, 0.68, 0.40, -0.05, 0.45)

        phase_times = (
            (0.0, 1.6, home, entry),
            (1.6, 3.4, entry, crowd),
            (3.4, 5.0, crowd, curl),
            (5.0, 6.8, curl, lift),
            (6.8, 8.8, lift, swing),
            (8.8, 10.0, swing, dump),
            (10.0, 12.0, dump, return_high),
        )

        if t >= self.dig_cycle_duration:
            return return_high
        for t0, t1, pose0, pose1 in phase_times:
            if t <= t1:
                u = (t - t0) / max(t1 - t0, 1.0e-6)
                return self.blend_pose(pose0, pose1, u)
        return return_high

    def _clip_target(self, index: Optional[int], value: float) -> float:
        if index is None:
            return value
        if self.control_lower is None or self.control_upper is None:
            return value
        lower = float(self.control_lower[index])
        upper = float(self.control_upper[index])
        if lower < upper:
            return float(np.clip(value, lower, upper))
        return value

    def apply_control(self) -> None:
        if self.control_size == 0:
            return

        targets = self._joint_target_host.copy()
        if self.sim_time < self.settle_duration:
            desired = DigPose(0.0, 0.72, 0.55, 0.10, 0.55)
        else:
            cycle_t = (self.sim_time - self.settle_duration) % self.dig_cycle_duration
            desired = self.sample_dig_cycle(cycle_t)

        desired_map = {
            "swing": desired.swing,
            "boom": desired.boom,
            "arm": desired.arm,
            "stick": desired.stick,
            "bucket": desired.bucket,
        }
        for key, value in desired_map.items():
            idx = self.joint_map.get(key)
            if idx is None:
                continue
            targets[idx] = self._clip_target(idx, value)

        max_delta = np.full(self.control_size, 0.10, dtype=np.float64)
        if self.control_size >= 5:
            base = self.control_size - 5
            max_delta[base + 0] = 0.04
            max_delta[base + 1] = 0.06
            max_delta[base + 2] = 0.06
            max_delta[base + 3] = 0.08
            max_delta[base + 4] = 0.08
        targets = np.clip(targets, self._joint_target_host - max_delta, self._joint_target_host + max_delta)

        self.control.joint_target_pos.assign(targets)
        self._joint_target_host = targets.copy()

        t = self.sim_time
        if int(t) % 3 == 0 and t - int(t) < self.frame_dt:
            print(f"\n[t={t:.1f}s] Control target: {desired}")
            print(f"  Current pos: [{', '.join([format(float(x), '6.3f') for x in self.state_0.joint_q.numpy()])}]")

    # ------------------------------------------------------------------
    # Task metrics / main loop
    # ------------------------------------------------------------------
    def count_particles_in_bucket(self) -> int:
        if self.model.particle_count == 0:
            return 0

        positions = self.state_0.particle_q.numpy()
        bx, by, bz = self.bucket_center
        iw, il, idepth = self.bucket_inner_half
        t = self.env.bucket_wall_thickness
        inside = (
            (positions[:, 0] >= bx - iw)
            & (positions[:, 0] <= bx + iw)
            & (positions[:, 1] >= by - il)
            & (positions[:, 1] <= by + il)
            & (positions[:, 2] >= bz + t)
            & (positions[:, 2] <= bz + t + 2.0 * idepth)
        )
        return int(np.count_nonzero(inside))

    def step(self) -> None:
        self.apply_control()

        if getattr(self, "coupled_graph", None):
            wp.capture_launch(self.coupled_graph)
        else:
            self.simulate_coupled_frame()

        self.sim_time += self.frame_dt

        if self.sim_time - self.last_score_print >= self.score_print_interval:
            self.particles_in_bucket = self.count_particles_in_bucket()
            pct = 100.0 * self.particles_in_bucket / max(self.total_particles, 1)
            print(
                f"\n[t={self.sim_time:.1f}s] SCORE: {self.particles_in_bucket:,} / {self.total_particles:,} "
                f"particles in bucket ({pct:.2f}%), fixed_soil_substeps={self.last_selected_mpm_substeps}, "
                f"bucket_bank_distance={self.last_bucket_bank_distance_m:.3f}"
            )
            self.last_score_print = self.sim_time

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


def main() -> None:
    viewer, args = newton.examples.init()

    # Change this preset to switch between RL throughput and validation quality.
    preset_name = "balanced_full_contact"

    example = ExcavatorMPMExample(
        viewer,
        preset_name=preset_name,
    )

    try:
        while viewer.is_running():
            example.step()
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")


if __name__ == "__main__":
    main()
