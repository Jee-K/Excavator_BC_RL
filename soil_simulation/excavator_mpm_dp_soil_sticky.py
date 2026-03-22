#!/usr/bin/env python3
"""
Excavator + MPM soil simulation tuned for a practical RL / sim-to-real compromise.
"""
# these aren't necessary, but they make it easier to read
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Iterable, Optional

# necessaries
import numpy as np

# main simulation tools
import warp as wp
import newton
import newton.examples
from newton.solvers import SolverImplicitMPM


@dataclass(frozen=True)
class SoilPreset:
    # material information
    name: str = "moist_sandy_loam"
    density_kg_m3: float = 1650.0
    youngs_modulus_pa: float = 8.0e6
    poisson_ratio: float = 0.28
    friction_angle_deg: float = 35.0
    cohesion_pa: float = 15_000.0
    dilatancy_deg: float = 3.0
    interface_friction_mu: float = 0.55

    # spawn information
    slope_angle_deg: float = 32.0
    bank_height_m: float = 0.85
    bank_width_m: float = 2.6
    bank_length_m: float = 2.2
    spawn_clearance_m: float = 0.025


@dataclass(frozen=True)
class EnvironmentPreset:
    # excavator information, rotation largely excluded
    excavator_position: tuple[float, float, float] = (0.0, 3.0, 0.6)

    # this looks slightly better than the built-in ground in the rendered, even though they act the same
    ground_size: tuple[float, float] = (20.0, 20.0)

    # goal information
    bucket_center: tuple[float, float, float] = (-4.0, 3.0, 0.0)
    bucket_inner_half: tuple[float, float, float] = (0.75, 1.5, 0.8)
    bucket_wall_thickness: float = 0.1

    # spawn information !!!
    bank_front_y: float = 1.10
    bank_back_y: float = -1.10


@dataclass(frozen=True)
class DigPose:
    swing: float
    arm: float
    stick: float
    bucket: float


# particles per cell is not a reasonable construction, just use voxel size
@dataclass(frozen=True)
class SimulationPreset:
    name: str
    voxel_size_m: float
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
        rigid_substeps=6,
        mpm_far=1,
        mpm_near=3,
        mpm_contact=6,
        mpm_iterations=40,
        rigid_ls_iterations=60,
        rigid_njmax=260,
    ),
}


@dataclass(frozen=True)
class DPRegularizerConfig:
    enabled: bool = True
    active_margin_m: float = 0.35
    bucket_margin_m: float = 0.25
    flow_length_m: float = 0.06
    static_damping: float = 12.0
    flow_speed_base_m_s: float = 0.18
    flow_speed_gain: float = 0.22
    rebound_damping: float = 8.0
    lock_speed_m_s: float = 0.03
    bucket_bonus: float = 0.75


@dataclass(frozen=True)
class BucketSealConfig:
    min_wall_m: float = 0.16
    min_wall_voxels: float = 4.0
    seam_overlap_m: float = 0.04
    seam_overlap_voxels: float = 1.0
    front_panel_extra_m: float = 0.03
    contact_hold_s: float = 0.60
    loaded_hold_s: float = 1.20
    loaded_particle_threshold: int = 12
    loaded_radius_m: float = 1.10
    projection_passes: int = 2


@wp.kernel
def drucker_prager_velocity_regularizer(
    particle_q: wp.array(dtype=wp.vec3), # type: ignore
    particle_qd: wp.array(dtype=wp.vec3), # type: ignore
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
): # !!!
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




@wp.kernel
def compute_body_forces_from_soil(
    dt: float,
    collider_ids: wp.array(dtype=int),
    collider_impulses: wp.array(dtype=wp.vec3),
    collider_impulse_pos: wp.array(dtype=wp.vec3),
    body_ids: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    body_f: wp.array(dtype=wp.spatial_vector),
):
    """Map MPM collider impulses into per-body spatial forces."""
    i = wp.tid()
    cid = collider_ids[i]
    if cid >= 0 and cid < body_ids.shape[0]:
        body_index = body_ids[cid]
        if body_index == -1:
            return
        f_world = collider_impulses[i] / dt
        x_wb = body_q[body_index]
        x_com = body_com[body_index]
        r = collider_impulse_pos[i] - wp.transform_point(x_wb, x_com)
        wp.atomic_add(body_f, body_index, wp.spatial_vector(f_world, wp.cross(r, f_world)))


@wp.kernel
def subtract_body_force_from_velocity(
    dt: float,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    body_inv_mass: wp.array(dtype=float),
    body_q_res: wp.array(dtype=wp.transform),
    body_qd_res: wp.array(dtype=wp.spatial_vector),
):
    """Remove previously applied soil-force contribution before the next MPM solve."""
    body_id = wp.tid()
    f = body_f[body_id]
    delta_v = dt * body_inv_mass[body_id] * wp.spatial_top(f)
    r = wp.transform_get_rotation(body_q[body_id])
    delta_w = dt * wp.quat_rotate(r, body_inv_inertia[body_id] * wp.quat_rotate_inv(r, wp.spatial_bottom(f)))
    body_q_res[body_id] = body_q[body_id]
    body_qd_res[body_id] = body_qd[body_id] - wp.spatial_vector(delta_v, delta_w)


@wp.kernel
def add_spatial_force_inplace(
    dst: wp.array(dtype=wp.spatial_vector),
    src: wp.array(dtype=wp.spatial_vector),
):
    i = wp.tid()
    dst[i] = dst[i] + src[i]


class ExcavatorMPMExample:
    def __init__(
        self,
        viewer,
        preset_name: str = "balanced",
        voxel_size: Optional[float] = None,
    ):
        if preset_name not in SIM_PRESETS:
            raise ValueError(f"Unknown preset '{preset_name}'. Available: {sorted(SIM_PRESETS)}")

        self.viewer = viewer
        self.device = wp.get_device()
        self.soil = SoilPreset()
        self.env = EnvironmentPreset()
        self.dp = DPRegularizerConfig()
        self.bucket_seal = BucketSealConfig()
        self.sim_preset = SIM_PRESETS[preset_name]

        self.voxel_size = float(voxel_size if voxel_size is not None else self.sim_preset.voxel_size_m)

        self.fps = 30
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = self.sim_preset.rigid_substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.score_print_interval = 5.0
        self.last_score_print = 0.0
        self.settle_duration = 1.5
        self.dig_cycle_duration = 12.0

        # ------------------------------------------------------------------
        # Contact scheduling thresholds (planner / adaptive stepping only).
        # These are *not* physical collision margins.
        # ------------------------------------------------------------------
        self.bucket_proxy_radius_m = 0.45
        self.bucket_contact_trigger_m = max(0.03, 1.0 * self.voxel_size)
        self.bucket_near_trigger_m = max(0.12, 3.0 * self.voxel_size)
        self.contact_hold_counter = 0
        self.loaded_hold_counter = 0
        self.last_selected_mpm_substeps = self.sim_preset.mpm_contact
        self.last_bucket_bank_distance_m = float("nan")
        self.last_particles_near_bucket = 0
        self.bucket_loaded_recently = False

        # ------------------------------------------------------------------
        # Physical collider tuning.
        # Keep margin small because it shifts the effective surface; use a
        # slightly larger gap to enable earlier detection without bloating the
        # bucket too much. Defaults are intentionally conservative.
        # ------------------------------------------------------------------
        self.shape_margin_m = min(max(0.08 * self.voxel_size, 0.0015), 0.0060)
        self.shape_gap_m = min(max(0.50 * self.voxel_size, 0.0080), 0.0250)
        self.soft_contact_margin_m = self.shape_gap_m

        self.bucket_proxy_body_name: Optional[str] = None
        self.bucket_proxy_body_index_builder: Optional[int] = None
        self.bucket_body_index: Optional[int] = None

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)

        # Apply rigid defaults before URDF import. In this Newton build,
        # margin/gap live on SolverMuJoCo rather than on ShapeConfig, so keep
        # the builder-side defaults focused on friction/contact stiffness and
        # particle-collision flags only.
        self.configure_rigid_defaults(builder)
        if hasattr(builder, "default_shape_cfg"):
            if hasattr(builder.default_shape_cfg, "has_particle_collision"):
                builder.default_shape_cfg.has_particle_collision = True
            if hasattr(builder.default_shape_cfg, "is_solid"):
                builder.default_shape_cfg.is_solid = True

        # Reusable explicit configs for static/proxy geoms added outside the URDF.
        self.static_particle_contact_cfg = newton.ModelBuilder.ShapeConfig(
            ke=1.0e6,
            kd=1.0e4,
            kf=1.0e3,
            mu=float(self.soil.interface_friction_mu),
            density=0.0,
            has_particle_collision=True,
            is_solid=True,
        )

        self.static_ground_cfg = newton.ModelBuilder.ShapeConfig(
            mu=0.60,
            density=0.0,
            has_particle_collision=True,
            is_solid=True,
        )

        SolverImplicitMPM.register_custom_attributes(builder)

        excavator_urdf = "./excavatorURDF/excavator_lowpoly_locked.urdf"
        print(f"Loading excavator from: {excavator_urdf}")

        body_count_before_urdf = getattr(builder, "body_count", 0)
        control_start = len(builder.joint_target_ke)
        builder.add_urdf(
            excavator_urdf,
            xform=wp.transform(wp.vec3(*self.env.excavator_position), wp.quat_identity()),
            floating=True,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            ignore_inertial_definitions=False,
        )
        control_end = len(builder.joint_target_ke)
        body_count_after_urdf = getattr(builder, "body_count", body_count_before_urdf)
        self._prune_excavator_particle_collisions(builder, body_count_before_urdf, body_count_after_urdf)

        for i in range(control_start, control_end):
            builder.joint_target_ke[i] = 5000.0
            builder.joint_target_kd[i] = 500.0
        if control_end > 8:
            builder.joint_target_ke[7] = 15000.0

        print("Creating sloped MPM excavation bank...")
        self.create_mpm_soil_bank(builder)
        self.add_ground_plane(builder)

        self.bucket_center = np.asarray(self.env.bucket_center, dtype=np.float64)
        self.bucket_inner_half = np.asarray(self.env.bucket_inner_half, dtype=np.float64)
        self.add_dump_bucket(builder)

        self.model = builder.finalize()

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
        # Match Newton's articulated-body MPM examples: estimate collider
        # boundary velocity from pose deltas for more reliable moving-tool
        # particle contact.
        try:
            mpm_options.collider_velocity_mode = "finite_difference"
        except Exception:
            pass

        mujoco_solver_kwargs = dict(
            ls_iterations=self.sim_preset.rigid_ls_iterations,
            njmax=self.sim_preset.rigid_njmax,
            margin=self.shape_margin_m,
            gap=self.shape_gap_m,
        )
        try:
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                **mujoco_solver_kwargs,
            )
            self._solver_accepts_margin_gap = True
        except TypeError: # !!! FIXME
            mujoco_solver_kwargs.pop("margin", None)
            mujoco_solver_kwargs.pop("gap", None)
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                **mujoco_solver_kwargs,
            )
            self._solver_accepts_margin_gap = False
            print("Warning: SolverMuJoCo margin/gap kwargs unsupported in this build; using solver defaults.")
        self.mpm_solver = SolverImplicitMPM(self.model, mpm_options)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.mpm_state = self.model.state()
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self._copy_state_to_mpm_state()

        # Dedicated collider pose/velocity buffers for the MPM contact path.
        self.collider_body_q = wp.zeros_like(self.state_0.body_q)
        self.collider_body_qd = wp.zeros_like(self.state_0.body_qd)
        self.collider_body_q.assign(self.state_0.body_q)
        self.collider_body_qd.assign(self.state_0.body_qd)
        self._collider_accepts_body_qd = False
        try:
            self.mpm_solver.setup_collider(
                body_mass=wp.zeros_like(self.model.body_mass),
                body_q=self.collider_body_q,
                body_qd=self.collider_body_qd,
            )
            self._collider_accepts_body_qd = True
        except TypeError:
            self.mpm_solver.setup_collider(
                body_mass=wp.zeros_like(self.model.body_mass),
                body_q=self.collider_body_q,
            )

        # Two-way coupling buffers (adapted from Newton's mpm_twoway_coupling example).
        max_nodes = 1 << 20
        self.collider_impulses = wp.zeros(max_nodes, dtype=wp.vec3, device=self.model.device)
        self.collider_impulse_pos = wp.zeros(max_nodes, dtype=wp.vec3, device=self.model.device)
        self.collider_impulse_ids = wp.full(max_nodes, value=-1, dtype=int, device=self.model.device)
        self.collider_body_id = getattr(self.mpm_solver, "collider_body_index", None)
        self.body_f_from_soil = wp.zeros_like(self.state_0.body_f)
        self.body_f_from_soil_prev = wp.zeros_like(self.state_0.body_f)
        self._zero_body_force = wp.zeros_like(self.state_0.body_f)
        self._twoway_coupling_enabled = self.collider_body_id is not None and hasattr(self.mpm_solver, "_collect_collider_impulses")
        if not self._twoway_coupling_enabled:
            print("Warning: MPM collider impulse API not available; soil->rigid feedback disabled.")
        self._collect_collider_impulses(self.mpm_state)

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

    def _fallback_bucket_proxy_panels(self) -> tuple[tuple[tuple[float, float, float], tuple[float, float, float], float], ...]:
        # Match the simplified box collisions in robot_fixed_cleaned.urdf.
        return (
            ((0.2014, 0.2989, 0.0550), (0.550, 0.140, 0.650), 3.0159),
            ((0.3931, 0.4359, 0.0550), (0.170, 0.700, 0.650), 3.0159),
            ((0.3017, 0.7498, 0.0550), (0.170, 0.600, 0.650), 3.82),
            ((0.2408, 0.6114, -0.1950), (0.550, 0.750, 0.140), 3.0159),
            ((0.2408, 0.6114, 0.3050), (0.550, 0.750, 0.140), 3.0159),
        )

    def _load_bucket_proxy_panels_from_urdf(
        self, urdf_path: str
    ) -> tuple[tuple[tuple[float, float, float], tuple[float, float, float], float], ...]:
        try:
            root = ET.parse(urdf_path).getroot()
        except Exception:
            return self._fallback_bucket_proxy_panels()

        bucket_link = None
        for link in root.findall("link"):
            if link.get("name") == "part04":
                bucket_link = link
                break
        if bucket_link is None:
            return self._fallback_bucket_proxy_panels()

        panels: list[tuple[tuple[float, float, float], tuple[float, float, float], float]] = []
        for collision in bucket_link.findall("collision"):
            geometry = collision.find("geometry")
            box = None if geometry is None else geometry.find("box")
            if box is None:
                continue

            origin = collision.find("origin")
            xyz = [0.0, 0.0, 0.0]
            rpy = [0.0, 0.0, 0.0]
            if origin is not None:
                if origin.get("xyz"):
                    xyz = [float(v) for v in origin.get("xyz").split()]
                if origin.get("rpy"):
                    rpy = [float(v) for v in origin.get("rpy").split()]

            size = [float(v) for v in box.get("size").split()]
            panels.append((tuple(xyz), tuple(size), float(rpy[2])))

        if not panels:
            return self._fallback_bucket_proxy_panels()
        return tuple(panels)

    def _prune_excavator_particle_collisions(self, builder, body_start: int, body_end: int) -> None:
        """Keep particle collision only on bucket-relevant excavator links.

        Mirrors the official ANYmal example's practice of pruning particle
        contact to the links that matter most.
        """
        if not hasattr(builder, "body_shapes") or not hasattr(builder, "shape_flags"):
            print("Particle-collision pruning unavailable on this builder; keeping imported defaults.")
            return

        body_labels = None
        for attr in ("body_label", "body_name", "body_names"):
            if hasattr(builder, attr):
                body_labels = getattr(builder, attr)
                break
        if body_labels is None:
            print("Particle-collision pruning skipped: no body labels available.")
            return

        keep_tokens = (
            "scoop",
            "bucket",
            "part04",
            "stick",
            "dipper",
            "uppertolow",
            "upper_to_low",
            "middle",
            "seal",
        )
        removed = 0
        for body in range(body_start, body_end):
            label = str(body_labels[body]).lower().replace("_", "")
            keep = any(token.replace("_", "") in label for token in keep_tokens)
            if keep:
                continue
            for shape in builder.body_shapes[body]:
                builder.shape_flags[shape] = builder.shape_flags[shape] & ~newton.ShapeFlags.COLLIDE_PARTICLES
                removed += 1
        print(f"Pruned particle collision from {removed} imported excavator shapes outside bucket/stick links.")

    def add_ground_plane(self, builder) -> None:
        width, length = self.env.ground_size
        builder.add_shape_plane(
            body=-1,
            cfg=self.static_ground_cfg,
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

    def add_dump_bucket(self, builder) -> None:
        bx, by, bz = self.bucket_center
        iw, il, idepth = self.bucket_inner_half
        t = self.env.bucket_wall_thickness

        # cfg = newton.ModelBuilder.ShapeConfig(
        #     ke=2.0e5,
        #     kd=2.0e3,
        #     kf=1.0e3,
        #     mu=0.6,
        #     density=0.0,
        # )
        cfg = self.static_particle_contact_cfg

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
        slope_tan = np.tan(np.radians(self.soil.slope_angle_deg))

        nx = int(np.ceil(self.soil.bank_width_m / self.voxel_size))
        ny = int(np.ceil(self.soil.bank_length_m / self.voxel_size))
        nz = int(np.ceil((max_height + z0 + self.voxel_size) / self.voxel_size))

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
        jitter = rng.random((occupied_cells.shape[0], 1, 3), dtype=np.float32) * self.voxel_size
        positions = (occupied_cells[:, None, :] + jitter).reshape(-1, 3)

        particle_mass = self.soil.density_kg_m3 * (self.voxel_size ** 3)
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
        sin_phi = np.sin(np.radians(self.soil.friction_angle_deg))
        cos_phi = np.cos(np.radians(self.soil.friction_angle_deg))
        denom = np.sqrt(3.0) * max(3.0 - sin_phi, 1.0e-6)
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
                np.tan(np.radians(self.soil.slope_angle_deg)),
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
        for key in ("swing", "arm", "stick", "bucket"):
            idx = find_best(canonical[key], used)
            if idx is not None:
                used.add(idx)
            resolved[key] = idx

        if sum(idx is not None for idx in resolved.values()) < 4 and self.control_size >= 5:
            base = self.control_size - 5
            resolved = {
                "swing": base + 0,
                "arm": base + 1,
                "stick": base + 2,
                "bucket": base + 3,
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
        print(f"Two-way coupling enabled: {self._twoway_coupling_enabled}")
        print(f"Particles: {self.total_particles:,}")
        print(f"Voxel size: {self.voxel_size:.3f} m")
        print(
            "Adaptive soil substeps per rigid step: "
            f"far={self.sim_preset.mpm_far}, near={self.sim_preset.mpm_near}, contact={self.sim_preset.mpm_contact}"
        )
        print(
            f"DP-like regularizer: enabled={self.dp.enabled}, alpha={self.dp_alpha:.4f}, "
            f"k={self.dp_k / 1000.0:.2f} kPa"
        )
        print(
            "Bucket sealing: "
            f"min_wall={max(self.bucket_seal.min_wall_m, self.bucket_seal.min_wall_voxels * self.voxel_size):.3f} m, "
            f"seam_overlap={max(self.bucket_seal.seam_overlap_m, self.bucket_seal.seam_overlap_voxels * self.voxel_size):.3f} m, "
            f"projection_passes={self.bucket_seal.projection_passes}"
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
        )
        print(f"Total soil particles: {self.total_particles:,}")
        print("=" * 72)

    # ------------------------------------------------------------------
    # Simulation stepping / soil-contact scheduling
    # ------------------------------------------------------------------
    def capture(self) -> None:
        self.coupled_graph = None

    def _copy_state_to_mpm_state(self) -> None:
        self.mpm_state.body_q.assign(self.state_0.body_q)
        self.mpm_state.body_qd.assign(self.state_0.body_qd)
        self.mpm_state.body_f.assign(self.state_0.body_f)
        self.mpm_state.particle_q.assign(self.state_0.particle_q)
        self.mpm_state.particle_qd.assign(self.state_0.particle_qd)

    def _copy_particles_from_mpm_state(self) -> None:
        self.state_0.particle_q.assign(self.mpm_state.particle_q)
        self.state_0.particle_qd.assign(self.mpm_state.particle_qd)

    def _sync_mpm_collider_pose(self, source_state=None) -> None:
        source_state = self.state_0 if source_state is None else source_state
        self.collider_body_q.assign(source_state.body_q)
        if self._collider_accepts_body_qd:
            self.collider_body_qd.assign(source_state.body_qd)

    def _zero_body_force_buffer(self, buf) -> None:
        buf.assign(self._zero_body_force)

    def _collect_collider_impulses(self, state) -> bool:
        if not self._twoway_coupling_enabled:
            return False
        try:
            collider_impulses, collider_impulse_pos, collider_impulse_ids = self.mpm_solver._collect_collider_impulses(state)
        except Exception:
            return False

        self.collider_impulse_ids.fill_(-1)
        n_colliders = min(collider_impulses.shape[0], self.collider_impulses.shape[0])
        if n_colliders <= 0:
            return False
        self.collider_impulses[:n_colliders].assign(collider_impulses[:n_colliders])
        self.collider_impulse_pos[:n_colliders].assign(collider_impulse_pos[:n_colliders])
        self.collider_impulse_ids[:n_colliders].assign(collider_impulse_ids[:n_colliders])
        return True

    def _compute_soil_reaction_forces(self, dt_divisor: float) -> None:
        if not self._twoway_coupling_enabled:
            return
        wp.launch(
            compute_body_forces_from_soil,
            dim=self.collider_impulse_ids.shape[0],
            inputs=[
                float(max(dt_divisor, 1.0e-6)),
                self.collider_impulse_ids,
                self.collider_impulses,
                self.collider_impulse_pos,
                self.collider_body_id,
                self.state_0.body_q,
                self.model.body_com,
                self.body_f_from_soil,
            ],
        )

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

    def refresh_bucket_load_estimate(self) -> None:
        pos = self._get_bucket_world_position()
        if pos is None or self.model.particle_count == 0:
            self.last_particles_near_bucket = 0
            self.bucket_loaded_recently = False
            return

        try:
            positions = self.state_0.particle_q.numpy()
        except Exception:
            self.last_particles_near_bucket = 0
            self.bucket_loaded_recently = False
            return

        radius = max(self.bucket_seal.loaded_radius_m, self.bucket_proxy_radius_m + 0.45)
        delta = positions - pos[None, :]
        near = np.einsum("ij,ij->i", delta, delta) <= radius * radius
        self.last_particles_near_bucket = int(np.count_nonzero(near))
        self.bucket_loaded_recently = self.last_particles_near_bucket >= self.bucket_seal.loaded_particle_threshold

    @staticmethod
    def _distance_to_aabb(point: np.ndarray, bb_min: np.ndarray, bb_max: np.ndarray) -> float:
        delta = np.maximum(0.0, np.maximum(bb_min - point, point - bb_max))
        return float(np.linalg.norm(delta))

    def select_mpm_substeps(self) -> int:
        pos = self._get_bucket_world_position()
        if pos is None:
            self.last_bucket_bank_distance_m = float("nan")
            self.last_selected_mpm_substeps = self.sim_preset.mpm_contact
            return self.last_selected_mpm_substeps

        distance = max(
            0.0,
            self._distance_to_aabb(pos, self.excavation_aabb_min, self.excavation_aabb_max) - self.bucket_proxy_radius_m,
        )
        self.last_bucket_bank_distance_m = distance

        contact_hold_steps = max(1, int(round(self.bucket_seal.contact_hold_s / max(self.sim_dt, 1.0e-6))))
        loaded_hold_steps = max(1, int(round(self.bucket_seal.loaded_hold_s / max(self.sim_dt, 1.0e-6))))

        if distance <= self.bucket_contact_trigger_m:
            self.contact_hold_counter = max(self.contact_hold_counter, contact_hold_steps)
            self.loaded_hold_counter = max(self.loaded_hold_counter, loaded_hold_steps)
            substeps = self.sim_preset.mpm_contact
        elif distance <= self.bucket_near_trigger_m:
            self.contact_hold_counter = max(self.contact_hold_counter, max(1, contact_hold_steps // 2))
            self.loaded_hold_counter = max(self.loaded_hold_counter, max(1, loaded_hold_steps // 2))
            substeps = self.sim_preset.mpm_near
        elif self.bucket_loaded_recently:
            self.loaded_hold_counter = max(self.loaded_hold_counter, loaded_hold_steps)
            substeps = max(self.sim_preset.mpm_near, self.sim_preset.mpm_contact - 1)
        elif self.contact_hold_counter > 0:
            self.contact_hold_counter -= 1
            substeps = self.sim_preset.mpm_near
        elif self.loaded_hold_counter > 0:
            self.loaded_hold_counter -= 1
            substeps = max(self.sim_preset.mpm_near, self.sim_preset.mpm_contact - 1)
        else:
            substeps = self.sim_preset.mpm_far

        self.last_selected_mpm_substeps = substeps
        return substeps

    def simulate_rigid_substep(self, dt: float) -> None:
        self.state_0.clear_forces()
        self.viewer.apply_forces(self.state_0)
        if self._twoway_coupling_enabled:
            wp.launch(
                add_spatial_force_inplace,
                dim=self.state_0.body_q.shape,
                inputs=[self.state_0.body_f, self.body_f_from_soil_prev],
            )
        self.solver.step(self.state_0, self.state_1, self.control, contacts=None, dt=dt)
        self.state_0, self.state_1 = self.state_1, self.state_0
        self._sync_mpm_collider_pose(self.state_0)

    def simulate_soil_substeps(self, count: int, dt: float) -> None:
        self._copy_state_to_mpm_state()
        if self._twoway_coupling_enabled:
            wp.launch(
                subtract_body_force_from_velocity,
                dim=self.state_0.body_q.shape,
                inputs=[
                    float(self.sim_dt),
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.body_f_from_soil_prev,
                    self.model.body_inv_inertia,
                    self.model.body_inv_mass,
                    self.mpm_state.body_q,
                    self.mpm_state.body_qd,
                ],
            )
        self._zero_body_force_buffer(self.body_f_from_soil)
        dt_divisor = float(max(count * dt, 1.0e-6))
        for _ in range(count):
            self._sync_mpm_collider_pose(self.mpm_state)
            self.mpm_solver.step(self.mpm_state, self.mpm_state, None, None, dt)
            for _ in range(self.bucket_seal.projection_passes):
                self.mpm_solver.project_outside(self.mpm_state, self.mpm_state, dt)
            if self._collect_collider_impulses(self.mpm_state):
                self._compute_soil_reaction_forces(dt_divisor)
        self._copy_particles_from_mpm_state()
        if self._twoway_coupling_enabled:
            self.body_f_from_soil_prev.assign(self.body_f_from_soil)

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
            arm=(1.0 - s) * a.arm + s * b.arm,
            stick=(1.0 - s) * a.stick + s * b.stick,
            bucket=(1.0 - s) * a.bucket + s * b.bucket,
        )

    def sample_dig_cycle(self, t: float) -> DigPose:
        home = DigPose(0.00 + 0.72, 0.55, 0.10, 0.55)
        entry = DigPose(0.04 + 0.42, 0.18, 0.22, 0.72)
        crowd = DigPose(0.08 + 0.18, -0.10, 0.28, 0.80)
        curl = DigPose(0.10 + 0.28, -0.02, -0.48, -0.72)
        lift = DigPose(0.12 + 0.62, 0.32, -0.72, -0.88)
        swing = DigPose(-1.00 + 0.58, 0.24, -0.56, -0.82)
        dump = DigPose(-1.15 + 0.50, 0.12, -0.32, 0.76)
        return_high = DigPose(-0.30 + 0.68, 0.40, -0.05, 0.45)

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
            desired = DigPose(0.72, 0.55, 0.10, 0.55)
        else:
            cycle_t = (self.sim_time - self.settle_duration) % self.dig_cycle_duration
            desired = self.sample_dig_cycle(cycle_t)

        desired_map = {
            "swing": desired.swing,
            "arm": desired.arm,
            "stick": desired.stick,
            "bucket": desired.bucket,
        }
        for key, value in desired_map.items():
            idx = self.joint_map.get(key)
            if idx is None:
                continue
            targets[idx] = self._clip_target(idx, value)

        t = self.sim_time
        targets[6] = 3
        targets[7] = -0.5 - np.sin(t / 5.0)
        targets[8] = -np.sin(t / 3.0)
        targets[9] = 5

        self.control.joint_target_pos.assign(targets)

        if int(t) % 3 == 0 and t - int(t) < self.frame_dt:
            print(f"\n[t={t:.1f}s] Control target: {desired}")
            print(f"  Current pos: [{', '.join([format(float(x), '6.3f') for x in self.state_0.joint_q.numpy()])}]")

        # !!! REMOVE
        if int(t) % 1 == 0 and t - int(t) < self.frame_dt:
            q = np.asarray(self.state_0.joint_q.numpy(), dtype=np.float64)
            qd = None
            if hasattr(self.state_0, "joint_qd"):
                try:
                    qd = np.asarray(self.state_0.joint_qd.numpy(), dtype=np.float64)
                except Exception:
                    qd = None

            def _fmt(x, width=8, prec=3):
                if x is None:
                    return " " * (width - 3) + "n/a"
                try:
                    return f"{float(x):{width}.{prec}f}"
                except Exception:
                    return " " * (width - 3) + "n/a"

            # Keep the hardcoded control slots, since that is what you're actually driving.
            tracked = [
                ("base",   6),
                ("boom",   7),
                ("arm",    8),
                ("bucket", 9),
            ]

            print(f"\n[t={t:.1f}s] Control Debug")
            print(f"  Desired pose: {desired}")
            print(f"  Control targets[6:10]: [{', '.join(f'{float(targets[i]): .3f}' for _, i in tracked if i < len(targets))}]")

            # Current q/qd at the same indices. Useful if your q layout matches these slots.
            # If not, this still gives a quick sanity check without crashing.
            print("  State slots (same indices as control; useful if your layout matches):")
            for name, idx in tracked:
                q_i = float(q[idx]) if idx < len(q) else None
                qd_i = float(qd[idx]) if (qd is not None and idx < len(qd)) else None
                tgt_i = float(targets[idx]) if idx < len(targets) else None
                err_i = (tgt_i - q_i) if (tgt_i is not None and q_i is not None) else None
                print(
                    f"    {name:>6s} idx={idx:2d}  "
                    f"target={_fmt(tgt_i)}  q={_fmt(q_i)}  err={_fmt(err_i)}  qd={_fmt(qd_i)}"
                )

            # Try to read raw MuJoCo generalized forces if the wrapper exposes them.
            mj_data = None
            for attr in ("data", "mj_data", "_data", "mjd"):
                cand = getattr(self.solver, attr, None)
                if cand is not None:
                    mj_data = cand
                    break

            if mj_data is not None and hasattr(mj_data, "qfrc_actuator"):
                try:
                    qfrc_act = np.asarray(mj_data.qfrc_actuator, dtype=np.float64).reshape(-1)
                    qfrc_con = np.asarray(
                        getattr(mj_data, "qfrc_constraint", np.zeros_like(qfrc_act)),
                        dtype=np.float64,
                    ).reshape(-1)

                    print("  Generalized forces from previous step:")
                    for name, idx in tracked:
                        act_i = float(qfrc_act[idx]) if idx < len(qfrc_act) else None
                        con_i = float(qfrc_con[idx]) if idx < len(qfrc_con) else None
                        print(
                            f"    {name:>6s} idx={idx:2d}  "
                            f"actuator={_fmt(act_i, width=10, prec=2)}  "
                            f"constraint={_fmt(con_i, width=10, prec=2)}"
                        )
                except Exception as e:
                    print(f"  Generalized force readout unavailable: {e}")
            else:
                print("  Generalized force readout unavailable: raw MuJoCo mjData not exposed on self.solver.")

            # Optional: show current soil reaction wrench on the bucket body if your two-way buffers exist.
            if hasattr(self, "bucket_body_index") and self.bucket_body_index is not None:
                for buf_name in ("body_f_from_soil_prev", "body_f_from_soil"):
                    if hasattr(self, buf_name):
                        try:
                            bf = getattr(self, buf_name).numpy()
                            if self.bucket_body_index < len(bf):
                                print(f"  Bucket soil wrench ({buf_name}): {bf[self.bucket_body_index]}")
                        except Exception:
                            pass
                        break

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
        if np.isclose(self.sim_time % 1, 0): # definitely a cheap hack. Does keep GPU time up, but doesn't seem to massively improve real speed
            self.refresh_bucket_load_estimate()
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
                f"particles in bucket ({pct:.2f}%), soil_substeps={self.last_selected_mpm_substeps}, "
                f"bucket_bank_distance={self.last_bucket_bank_distance_m:.3f}, "
                f"particles_near_bucket={self.last_particles_near_bucket}"
            )
            self.last_score_print = self.sim_time

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


def main() -> None:
    viewer, args = newton.examples.init()

    # Change this preset to switch between RL throughput and validation quality.
    preset_name = "balanced"

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
