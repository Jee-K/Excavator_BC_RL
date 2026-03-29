#!/usr/bin/env python3
"""
Excavator + MPM soil simulation with necessary RL compromise.
"""
# these aren't necessary, but they make it easier to read
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# necessaries
import numpy as np

# main simulation tools
import warp as wp
import newton
import newton.examples
from newton.solvers import SolverImplicitMPM

# TODO: this solution can use fixed background kinda, but can't seem to use graphs

@dataclass(frozen=True)
class SoilProperties:
    # material information
    name: str = "moist_sandy_loam"

    # mpm primaries
    # density_kg_m3: float = 1800.0
    # youngs_modulus_pa: float = 2.0e7
    # poisson_ratio: float = 0.30
    # friction_angle_deg: float = 32.0
    # cohesion_pa: float = 12000.0

    # interface_friction_mu: float = 0.55

    density_kg_m3: float = 1650.0
    youngs_modulus_pa: float = 8.0e6
    poisson_ratio: float = 0.28
    friction_angle_deg: float = 32.0
    cohesion_pa: float = 12_000.0
    interface_friction_mu: float = 0.55
    internal_friction_mu: float = .45

@dataclass(frozen=True)
class EnvironmentPreset:
    # excavator information, rotation largely excluded
    excavator_position: tuple[float, float, float] = (0.0, 3.0, 1.1)
    excavator_platform_height_m: float = 0.6
    excavator_platform_size: tuple[float, float] = (3.2, 2.4)

    # this looks slightly better than the built-in ground in the rendered, even though they act the same
    ground_size: tuple[float, float] = (15.0, 15.0)

    # goal information
    bucket_center: tuple[float, float, float] = (-4.0, 3.0, 0.0)
    bucket_inner_half: tuple[float, float, float] = (0.75, 1.5, 0.8)
    bucket_wall_thickness: float = 0.1

    # spawn information
    slope_angle_deg: float = 32.0
    bank_height_m: float = 0.85
    bank_width_m: float = 2.6
    bank_length_m: float = 2.2
    spawn_clearance_m: float = 0.025
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
class SimulationFidelity:
    voxel_size_m: float
    rigid_substeps: int
    mpm_iterations_per_rigid: int
    rigid_ls_iterations: int
    rigid_njmax: int # ???
    fps: int
    projections : int
    particles_per_cell: int


SIM_PRESETS: dict[str, SimulationFidelity] = {
    "balanced": SimulationFidelity(
        voxel_size_m=0.020,
        rigid_substeps=6,
        mpm_iterations_per_rigid=6,
        rigid_ls_iterations=60,
        rigid_njmax=260,
        fps = 30,
        projections = 2,
        particles_per_cell = 2,
    ),
    "experimental": SimulationFidelity(
        voxel_size_m=0.030,
        rigid_substeps=3,
        mpm_iterations_per_rigid=6,
        rigid_ls_iterations=30,
        rigid_njmax=260,
        fps = 60,
        projections = 1,
        particles_per_cell = 1,
    )
}


@wp.kernel
def compute_body_forces_from_soil(
    dt: float,
    collider_ids: wp.array(dtype=int), # type: ignore
    collider_impulses: wp.array(dtype=wp.vec3), # type: ignore
    collider_impulse_pos: wp.array(dtype=wp.vec3), # type: ignore
    body_ids: wp.array(dtype=int), # type: ignore
    body_q: wp.array(dtype=wp.transform), # type: ignore
    body_com: wp.array(dtype=wp.vec3), # type: ignore
    body_f: wp.array(dtype=wp.spatial_vector), # type: ignore
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
    body_q: wp.array(dtype=wp.transform), # type: ignore
    body_qd: wp.array(dtype=wp.spatial_vector), # type: ignore
    body_f: wp.array(dtype=wp.spatial_vector), # type: ignore
    body_inv_inertia: wp.array(dtype=wp.mat33), # type: ignore
    body_inv_mass: wp.array(dtype=float), # type: ignore
    body_q_res: wp.array(dtype=wp.transform), # type: ignore
    body_qd_res: wp.array(dtype=wp.spatial_vector), # type: ignore
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
    dst: wp.array(dtype=wp.spatial_vector), # type: ignore
    src: wp.array(dtype=wp.spatial_vector), # type: ignore
):
    i = wp.tid()
    dst[i] += src[i]


class ExcavatorMPM:
    def __init__(
        self,
        viewer,
        fidelity: SimulationFidelity,
        enable_cuda_graph: bool = True,
    ):

        self.viewer = viewer
        self.device = wp.get_device()
        self.soil_info = SoilProperties()
        self.env_info = EnvironmentPreset()
        self.fidelity = fidelity

        self.voxel_size = self.fidelity.voxel_size_m

        self.fps = self.fidelity.fps
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = self.fidelity.rigid_substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.mpm_substeps_per_rigid = self.fidelity.mpm_iterations_per_rigid
        self.mpm_dt = self.sim_dt / float(self.mpm_substeps_per_rigid)
        self.enable_cuda_graph = bool(enable_cuda_graph)
        self.score_print_interval = 5.0
        self.last_score_print = 0.0
        self.settle_duration = 0.5
        self.dig_cycle_duration = 12.0

        # ------------------------------------------------------------------
        # Contact scheduling thresholds (planner / adaptive stepping only).
        # These are *not* physical collision margins.
        # ------------------------------------------------------------------
        self.bucket_proxy_radius_m = 0.45
        self.coupled_graph = None

        # ------------------------------------------------------------------
        # Physical collider tuning.
        # Keep margin small because it shifts the effective surface; use a
        # slightly larger gap to enable earlier detection without bloating the
        # bucket too much. Defaults are intentionally conservative.
        # ------------------------------------------------------------------
        self.shape_margin_m = min(max(0.08 * self.voxel_size, 0.0015), 0.0060)
        self.shape_gap_m = min(max(0.25 * self.voxel_size, 0.0080), 0.0250)
        self.soft_contact_margin_m = self.shape_gap_m

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)

        # Apply rigid defaults before URDF import. In this Newton build,
        # margin/gap live on SolverMuJoCo rather than on ShapeConfig, so keep
        # the builder-side defaults focused on friction/contact stiffness and
        # particle-collision flags only.
        self.configure_rigid_defaults(builder)
        builder.default_shape_cfg.has_particle_collision = True
        builder.default_shape_cfg.is_solid = True

        # Reusable explicit configs for static/proxy geoms added outside the URDF.
        self.static_particle_contact_cfg = newton.ModelBuilder.ShapeConfig(
            ke=1.0e6,
            kd=1.0e4,
            kf=1.0e3,
            mu=self.soil_info.interface_friction_mu,
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

        excavator_urdf = "./excavatorURDF/excavator_lowpoly_locked_splitbucket.urdf"
        print(f"Loading excavator from: {excavator_urdf}")

        control_start = len(builder.joint_target_ke)
        builder.add_urdf(
            excavator_urdf,
            xform=wp.transform(wp.vec3(*self.env_info.excavator_position), wp.quat_identity()),
            floating=True,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            ignore_inertial_definitions=False,
        )
        control_end = len(builder.joint_target_ke)        

        # Limit mpm collisions to only the portions we expect to be relevant. 
        # The actual performance impact of this is debatable
        for body in range(builder.body_count):
            if "bucketry" not in builder.body_key[body]: # !!! must remove the True or to have collision
                for shape in builder.body_shapes[body]:
                    builder.shape_flags[shape] = builder.shape_flags[shape] & ~newton.ShapeFlags.COLLIDE_PARTICLES

        # !!! control gains
        for i in range(control_start, control_end):
            builder.joint_target_ke[i] = 5000.0
            builder.joint_target_kd[i] = 500.0
        builder.joint_target_ke[7] = 50000 # back boom needs some help

        self.create_mpm_soil_bank(builder)
        self.add_ground_plane(builder)
        self.add_excavator_platform(builder)

        self.bucket_center = np.asarray(self.env_info.bucket_center, dtype=np.float64)
        self.bucket_inner_half = np.asarray(self.env_info.bucket_inner_half, dtype=np.float64)
        self.add_dump_bucket(builder)

        self.model = builder.finalize()
        # configure particle contact material
        self.model.particle_mu = self.soil_info.internal_friction_mu # particle-particle
        self.model.particle_ke = self.soil_info.youngs_modulus_pa

        self.bucket_body_index = self.model.body_key.index('bucketry')

        self._build_excavation_aabb()

        mpm_options = SolverImplicitMPM.Options()
        mpm_options.voxel_size = self.voxel_size
        mpm_options.tolerance = 1.0e-5
        mpm_options.transfer_scheme = "pic"
        mpm_options.grid_type = "sparse" # !!! we want fixed if we plan to use graphs
        mpm_options.strain_basis = "P0"
        mpm_options.max_iterations = 50
        mpm_options.critical_fraction = 0.0
        mpm_options.air_drag = 0.5
        mpm_options.collider_basis = "Q1" # big alt is P0
        mpm_options.collider_velocity_mode = "finite_difference"

        mujoco_solver_kwargs = dict(
            ls_iterations=self.fidelity.rigid_ls_iterations,
            njmax=self.fidelity.rigid_njmax,
            margin=self.shape_margin_m,
            gap=self.shape_gap_m,
            ccd_iterations = 60,
        )
        try:
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                **mujoco_solver_kwargs,
            )
            self._solver_accepts_margin_gap = True
        except TypeError:
            mujoco_solver_kwargs.pop("margin", None)
            mujoco_solver_kwargs.pop("gap", None)
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                **mujoco_solver_kwargs,
            )
            self._solver_accepts_margin_gap = False
            print("Warning: SolverMuJoCo margin/gap kwargs unsupported in this build; using solver defaults.")
        self.mpm_model = None
        model_ctor = getattr(SolverImplicitMPM, "Model", None)
        if model_ctor is not None:
            try:
                self.mpm_model = model_ctor(self.model, mpm_options)
            except Exception as exc:
                self.mpm_model = None
                print(f"Warning: native MPM material model unavailable in this build: {exc}")

        solver_input = self.mpm_model if self.mpm_model is not None else self.model
        try:
            self.mpm_solver = SolverImplicitMPM(solver_input, mpm_options)
        except Exception:
            self.mpm_solver = SolverImplicitMPM(self.model, mpm_options)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.mpm_state = self.model.state()
        # Only the dedicated MPM working state needs solver-specific
        # enrichment in this split rigid/MPM loop. Keeping state_0/state_1
        # lean avoids duplicating APIC/Q1 auxiliary particle storage.
        self._enrich_state_for_mpm(self.mpm_state)
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
        self._collect_collider_impulses(self.mpm_state)

        self.control = self.model.control()
        self._joint_target_host = self.control.joint_target_pos.numpy()
        self.control_size = int(self._joint_target_host.shape[0])

        self.control_joint_names = self.model.joint_key
        self.control_lower, self.control_upper = self.model.joint_limit_lower, self.model.joint_limit_upper
        self.joint_map = {"swing": 6, "arm":7, "stick":8, "bucket":9}

        self.target_replay_path = Path("./bc_component/BC_dataset/dig/dig_2_bc_dataset.npz")
        self.target_replay_hz = 10.0
        self.target_replay_loop = True
        self.target_replay_interp = "linear"
        self.target_replay_start_time_s = 0.0
        self.target_replay_states: Optional[np.ndarray] = None
        self.target_replay_key: Optional[str] = None
        self.target_replay_active = False
        self._load_target_state_replay()

        self.total_particles = int(self.model.particle_count)
        self.particles_in_bucket = 0

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.viewer.show_visual = False
        self.viewer.show_collision = True
        self.viewer.show_cloth = False

        # self.capture() # !!!

    # ------------------------------------------------------------------
    # Target-state replay inputs
    # ------------------------------------------------------------------
    @staticmethod
    def _select_numeric_array_from_npz(data: np.lib.npyio.NpzFile) -> tuple[str, np.ndarray]:
        preferred_keys = ("target_states", "states", "targets", "target", "arr_0")
        for key in preferred_keys:
            if key in data.files:
                arr = np.asarray(data[key])
                if np.issubdtype(arr.dtype, np.number):
                    return key, arr
        for key in data.files:
            arr = np.asarray(data[key])
            if np.issubdtype(arr.dtype, np.number):
                return key, arr
        raise ValueError("No numeric arrays found in target-state npz file.")

    def _load_target_state_replay(self) -> None:
        self.target_replay_active = False
        if not self.target_replay_path:
            print(
                "Target-state replay disabled. Place target_angles.npz next to this Python file to drive the excavator from a 10 Hz target-state file."
            )
            return

        try:
            with np.load(self.target_replay_path, allow_pickle=False) as data:
                key, states = self._select_numeric_array_from_npz(data)
        except Exception as exc:
            print(f"Failed to load target-state replay file '{self.target_replay_path.name}': {exc}")
            return

        states = np.asarray(states, dtype=np.float64)
        if states.ndim == 1:
            states = states.reshape(1, -1)
        if states.ndim != 2:
            print(
                f"Target-state replay file must contain a 2D array [T, D], but got shape {states.shape}."
            )
            return

        self.target_replay_states = states
        self.target_replay_key = key

        # hardcoded, but checked
        assert states.shape[1] == 4
        
        self.target_replay_active = bool(states.shape[0] > 0)

        duration_s = 0.0
        if states.shape[0] > 1 and self.target_replay_hz > 0.0:
            duration_s = float(states.shape[0] - 1) / float(self.target_replay_hz)
        print(
            "Loaded target-state replay:",
            f"file={self.target_replay_path.name}",
            f"array={key}",
            f"hz={self.target_replay_hz:.3f}",
            f"loop={self.target_replay_loop}",
            f"duration_s={duration_s:.3f}",
        )

    def _sample_target_state_row(self, sim_time_s: float) -> Optional[np.ndarray]:

        states = self.target_replay_states
        n = int(states.shape[0])

        replay_time = max(0.0, float(sim_time_s) - self.target_replay_start_time_s)
        replay_sample = replay_time * float(self.target_replay_hz)
        max_sample = float(n - 1)

        replay_sample = float(np.clip(replay_sample, 0.0, max_sample))

        idx0 = int(np.floor(replay_sample))

        # return states[idx0] # could choose to interpolate instead
        return states[idx0]

    def _get_replay_desired_map(self) -> Optional[dict[str, float]]:
        row = self._sample_target_state_row(self.sim_time)

        return {
            "swing": 0, # !!!
            "arm": float(row[1]),
            "stick": float(row[2]),
            "bucket": float(row[3]),
        }

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
        builder.default_shape_cfg.mu = self.soil_info.interface_friction_mu

    def add_ground_plane(self, builder) -> None:
        width, length = self.env_info.ground_size
        builder.add_shape_plane(
            body=-1,
            cfg=self.static_ground_cfg,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            width=width,
            length=length,
        )

    def add_excavator_platform(self, builder) -> None:
        width, length = self.env_info.excavator_platform_size
        height = self.env_info.excavator_platform_height_m
        if height <= 0.0:
            return

        ex, ey, _ = self.env_info.excavator_position
        builder.add_shape_box(
            body=-1,
            cfg=self.static_ground_cfg,
            xform=wp.transform(wp.vec3(ex, ey, 0.5 * height), wp.quat_identity()),
            hx=0.5 * width,
            hy=0.5 * length,
            hz=0.5 * height,
        )

    def add_dump_bucket(self, builder) -> None:
        bx, by, bz = self.bucket_center
        iw, il, idepth = self.bucket_inner_half
        t = self.env_info.bucket_wall_thickness

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

    def _enrich_state_for_mpm(self, state) -> None:
        enrich = getattr(self.mpm_solver, "enrich_state", None)
        if enrich is None:
            return
        try:
            enrich(state)
        except Exception:
            pass

    @staticmethod
    def _copy_assignable_fields(src, dst, prefixes: tuple[str, ...]) -> None:
        for name in dir(src):
            if not any(name.startswith(prefix) for prefix in prefixes):
                continue
            if not hasattr(dst, name):
                continue
            src_attr = getattr(src, name)
            dst_attr = getattr(dst, name)
            if not hasattr(dst_attr, "assign"):
                continue
            try:
                dst_attr.assign(src_attr)
            except Exception:
                pass

    def _particle_cell_offsets(self) -> np.ndarray:
        ppc = max(1, int(self.fidelity.particles_per_cell))
        if ppc == 1:
            return np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
        if ppc == 2:
            return np.array([
                [0.25, 0.25, 0.25],
                [0.75, 0.75, 0.75],
            ], dtype=np.float32)
        if ppc == 4:
            return np.array([
                [0.25, 0.25, 0.25],
                [0.75, 0.75, 0.25],
                [0.75, 0.25, 0.75],
                [0.25, 0.75, 0.75],
            ], dtype=np.float32)
        grid_n = int(np.ceil(ppc ** (1.0 / 3.0)))
        coords = (np.arange(grid_n, dtype=np.float32) + 0.5) / float(grid_n)
        grid = np.stack(np.meshgrid(coords, coords, coords, indexing="ij"), axis=-1).reshape(-1, 3)
        return grid[:ppc].astype(np.float32, copy=False)

    # def _compute_native_mpm_material_arrays(self) -> dict[str, np.ndarray]:
    #     particle_count = int(self.model.particle_count)
    #     phi = np.radians(self.soil_info.friction_angle_deg)
    #     # Currently biases the built-in material toward dry, frictional bulk behavior.
    #     # In the low-memory regime, extra apparent cohesion and any non-trivial
    #     # tensile support show up visually as sticky clumping.
    #     friction = float(1.10 * np.tan(phi))
    #     cohesion = float(0.25 * self.soil_info.cohesion_pa)
    #     yield_pressure = max(cohesion / max(friction, 1.0e-3), 5.0e2)
    #     yield_stress = max(np.sqrt(3.0) * cohesion, 5.0e2)
    #     hardening = 0.0
    #     tensile_yield_ratio = float(self.native_mpm.tensile_yield_ratio)
    #     return {
    #         "yield_pressure": np.full(particle_count, yield_pressure, dtype=np.float32),
    #         "yield_stress": np.full(particle_count, yield_stress, dtype=np.float32),
    #         "tensile_yield_ratio": np.full(particle_count, tensile_yield_ratio, dtype=np.float32),
    #         "friction": np.full(particle_count, friction, dtype=np.float32),
    #         "hardening": np.full(particle_count, hardening, dtype=np.float32),
    #     }

    def create_mpm_soil_bank(self, builder : newton.ModelBuilder) -> None:
        x_half = 0.5 * self.env_info.bank_width_m
        y_front = self.env_info.bank_front_y
        y_back = self.env_info.bank_back_y
        max_height = self.env_info.bank_height_m
        z0 = self.env_info.spawn_clearance_m
        slope_tan = np.tan(np.radians(self.env_info.slope_angle_deg))

        nx = int(np.ceil(self.env_info.bank_width_m / self.voxel_size))
        ny = int(np.ceil(self.env_info.bank_length_m / self.voxel_size))
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
        occupied_cells = cell_origins[occupancy_mask].astype(np.float32, copy=False)

        rng = np.random.default_rng(7)
        jitter = rng.random((occupied_cells.shape[0], 1, 3), dtype=np.float32) * self.voxel_size
        positions = (occupied_cells[:, None, :] + jitter).reshape(-1, 3)

        particle_mass = self.soil_info.density_kg_m3 * ((self.voxel_size / 2)** 3) * 4/3 * np.pi
        for p in positions:
            builder.add_particle(
                pos=wp.vec3(float(p[0]), float(p[1]), float(p[2])),
                vel=wp.vec3(0.0),
                mass=particle_mass,
                radius=self.voxel_size / 2,
            )

        e = self.soil_info.youngs_modulus_pa
        nu = self.soil_info.poisson_ratio
        builder.mpm_E = e
        builder.mpm_nu = nu
        builder.mpm_mu = e / (2.0 * (1.0 + nu))
        builder.mpm_lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    def _build_excavation_aabb(self) -> None:
        x_half = 0.5 * self.env_info.bank_width_m
        self.excavation_aabb_min = np.array(
            [-x_half - 0.30, self.env_info.bank_back_y - 0.30, -0.05],
            dtype=np.float64,
        )
        self.excavation_aabb_max = np.array(
            [x_half + 0.30, self.env_info.bank_front_y + 0.30, self.env_info.bank_height_m + 0.20],
            dtype=np.float64,
        )


    # ------------------------------------------------------------------
    # Simulation stepping / soil-contact scheduling
    # ------------------------------------------------------------------
    def _clone_wp_array(self, arr):
        clone = wp.zeros_like(arr)
        clone.assign(arr)
        return clone

    def _snapshot_runtime_state(self) -> dict[str, object]:
        return {
            "state_0_joint_q": self._clone_wp_array(self.state_0.joint_q),
            "state_0_joint_qd": self._clone_wp_array(self.state_0.joint_qd),
            "state_0_body_q": self._clone_wp_array(self.state_0.body_q),
            "state_0_body_qd": self._clone_wp_array(self.state_0.body_qd),
            "state_0_body_f": self._clone_wp_array(self.state_0.body_f),
            "state_0_particle_q": self._clone_wp_array(self.state_0.particle_q),
            "state_0_particle_qd": self._clone_wp_array(self.state_0.particle_qd),
            "state_1_joint_q": self._clone_wp_array(self.state_1.joint_q),
            "state_1_joint_qd": self._clone_wp_array(self.state_1.joint_qd),
            "state_1_body_q": self._clone_wp_array(self.state_1.body_q),
            "state_1_body_qd": self._clone_wp_array(self.state_1.body_qd),
            "state_1_body_f": self._clone_wp_array(self.state_1.body_f),
            "state_1_particle_q": self._clone_wp_array(self.state_1.particle_q),
            "state_1_particle_qd": self._clone_wp_array(self.state_1.particle_qd),
            "mpm_state_body_q": self._clone_wp_array(self.mpm_state.body_q),
            "mpm_state_body_qd": self._clone_wp_array(self.mpm_state.body_qd),
            "mpm_state_body_f": self._clone_wp_array(self.mpm_state.body_f),
            "mpm_state_particle_q": self._clone_wp_array(self.mpm_state.particle_q),
            "mpm_state_particle_qd": self._clone_wp_array(self.mpm_state.particle_qd),
            "collider_body_q": self._clone_wp_array(self.collider_body_q),
            "collider_body_qd": self._clone_wp_array(self.collider_body_qd),
            "body_f_from_soil": self._clone_wp_array(self.body_f_from_soil),
            "body_f_from_soil_prev": self._clone_wp_array(self.body_f_from_soil_prev),
            "sim_time": float(self.sim_time),
        }

    def _restore_runtime_state(self, snapshot: dict[str, object]) -> None:
        self.state_0.joint_q.assign(snapshot["state_0_joint_q"])
        self.state_0.joint_qd.assign(snapshot["state_0_joint_qd"])
        self.state_0.body_q.assign(snapshot["state_0_body_q"])
        self.state_0.body_qd.assign(snapshot["state_0_body_qd"])
        self.state_0.body_f.assign(snapshot["state_0_body_f"])
        self.state_0.particle_q.assign(snapshot["state_0_particle_q"])
        self.state_0.particle_qd.assign(snapshot["state_0_particle_qd"])
        self.state_1.joint_q.assign(snapshot["state_1_joint_q"])
        self.state_1.joint_qd.assign(snapshot["state_1_joint_qd"])
        self.state_1.body_q.assign(snapshot["state_1_body_q"])
        self.state_1.body_qd.assign(snapshot["state_1_body_qd"])
        self.state_1.body_f.assign(snapshot["state_1_body_f"])
        self.state_1.particle_q.assign(snapshot["state_1_particle_q"])
        self.state_1.particle_qd.assign(snapshot["state_1_particle_qd"])
        self.mpm_state.body_q.assign(snapshot["mpm_state_body_q"])
        self.mpm_state.body_qd.assign(snapshot["mpm_state_body_qd"])
        self.mpm_state.body_f.assign(snapshot["mpm_state_body_f"])
        self.mpm_state.particle_q.assign(snapshot["mpm_state_particle_q"])
        self.mpm_state.particle_qd.assign(snapshot["mpm_state_particle_qd"])
        self.collider_body_q.assign(snapshot["collider_body_q"])
        self.collider_body_qd.assign(snapshot["collider_body_qd"])
        self.body_f_from_soil.assign(snapshot["body_f_from_soil"])
        self.body_f_from_soil_prev.assign(snapshot["body_f_from_soil_prev"])
        self.sim_time = float(snapshot["sim_time"])

    def capture(self) -> None:
        if not self.enable_cuda_graph:
            return

        snapshot = self._snapshot_runtime_state()
        try:
            with wp.ScopedCapture(device=self.device) as capture:
                self.simulate_coupled_frame_fixed(apply_viewer_forces=False)
            self.coupled_graph = capture.graph
            print(
                "Captured fixed-ratio CUDA graph: "
                f"rigid_substeps={self.sim_substeps}, mpm_substeps_per_rigid={self.mpm_substeps_per_rigid}"
            )
        except Exception as exc:
            self.coupled_graph = None
            print(f"CUDA graph capture unavailable for this configuration: {exc}")
        finally:
            self._restore_runtime_state(snapshot)

    def _copy_state_to_mpm_state(self) -> None:
        self._copy_assignable_fields(self.state_0, self.mpm_state, prefixes=("body_", "particle_", "mpm_", "mpm_particle_"))

    def _copy_particles_from_mpm_state(self) -> None:
        self._copy_assignable_fields(self.mpm_state, self.state_0, prefixes=("particle_", "mpm_", "mpm_particle_"))

    def _sync_mpm_collider_pose(self, source_state=None) -> None:
        source_state = self.state_0 if source_state is None else source_state
        self.collider_body_q.assign(source_state.body_q)
        if self._collider_accepts_body_qd:
            self.collider_body_qd.assign(source_state.body_qd)

    def _zero_body_force_buffer(self, buf) -> None:
        buf.assign(self._zero_body_force)

    def _collect_collider_impulses(self, state) -> bool:
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
                        return ExcavatorMPM._coerce_vec3(obj[key])
                for key in obj.dtype.names:
                    candidate = ExcavatorMPM._coerce_vec3(obj[key])
                    if candidate is not None:
                        return candidate
            return None
        if isinstance(obj, np.ndarray):
            if obj.dtype.names:
                return ExcavatorMPM._coerce_vec3(obj[()])
            flat = np.asarray(obj, dtype=np.float64).reshape(-1)
            if flat.size >= 3:
                return flat[:3]
            return None
        if hasattr(obj, "p"):
            return ExcavatorMPM._coerce_vec3(getattr(obj, "p"))
        if hasattr(obj, "tolist"):
            return ExcavatorMPM._coerce_vec3(obj.tolist())
        if isinstance(obj, (tuple, list)):
            if len(obj) >= 3 and all(np.isscalar(v) for v in obj[:3]):
                return np.asarray(obj[:3], dtype=np.float64)
            for item in obj:
                candidate = ExcavatorMPM._coerce_vec3(item)
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
        """CPU-side diagnostics update kept off the hot path.

        This function intentionally performs host reads and should only be called
        occasionally (for example at score-print intervals), not every frame.
        """
        pos = self._get_bucket_world_position()
        if pos is None or self.model.particle_count == 0:
            return

        try:
            positions = self.state_0.particle_q.numpy()
        except Exception:
            return

        radius = self.bucket_proxy_radius_m + 0.45
        delta = positions - pos[None, :]
        near = np.einsum("ij,ij->i", delta, delta) <= radius * radius

    def simulate_rigid_substep(self, dt: float, apply_viewer_forces: bool = True) -> None:
        self.state_0.clear_forces()
        if apply_viewer_forces:
            self.viewer.apply_forces(self.state_0)
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
        # this loop is by far the biggest compute-time sink
        for _ in range(count):
            self._sync_mpm_collider_pose(self.mpm_state)
            self.mpm_solver.step(self.mpm_state, self.mpm_state, contacts=None, control=None, dt=dt)
            for _ in range(self.fidelity.projections):
                self.mpm_solver.project_outside(self.mpm_state, self.mpm_state, dt)
            if self._collect_collider_impulses(self.mpm_state):
                self._compute_soil_reaction_forces(dt_divisor)
        self._copy_particles_from_mpm_state()
        self.body_f_from_soil_prev.assign(self.body_f_from_soil)

    def simulate_coupled_frame_fixed(self, apply_viewer_forces: bool = True) -> None:
        for _ in range(self.sim_substeps):
            self.simulate_rigid_substep(self.sim_dt, apply_viewer_forces=apply_viewer_forces)
            self.simulate_soil_substeps(self.mpm_substeps_per_rigid, self.mpm_dt)

    # ------------------------------------------------------------------
    # Controller (kept mainly for the built-in digging demo)
    # ------------------------------------------------------------------
    @staticmethod
    def smoothstep(u: float) -> float:
        u = float(np.clip(u, 0.0, 1.0))
        return u * u * (3.0 - 2.0 * u)

    @staticmethod
    def blend_pose(a: DigPose, b: DigPose, u: float) -> DigPose:
        s = ExcavatorMPM.smoothstep(u)
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

    def apply_control(self) -> None:
        if self.control_size == 0:
            return

        targets = self._joint_target_host.copy()
        desired_map = self._get_replay_desired_map()

        if desired_map is None:
            raise RuntimeError()
            if self.sim_time < self.settle_duration:
                desired = DigPose(0.0, 0.55, 0.10, 0.55)
            else:
                cycle_t = (self.sim_time - self.settle_duration) % self.dig_cycle_duration
                desired = self.sample_dig_cycle(cycle_t)

            desired_map = {
                "swing": desired.swing,
                "arm": desired.arm,
                "stick": desired.stick,
                "bucket": desired.bucket,
            }

        joint_vel_limit = {
            "swing": 1.0,
            "arm": 0.8,
            "stick": 1.5,
            "bucket": 3.0,
        }

        print(self.sim_time)
        # print([desired_map["swing"], desired_map["arm"], desired_map["stick"], desired_map["bucket"]])

        q_prevs = self.state_0.joint_q.numpy()
        print(q_prevs[-4:])

        for idx, section in enumerate(["swing", "arm", "stick", "bucket"], start=6):
            q_prev = q_prevs[idx]
            dq_max = float(joint_vel_limit[section]) * self.frame_dt
            q_cmd = desired_map[section]
            q_cmd = q_prev + np.clip(q_cmd - q_prev, -dq_max, dq_max)
            # might choose to clip to behaviour range, but the model won't disobey them regardless
            q_cmd = desired_map[section]
            targets[idx] = q_cmd

        print([desired_map["swing"], desired_map["arm"], desired_map["stick"], desired_map["bucket"]])
        print(targets[-4:])
        

        self.control.joint_target_pos.assign(targets)

        # if int(self.sim_time) % 1 == 0 and self.sim_time - int(self.sim_time) < self.frame_dt:
        #     print(targets)
        #     replay_info = "manual_demo"
        #     if self.target_replay_active:
        #         replay_info = (
        #             f"target_replay[{self.target_replay_key}] "
        #         )
        #     print(f"\n[t={self.sim_time:.1f}s] Control target: {targets[6:10]} ({replay_info})")
        #     print(f"  Current pos: [{', '.join([format(float(x), '6.3f') for x in self.state_0.joint_q.numpy()])}]")

    # ------------------------------------------------------------------
    # Task metrics / main loop
    # ------------------------------------------------------------------
    def count_particles_in_bucket(self) -> int:
        if self.model.particle_count == 0:
            return 0

        positions = self.state_0.particle_q.numpy()
        bx, by, bz = self.bucket_center
        iw, il, idepth = self.bucket_inner_half
        t = self.env_info.bucket_wall_thickness
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

        if self.coupled_graph:
            print("Success")
            wp.capture_launch(self.coupled_graph)
        else:
            self.simulate_coupled_frame_fixed(apply_viewer_forces=True)

        self.sim_time += self.frame_dt

        if self.sim_time - self.last_score_print >= self.score_print_interval:
            self.refresh_bucket_load_estimate()
            self.particles_in_bucket = self.count_particles_in_bucket()
            pct = 100.0 * self.particles_in_bucket / max(self.total_particles, 1)
            print(
                f"\n[t={self.sim_time:.1f}s] SCORE: {self.particles_in_bucket:,} / {self.total_particles:,} "
                f"particles in bucket ({pct:.2f}%) "
            )
            self.last_score_print = self.sim_time


        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


def main() -> None:
    viewer, args = newton.examples.init()


    example = ExcavatorMPM(
        viewer,
        fidelity=SIM_PRESETS["experimental"],
    )

    while viewer.is_running():
        example.step()


if __name__ == "__main__":
    main()
