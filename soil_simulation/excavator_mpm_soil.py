#!/usr/bin/env python3
"""
Lighter-weight, task-specific excavator + MPM soil simulation
"""
# these aren't necessary, but they make it easier to read
from typing import Iterable

# necessaries
from dataclasses import dataclass
import numpy as np

# main simulation tools
import warp as wp
import newton
import newton.examples
from newton.solvers import SolverImplicitMPM

@dataclass(frozen=True)
class SoilProperties:
    # material information
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
class SimulationFidelity:
    voxel_size_m: float
    rigid_substeps: int
    mpm_iterations_per_rigid: int
    rigid_ls_iterations: int
    rigid_njmax: int # the values for this aren't particularly well motivated, but do fit precedent
    fps: int
    projections : int


SIM_PRESETS: dict[str, SimulationFidelity] = {
    "balanced": SimulationFidelity(
        voxel_size_m=0.020,
        rigid_substeps=6,
        mpm_iterations_per_rigid=6,
        rigid_ls_iterations=60,
        rigid_njmax=260,
        fps = 30,
        projections = 2,
    ),
    "experimental": SimulationFidelity(
        voxel_size_m=0.030,
        rigid_substeps=3,
        mpm_iterations_per_rigid=6,
        rigid_ls_iterations=30,
        rigid_njmax=260,
        fps = 60,
        projections = 1,
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
        viewer: newton.viewer,
        fidelity: SimulationFidelity,
        enable_cuda_graph: bool = True,
        debug: bool = False
    ):

        self.viewer = viewer
        self.device = wp.get_device()
        self.soil_info = SoilProperties()
        self.env_info = EnvironmentPreset()
        self.fidelity = fidelity

        self.debug = debug

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

        self.coupled_graph = None
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)

        # Apply rigid defaults before URDF import
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
            if "bucketry" not in builder.body_key[body]: 
                for shape in builder.body_shapes[body]:
                    builder.shape_flags[shape] = builder.shape_flags[shape] & ~newton.ShapeFlags.COLLIDE_PARTICLES

        # control gains
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

        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            ls_iterations=self.fidelity.rigid_ls_iterations,
            njmax=self.fidelity.rigid_njmax,
            ccd_iterations = 60,
        )

        self.mpm_solver = SolverImplicitMPM(self.model, mpm_options)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.mpm_state = self.model.state()
        self.mpm_state.body_q = wp.empty_like(self.state_0.body_q)
        self.mpm_state.body_qd = wp.empty_like(self.state_0.body_qd)
        self.mpm_state.body_f = wp.empty_like(self.state_0.body_f)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self._copy_state_to_mpm_state()

        # Collider buffer for the MPM contact path.
        self.collider_body_q = wp.zeros_like(self.state_0.body_q)
        self.collider_body_q.assign(self.state_0.body_q)
        self.mpm_solver.setup_collider(
            model=self.model,
            body_mass=wp.zeros_like(self.model.body_mass),
            body_q=self.collider_body_q,
        )

        # Two-way coupling buffers (adapted from Newton's mpm_twoway_coupling example).
        max_nodes = 1 << 20
        self.collider_impulses = wp.zeros(max_nodes, dtype=wp.vec3, device=self.model.device)
        self.collider_impulse_pos = wp.zeros(max_nodes, dtype=wp.vec3, device=self.model.device)
        self.collider_impulse_ids = wp.full(max_nodes, value=-1, dtype=int, device=self.model.device)
        self.collider_body_id = self.mpm_solver.collider_body_index
        self.body_f_from_soil = wp.zeros_like(self.state_0.body_f)
        self.body_f_from_soil_prev = wp.zeros_like(self.state_0.body_f)
        self._zero_body_force = wp.zeros_like(self.state_0.body_f)
        self._collect_collider_impulses(self.mpm_state)

        self.control = self.model.control()
        self._joint_target_host = self.control.joint_target_pos.numpy()

        self.control_lower, self.control_upper = self.model.joint_limit_lower, self.model.joint_limit_upper
        self.first_joint_idx = 6
        self.joint_vel_limits = [1.0, 0.8, 1.5, 3.0] # swing, arm, stick bucket # ??? could be parsed from urdf

        self.total_particles = int(self.model.particle_count)
        self.particles_in_bucket = 0

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.viewer.show_visual = False
        self.viewer.show_collision = True
        self.viewer.show_triangles = False # corresponds to the cloth option

        # self.capture() # !!!

    # Scene and material setup
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

    # ??? consider using some of these values to improve simulation
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
        y_front = self.env_info.bank_front_y
        y_back = self.env_info.bank_back_y
        max_height = self.env_info.bank_height_m
        z0 = self.env_info.spawn_clearance_m
        slope_tan = np.tan(np.radians(self.env_info.slope_angle_deg))

        nx = int(np.ceil(self.env_info.bank_width_m / self.voxel_size))
        ny = int(np.ceil(self.env_info.bank_length_m / self.voxel_size))
        nz = int(np.ceil((max_height + z0 + self.voxel_size) / self.voxel_size))

        xs = -0.5 * self.env_info.bank_width_m + np.arange(nx, dtype=np.float32) * self.voxel_size
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


    # Simulation stepping / soil-contact scheduling
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
        self.mpm_state.body_f.assign(self.state_0.body_f)
        self.mpm_state.particle_q.assign(self.state_0.particle_q)
        self.mpm_state.particle_qd.assign(self.state_0.particle_qd)
        self.mpm_state.body_q.assign(self.state_0.body_q)
        self.mpm_state.body_qd.assign(self.state_0.body_qd)
        self.mpm_state.particle_f.assign(self.state_0.particle_f)

    def _copy_particles_from_mpm_state(self) -> None:
        self.state_0.particle_q.assign(self.mpm_state.particle_q)
        self.state_0.particle_qd.assign(self.mpm_state.particle_qd)
        self.state_0.particle_f.assign(self.mpm_state.particle_f)


    def _sync_mpm_collider_pose(self, source_state=None) -> None:
        source_state = self.state_0 if source_state is None else source_state
        self.collider_body_q.assign(source_state.body_q)

    def _collect_collider_impulses(self, state) -> None:
        collider_impulses, collider_impulse_pos, collider_impulse_ids = self.mpm_solver.collect_collider_impulses(state)

        self.collider_impulse_ids.fill_(-1)
        n_colliders = min(collider_impulses.shape[0], self.collider_impulses.shape[0])

        self.collider_impulses[:n_colliders].assign(collider_impulses[:n_colliders])
        self.collider_impulse_pos[:n_colliders].assign(collider_impulse_pos[:n_colliders])
        self.collider_impulse_ids[:n_colliders].assign(collider_impulse_ids[:n_colliders])


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

        self.body_f_from_soil.assign(self._zero_body_force) # zeroes body force buffer
        dt_divisor = float(count * dt)
        # this loop is by far the biggest compute-time sink
        for _ in range(count):
            self._sync_mpm_collider_pose(self.mpm_state)
            self.mpm_solver.step(self.mpm_state, self.mpm_state, contacts=None, control=None, dt=dt)
            for _ in range(self.fidelity.projections):
                self.mpm_solver.project_outside(self.mpm_state, self.mpm_state, dt)
            self._compute_soil_reaction_forces(dt_divisor)
        self._copy_particles_from_mpm_state()
        self.body_f_from_soil_prev.assign(self.body_f_from_soil)

    def simulate_coupled_frame_fixed(self, apply_viewer_forces: bool = True) -> None:
        for _ in range(self.sim_substeps):
            self.simulate_rigid_substep(self.sim_dt, apply_viewer_forces=apply_viewer_forces)
            self.simulate_soil_substeps(self.mpm_substeps_per_rigid, self.mpm_dt)

    # Controller
    def apply_control(self, user_targets : Iterable[float, float, float, float] | None = None) -> None:
        if not user_targets:
            return

        targets = self._joint_target_host.copy()
        

        if self.sim_time < self.settle_duration:
            user_targets = [0.0, 0.55, 0.10, 0.55]

        q_prevs = self.state_0.joint_q.numpy()

        # this could be vectorized, likely
        for local_idx, target in enumerate(user_targets):
            idx = local_idx + self.first_joint_idx
            
            q_prev = q_prevs[idx]
            dq_max = self.joint_vel_limits[local_idx] * self.frame_dt
            q_cmd = target
            # q_cmd = q_prev + np.clip(q_cmd - q_prev, -dq_max, dq_max) # !!! this is what enforces motion limits
            targets[idx] = q_cmd

        if self.debug:
            print(self.sim_time)
            print(q_prevs[-4:])
            print(user_targets)
            print(targets[-4:])
        
        # may desire some way to detect if the excavator leaves the platform
        # for our purposes, we'll presume that leaving the platform is very unlikely
        # to achieve any positive results within the time limit and just
        # leave it at that

        self.control.joint_target_pos.assign(targets)

    # Task metrics / main loop
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
            if self.debug:
                print("Graph does exist, attempting launch")
            wp.capture_launch(self.coupled_graph)
        else:
            self.simulate_coupled_frame_fixed(apply_viewer_forces=True)

        self.sim_time += self.frame_dt

        if self.sim_time - self.last_score_print >= self.score_print_interval:
            self.particles_in_bucket = self.count_particles_in_bucket()

            if self.debug:
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

    preset = SIM_PRESETS["experimental"]

    sim_env = ExcavatorMPM(
        viewer,
        fidelity=preset,
        debug=True
    )

    COMMAND_HZ = 10

    while viewer.is_running():
        for _ in range(preset.fps // COMMAND_HZ):
            sim_env.step()
        # various data can be pulled from sim_env.attribute_name here, namely sim_env.state_0.joint_q has the current position information
        # sim_env.state_0.joint_q.numpy()
        sim_env.apply_control(None) # commands can be put in here


if __name__ == "__main__":
    main()
