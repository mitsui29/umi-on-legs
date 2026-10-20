"""Isaac Lab 1.2 / Isaac Sim 4.2 configuration (import after AppLauncher)."""

from omni.isaac.lab.actuators import IdealPDActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnvCfg
from omni.isaac.lab.envs.common import ViewerCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg
from omni.isaac.lab.sim import SimulationCfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils import configclass

from .assets import ROOT, prepare_urdf
from .config import DEFAULT_POS, JOINT_NAMES


@configclass
class ReachingEnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 7.0
    num_actions = 18
    num_observations = 96
    num_states = 225
    sim = SimulationCfg(
        dt=0.005, render_interval=4,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
            friction_combine_mode="average", restitution_combine_mode="average"),
        physx=sim_utils.PhysxCfg(bounce_threshold_velocity=0.5, gpu_max_rigid_contact_count=2**24),
    )
    viewer = ViewerCfg(eye=(2.5, 2.5, 2.0), lookat=(0.0, 0.0, 0.4), origin_type="env")
    scene = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)
    contacts = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/.*", update_period=0.005, history_length=1)
    robot: ArticulationCfg | None = None
    trajectory_path: str | None = None
    trajectory_dt: float = 0.005
    # Synthetic trajectories are an explicit smoke-test option, never a silent fallback.
    synthetic: bool = False
    randomization: bool = True
    observation_noise: bool = True
    control_delay_s: float = 0.020


def make_cfg(*, num_envs, device, seed, trajectory_path=None, trajectory_dt=0.005,
             synthetic=False, randomization=True, observation_noise=True, cache_dir=None):
    cfg = ReachingEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.sim.device = device
    cfg.seed = seed
    cfg.trajectory_path = trajectory_path
    cfg.trajectory_dt = trajectory_dt
    cfg.synthetic = synthetic
    cfg.randomization = randomization
    cfg.observation_noise = observation_noise
    cache = str(cache_dir or ROOT / ".cache/isaaclab/go2_arx5")
    cfg.robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=prepare_urdf(cache), usd_dir=cache, usd_file_name="go2_arx5.usd",
            fix_base=False, merge_fixed_joints=True, self_collision=True,
            default_drive_type="none", make_instanceable=False,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, linear_damping=0.0, angular_damping=0.0,
                max_linear_velocity=1000.0, max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=4,
                solver_velocity_iteration_count=0),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(-0.5, 0.0, 0.3),
                                                 joint_pos=dict(zip(JOINT_NAMES, DEFAULT_POS))),
        # Manual PD below runs every physics step; this actuator only passes effort through.
        actuators={"effort": IdealPDActuatorCfg(joint_names_expr=[".*"],
                                               stiffness=0.0, damping=0.0, effort_limit=1000.0)},
    )
    return cfg
