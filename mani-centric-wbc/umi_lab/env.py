"""Go2 + ARX5 world-frame end-effector tracking on Isaac Lab 1.2.

Policy tensors always follow config.JOINT_NAMES, never PhysX traversal order.
"""

import torch

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.envs import DirectRLEnv
from omni.isaac.lab.sensors import ContactSensor
import omni.isaac.lab.sim as sim_utils

from .config import (DEFAULT_POS, FOOT_NAMES, JOINT_NAMES, KD, KP, ORN_CURRICULUM,
                     POS_CURRICULUM, RANDOM_BODY_NAMES, TARGET_TIMES, TORQUE_LIMIT, ordered_indices)
from .math import (euler_xyz_to_quat, quat_mul, quat_to_matrix,
                   relative_pose_observation, rotation_angle)
from .trajectory import TrajectoryLibrary


class ReachingEnv(DirectRLEnv):
    def __init__(self, cfg, **kwargs):
        if not cfg.synthetic and not cfg.trajectory_path:
            raise ValueError("Provide --trajectory PATH, or explicitly use --synthetic for a smoke test")
        self.trajectories = (TrajectoryLibrary(cfg.trajectory_path, cfg.trajectory_dt, cfg.episode_length_s)
                             if cfg.trajectory_path else None)
        super().__init__(cfg, **kwargs)
        if self.robot.num_joints != 18:
            raise ValueError(f"Expected 18 actuated joints, found {self.robot.joint_names}")
        self.joints = ordered_indices(self.robot.joint_names, JOINT_NAMES)
        self.feet = ordered_indices(self.robot.body_names, FOOT_NAMES)
        self.hips = ordered_indices(self.robot.body_names, [n.replace("foot", "thigh") for n in FOOT_NAMES])
        self.ee = ordered_indices(self.robot.body_names, ["end_effector"])[0]
        self.random_bodies = ordered_indices(self.robot.body_names, RANDOM_BODY_NAMES)
        self.contact_feet = ordered_indices(self.contacts.body_names, FOOT_NAMES)
        termination = ("base", "hip", "Head", "thigh", "base_arm_link", "link1", "link2", "link3", "link4", "link5")
        self.termination_contacts = [i for i, n in enumerate(self.contacts.body_names)
                                     if any(part in n for part in termination)]
        collision_names = ("Head_lower", "Head_upper", "base") + tuple(
            f"{leg}_{part}" for leg in ("FR", "FL", "RR", "RL") for part in ("hip", "thigh", "calf")
        ) + tuple(f"link{i}" for i in range(1, 6))
        self.collision_contacts = ordered_indices(self.contacts.body_names, collision_names)
        self.offset = torch.tensor(DEFAULT_POS, device=self.device)
        self.base_kp = torch.tensor(KP, device=self.device)
        self.base_kd = torch.tensor(KD, device=self.device)
        self.limits = torch.tensor(TORQUE_LIMIT, device=self.device)
        self.kp = self.base_kp.repeat(self.num_envs, 1)
        self.kd = self.base_kd.repeat(self.num_envs, 1)
        self.actions = torch.zeros(self.num_envs, 18, device=self.device)
        self.previous_actions = torch.zeros_like(self.actions)
        self.torque = torch.zeros_like(self.actions)
        self.previous_velocity = torch.zeros_like(self.actions)
        delay_steps = cfg.control_delay_s / self.physics_dt
        if abs(delay_steps - round(delay_steps)) > 1e-6 or delay_steps < 0:
            raise ValueError("control_delay_s must be a nonnegative multiple of physics dt")
        self.delay_steps = round(delay_steps)
        self.action_history = torch.zeros(self.num_envs, self.delay_steps + 1, 18, device=self.device)
        self.trajectory_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.synthetic_center = torch.zeros(self.num_envs, 3, device=self.device)
        self.synthetic_rotation = torch.eye(3, device=self.device).repeat(self.num_envs, 1, 1)
        self.past_pos_error = torch.ones(self.num_envs, device=self.device)
        self.past_orn_error = torch.ones(self.num_envs, device=self.device)
        self.pos_sigma, self.orn_sigma = 1.0, 4.0
        self.total_resets = 0
        self.total_timeouts = 0
        self._randomize_setup()
        # Print explicit contract once, including the actual simulator mapping.
        print(f"[UMI] policy joints: {JOINT_NAMES}\n[UMI] PhysX indices: {self.joints}", flush=True)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot
        self.contacts = ContactSensor(self.cfg.contacts)
        self.scene.sensors["contacts"] = self.contacts
        plane = sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0,
                                                           restitution=0.0))
        plane.func("/World/ground", plane)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)

    def _uniform(self, shape, lower, upper, device=None):
        return lower + (upper - lower) * torch.rand(shape, device=device or self.device)

    def _randomize_setup(self):
        """PhysX property setters use CPU buffers; run once at startup."""
        view = self.robot.root_physx_view
        ids = torch.arange(self.num_envs, device="cpu")
        materials = view.get_material_properties().clone()
        if materials.shape[1] != 34:
            raise ValueError(f"Expected original 34 collision shapes for critic, found {materials.shape[1]}")
        masses = view.get_masses().clone()
        coms = view.get_coms().clone()
        inertias = view.get_inertias().clone()
        damping = view.get_dof_dampings().clone()
        friction = view.get_dof_friction_coefficients().clone()
        if self.cfg.randomization:
            # Share 64 material buckets to stay below PhysX's material limit.
            buckets = self._uniform((64,), 0.1, 8.0, "cpu")
            coefficients = buckets[torch.randint(64, materials.shape[:2], device="cpu")]
            materials[..., :2] = coefficients[..., None]
            materials[..., 2] = 0.0
            view.set_material_properties(materials, ids)
            old_mass = masses[:, self.random_bodies].clone()
            masses[:, self.random_bodies] = (
                old_mass + self._uniform(old_mass.shape, -0.25, 0.25, "cpu")
            ).clamp_min(0.01)
            inertias[:, self.random_bodies] *= (masses[:, self.random_bodies] / old_mass)[..., None]
            view.set_masses(masses, ids)
            view.set_inertias(inertias, ids)
            offsets = self._uniform((self.num_envs, 5, 3), -0.1, 0.1, "cpu")
            coms[:, self.random_bodies, :3] += offsets
            view.set_coms(coms, ids)
            damping[:] = self._uniform(damping.shape, 0.01, 0.5, "cpu")
            friction[:] = self._uniform(friction.shape, 0.0, 0.05, "cpu")
        # These are passive PhysX properties; manual controller gains are separate.
        self.robot.write_joint_damping_to_sim(damping.to(self.device))
        self.robot.write_joint_friction_to_sim(friction.to(self.device))
        self.setup_mass = masses[:, self.random_bodies].to(self.device)
        self.com_offset = coms[:, self.random_bodies, :3].to(self.device)
        self.setup_friction = materials[..., 0].to(self.device)
        self.setup_damping = damping[:, self.joints].to(self.device)
        self.setup_joint_friction = friction[:, self.joints].to(self.device)

    def _pre_physics_step(self, actions):
        self.previous_actions.copy_(self.actions)
        self.actions = actions.clamp(-100.0, 100.0).clone()

    def _apply_action(self):
        # A 20 ms delay at 200 Hz is four PHYSICS steps, independent of decimation.
        self.action_history = torch.cat((self.actions[:, None], self.action_history[:, :-1]), 1)
        delayed = self.action_history[:, self.delay_steps]
        q = self.robot.data.joint_pos[:, self.joints]
        qd = self.robot.data.joint_vel[:, self.joints]
        self.previous_velocity.copy_(qd)
        self.torque = (self.kp * (self.offset + 0.25 * delayed - q) - self.kd * qd).clamp(-self.limits, self.limits)
        self.robot.set_joint_effort_target(self.torque, joint_ids=self.joints)

    def _targets(self, offsets):
        times = self.episode_length_buf[:, None] * self.step_dt + torch.tensor(offsets, device=self.device)[None]
        if self.trajectories:
            pos, rot = self.trajectories.query(self.trajectory_ids, times, self.device)
            return pos + self.scene.env_origins[:, None], rot
        phase = times * (2 * torch.pi / self.cfg.episode_length_s)
        displacement = torch.stack((0.08 * phase.sin(), 0.08 * (phase.cos()-1), 0.04 * phase.sin()), -1)
        return self.synthetic_center[:, None] + displacement, self.synthetic_rotation[:, None].expand(-1, len(offsets), -1, -1)

    def _pose_error(self):
        pos, rot = self._targets((0.0,))
        ee_rot = quat_to_matrix(self.robot.data.body_quat_w[:, self.ee])
        return ((pos[:, 0] - self.robot.data.body_pos_w[:, self.ee]).norm(dim=-1),
                rotation_angle(rot[:, 0], ee_rot))

    def _get_observations(self):
        data = self.robot.data
        q, qd = data.joint_pos[:, self.joints], data.joint_vel[:, self.joints]
        gravity, ang_vel = data.projected_gravity_b, data.root_ang_vel_b
        target_pos, target_rot = self._targets(TARGET_TIMES)
        task_obs = relative_pose_observation(target_pos, target_rot, data.body_pos_w[:, self.ee],
                                             quat_to_matrix(data.body_quat_w[:, self.ee]))
        # Match upstream alphabetical ordering of observation/config keys.
        state = [q - self.offset, qd * 0.05, gravity, ang_vel * 0.25]
        if self.cfg.observation_noise:
            state = [x + torch.randn_like(x) * std for x, std in zip(state, (0.01, 0.075, 0.05, 0.05))]
        actor = torch.cat([*(x.clamp(-100, 100) for x in state), task_obs, self.actions], -1)
        setup = [self.setup_damping, self.setup_joint_friction * 10, self.kd * 10, self.kp * 0.1,
                 self.setup_friction, self.com_offset.flatten(1) * 10, self.setup_mass]
        privileged_state = [q - self.offset, qd * 0.05, gravity, ang_vel * 0.25, data.root_lin_vel_b * 2]
        critic = torch.cat([*(x.clamp(-100, 100) for x in setup + privileged_state), task_obs, self.actions], -1)
        if actor.shape[1] != 96 or critic.shape[1] != 225:
            raise RuntimeError(f"Observation contract changed: actor={actor.shape}, critic={critic.shape}")
        return {"policy": actor, "critic": critic}

    def _get_rewards(self):
        data = self.robot.data
        q, qd = data.joint_pos[:, self.joints], data.joint_vel[:, self.joints]
        pos_err, orn_err = self._pose_error()
        smoothing = self.physics_dt * 0.25
        self.past_pos_error.lerp_(pos_err, smoothing)
        self.past_orn_error.lerp_(orn_err, smoothing)
        forces = self.contacts.data.net_forces_w
        vertical = forces[:, self.contact_feet, 2].clamp_min(0)
        distribution = vertical / (vertical.sum(-1, keepdim=True) + 1e-8)
        mass_penalty = torch.where(vertical.sum(-1) > 1e-8, distribution.std(-1).square(), 100.0)
        foot_distance = (data.body_pos_w[:, self.feet, :2] - data.body_pos_w[:, self.hips, :2]).norm(dim=-1)
        lower, upper = data.joint_limits[:, self.joints].unbind(-1)
        mid, span = (lower + upper) * 0.5, upper - lower
        limit_penalty = ((mid - 0.45 * span - q).clamp_min(0) + (q - mid - 0.45 * span).clamp_min(0)).sum(-1)
        terms = {
            "pose": 4.0 * torch.exp(-pos_err.square() / self.pos_sigma - orn_err / self.orn_sigma),
            "action_rate": -0.05 * (self.actions - self.previous_actions).square().sum(-1),
            "joint_acc": -2.5e-7 * ((qd - self.previous_velocity) / self.physics_dt).square().sum(-1),
            "torque": -1e-4 * self.torque.square().sum(-1),
            "joint_limit": -10.0 * limit_penalty,
            "collision": -(forces[:, self.collision_contacts].norm(dim=-1) > 0.1).float().sum(-1),
            "mass_distribution": -mass_penalty,
            "feet_under_hips": -(1 - torch.exp(-foot_distance / 0.5)).sum(-1),
            "aligned_body_ee": -q[:, [12, 16]].square().sum(-1),
            "root_height": -0.5 * (data.root_pos_w[:, 2] - 0.35).square(),
            "orientation": -data.projected_gravity_b[:, :2].square().sum(-1),
        }
        self.extras["log"] = {f"reward/{k}": v.mean() for k, v in terms.items()}
        self.extras["log"].update({"tracking/position_error": pos_err.mean(),
                                   "tracking/orientation_error": orn_err.mean(),
                                   "curriculum/pos_sigma": self.pos_sigma, "curriculum/orn_sigma": self.orn_sigma})
        reward = torch.stack(list(terms.values())).sum(0).clamp_min(0) * self.step_dt
        self._perturb_root()
        return reward

    def _get_dones(self):
        forces = self.contacts.data.net_forces_w[:, self.termination_contacts]
        terminated = (forces.norm(dim=-1) > 1.0).any(-1)
        timed_out = self.episode_length_buf >= self.max_episode_length
        return terminated, timed_out

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        if len(env_ids) == 0:
            return
        self.total_resets += len(env_ids)
        self.total_timeouts += int(self.reset_time_outs[env_ids].sum())
        super()._reset_idx(env_ids)
        root = self.robot.data.default_root_state[env_ids].clone()
        root[:, :3] += self.scene.env_origins[env_ids]
        if self.cfg.randomization:
            root[:, :3] += self._uniform((len(env_ids), 3), -1.0, 1.0) * torch.tensor((0.1, 0.1, 0.0), device=self.device)
            euler = self._uniform((len(env_ids), 3), -1.0, 1.0) * torch.tensor((0.05, 0.05, 0.5), device=self.device)
            root[:, 3:7] = quat_mul(root[:, 3:7], euler_xyz_to_quat(euler))
            self.kp[env_ids] = self.base_kp * self._uniform((len(env_ids), 18), 0.5, 1.5)
            self.kd[env_ids] = self.base_kd * self._uniform((len(env_ids), 18), 0.5, 1.5)
        q = self.robot.data.default_joint_pos[env_ids].clone()
        # Original reset noise is 5% of joint range.
        limits = self.robot.data.joint_limits[env_ids]
        q += 0.05 * torch.randn_like(q) * (limits[..., 1] - limits[..., 0])
        q = q.clamp(limits[..., 0], limits[..., 1])
        self.robot.write_root_state_to_sim(root, env_ids)
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
        self.actions[env_ids] = 0
        self.previous_actions[env_ids] = 0
        self.action_history[env_ids] = 0
        self.previous_velocity[env_ids] = 0
        self.torque[env_ids] = 0
        if self.trajectories:
            self.trajectory_ids[env_ids] = torch.randint(len(self.trajectories.lengths), (len(env_ids),), device=self.device)
        else:
            # Forward kinematics is refreshed by Lab on accessing body poses after state writes.
            self.synthetic_center[env_ids] = self.robot.data.body_pos_w[env_ids, self.ee]
            self.synthetic_rotation[env_ids] = quat_to_matrix(self.robot.data.body_quat_w[env_ids, self.ee])
        for threshold, sigma in POS_CURRICULUM:
            if self.past_pos_error.mean().item() < threshold:
                self.pos_sigma = sigma
        for threshold, sigma in ORN_CURRICULUM:
            if self.past_orn_error.mean().item() < threshold:
                self.orn_sigma = sigma

    def _perturb_root(self):
        # Apply after reward calculation and before automatic resets, as upstream does.
        if self.cfg.randomization:
            root = None
            if self.common_step_counter % round(5.0 / self.step_dt) == 0:
                root = self.robot.data.root_state_w.clone()
                root[:, 7:13] = self._uniform((self.num_envs, 6), -1.0, 1.0)
                root[:, 9] = 0.0
            if self.common_step_counter % round(4.0 / self.step_dt) == 0:
                root = self.robot.data.root_state_w.clone() if root is None else root
                root[:, :3] += torch.randn_like(root[:, :3]) * torch.tensor((0.1, 0.1, 0.01), device=self.device)
                angles = torch.randn(self.num_envs, 3, device=self.device) * torch.tensor((0.01, 0.01, 0.5), device=self.device)
                root[:, 3:7] = quat_mul(root[:, 3:7], euler_xyz_to_quat(angles))
            if root is not None:
                self.robot.write_root_state_to_sim(root)
