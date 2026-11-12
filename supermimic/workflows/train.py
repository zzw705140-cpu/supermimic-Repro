"""SuperMimic 训练流程: 真实数据采集、模型更新和预测轨迹调度."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import warnings

import torch

from sim_platform.assets.train_data.kinematics import G1Kinematics
from sim_platform.assets.train_data.motion import MotionLib
from sim_platform.sim.mdp import TrackingMDP, TrackingMDPConfig
from sim_platform.wrappers.randomization import perturb_reset_state, sample_joint_offsets, sample_physical_parameters
from supermimic.agent.algorithm.supermimic_repro import Critic, Policy, SuperMimicAgent, SuperMimicConfig, WorldModel
from supermimic.data.imagined_rollout.imagined_rollout import ImaginedRollout
from supermimic.data.replay_buffer.replay_buffer import ReplayBuffer
from supermimic.utils.checkpoint_utils import create_run_dir, save_run_config
from supermimic.utils.gpu_memory import read_gpu_memory, reset_gpu_memory_peak


##############################################################################
#                         并行任务与真实数据采集                              #
##############################################################################


class Collector:
    """维护并行环境的参考帧、历史和回合状态; 环境本身只执行步进与重置."""

    def __init__(self, env, mdp, agent, replay, *, action_noise, max_episode_steps):
        self.env, self.mdp, self.agent, self.replay = env, mdp, agent, replay
        self.motion, self.device, self.n = mdp.motion, mdp.device, env.num_envs
        self.action_noise, self.max_episode_steps = action_noise, max_episode_steps
        self.clip = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.frame = torch.zeros_like(self.clip)
        self.steps = torch.zeros_like(self.clip)
        self.previous = torch.zeros(self.n, 29, device=self.device)
        self.context = torch.zeros(self.n, 33, device=self.device)
        # 环境创建后采样并写入摩擦与质心, 只记录实际已提交的参数.
        friction, com_offset = sample_physical_parameters(self.n, self.device)
        env.set_physical_parameters(friction, com_offset)
        self.context[:, :1], self.context[:, 30:] = friction, com_offset
        self.history = torch.zeros(self.n, agent.config.history_length, 99, device=self.device)
        self.fresh = torch.ones(self.n, dtype=torch.bool, device=self.device)
        # 约一秒一段, 按回报调整起点采样权重; 同时保留均匀采样.
        bin_size = max(1, round(1.0 / mdp.config.control_dt))
        clip_bins = ((self.motion.clip_lengths - 2) // bin_size + 1).long()
        self.bin_clip = torch.repeat_interleave(torch.arange(self.motion.num_clips, device=self.device), clip_bins)
        offsets = torch.cumsum(clip_bins, 0) - clip_bins
        self.bin_start = (torch.arange(self.bin_clip.numel(), device=self.device) - offsets[self.bin_clip]) * bin_size
        self.bin_end = torch.minimum(self.bin_start + bin_size, self.motion.clip_lengths[self.bin_clip] - 1)
        self.bin_value = torch.zeros(self.bin_clip.numel(), device=self.device)
        self.bin_weight = torch.full_like(self.bin_value, 1.0 / self.bin_value.numel())
        self.start_bin = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.episode_return = torch.zeros(self.n, device=self.device)
        self.discounted_return = torch.zeros(self.n, device=self.device)
        self.reset(torch.arange(self.n, device=self.device))

    @torch.no_grad()
    def reset(self, ids):
        """上层选择参考帧并重置指定环境, 不影响其他环境的任务状态."""
        if ids.numel() == 0:
            return
        # 50% 均匀, 40% 偏向低回报, 10% 使用平滑采样权重.
        max_return = self.mdp.max_step_reward / (1 - self.agent.config.gamma)
        soft = torch.softmax(-25.0 * self.bin_value / max_return, dim=0)
        self.bin_weight.mul_(0.999).add_(soft, alpha=0.001)
        weights = 0.5 / self.bin_value.numel() + 0.4 * soft + 0.1 * self.bin_weight / self.bin_weight.sum()
        bins = torch.multinomial(weights, ids.numel(), replacement=True)
        clip = self.bin_clip[bins]
        frame = self.bin_start[bins] + (torch.rand(ids.numel(), device=self.device) * (self.bin_end[bins] - self.bin_start[bins])).long()
        self.clip[ids], self.frame[ids], self.steps[ids] = clip, frame, 0
        self.start_bin[ids] = bins
        self.episode_return[ids] = 0
        self.discounted_return[ids] = 0
        self.previous[ids] = 0
        self.fresh[ids] = True
        # 每个回合采样目标偏移和初始状态扰动, 其他环境保持原值.
        offsets = sample_joint_offsets(ids.numel(), self.device)
        reference = self.motion.get_reference(clip, frame, include_body=False)
        initial = perturb_reset_state(reference, self.mdp.kinematics.q_lo, self.mdp.kinematics.q_hi)
        self.env.set_joint_offsets(ids, offsets)
        self.env.reset(ids, initial)
        self.context[ids, 1:30] = offsets

    @torch.no_grad()
    def collect(self, count):
        """批量记录真实交互, 重置前写入终止状态并更新模型的真实数据统计."""
        total_reward = 0.0
        for _ in range(count):
            # 仿真状态可能共享底层缓存, 步进前仅保存这一份必要快照.
            state = {key: value.clone() for key, value in self.env.state().items()}
            observation = self.mdp.add_observation_noise(self.mdp.observation(state, self.previous, self.clip, self.frame, self.context))
            shifted = torch.cat((self.history[:, 1:], observation.unsqueeze(1)), dim=1)
            self.history = torch.where(self.fresh[:, None, None], observation.unsqueeze(1).expand_as(shifted), shifted)
            self.fresh.fill_(False)
            goal = self.mdp.goal_features(self.clip, self.frame)
            action = self.agent.policy.predict_actions(self.history, goal)[:, 0]
            action = (action + self.action_noise * torch.randn_like(action)).clamp(-20, 20)
            next_state = self.env.step(action)
            next_frame = self.frame + 1
            reward = self.mdp.reward(next_state, self.clip, next_frame, action, self.previous)
            terminated = self.mdp.termination(next_state, self.clip, next_frame)
            truncated = (next_frame >= self.motion.clip_lengths[self.clip] - 1) | (self.steps + 1 >= self.max_episode_steps)
            transition = {"state": state, "next_state": next_state, "action": action, "previous_action": self.previous,
                          "context": self.context, "clip_id": self.clip, "frame_id": self.frame,
                          "episode_step": self.steps, "terminated": terminated, "truncated": truncated}
            self.agent.update_statistics(transition, observation, goal)
            self.replay.add(move_data(transition, self.replay.device))
            total_reward += float(reward.mean())
            self.episode_return += reward
            self.discounted_return += self.agent.config.gamma ** self.steps.float() * reward
            self.frame = next_frame
            self.steps = self.steps + 1
            self.previous = action
            ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
            if ids.numel():
                length = self.steps[ids].float()
                bootstrap = self.agent.config.gamma ** length * self.episode_return[ids] / length / (1 - self.agent.config.gamma)
                estimate = self.discounted_return[ids] + torch.where(terminated[ids], 0.0, bootstrap)
                bins = self.start_bin[ids]
                # 同段有多个环境同时结束时, 先取均值再更新, 避免写入次序影响结果.
                sums, counts = torch.zeros_like(self.bin_value), torch.zeros_like(self.bin_value)
                sums.scatter_add_(0, bins, estimate)
                counts.scatter_add_(0, bins, torch.ones_like(estimate))
                selected = counts > 0
                self.bin_value[selected] = 0.8 * self.bin_value[selected] + 0.2 * sums[selected] / counts[selected]
                self.reset(ids)
        return total_reward / max(count, 1)


def move_data(value, device):
    """递归将经验池输入或采样批次放到对应设备."""
    if isinstance(value, dict):
        return {key: move_data(item, device) for key, item in value.items()}
    return value.to(device)


def select_data(value, indices):
    """按批量维筛选嵌套状态和参考信息."""
    if isinstance(value, dict):
        return {key: select_data(item, indices) for key, item in value.items()}
    return value[indices]


def cat_data(items):
    """合并不同次采样得到的起点或历史."""
    if isinstance(items[0], dict):
        return {key: cat_data([item[key] for item in items]) for key in items[0]}
    return torch.cat(items, dim=0)


def sample_starts(replay, motion, batch_size, history_length, horizon, device):
    """筛选参考片段足够长的起点; 有限次数失败时明确提示, 不无限循环."""
    starts, histories, remaining = [], [], batch_size
    for _ in range(128):
        start, history = replay.sample_starts(max(remaining * 2, 64), history_length)
        start, history = move_data(start, device), move_data(history, device)
        valid = start["frame_id"] + horizon < motion.clip_lengths[start["clip_id"]]
        ids = valid.nonzero(as_tuple=False).squeeze(-1)[:remaining]
        if ids.numel():
            starts.append(select_data(start, ids))
            histories.append(select_data(history, ids))
            remaining -= ids.numel()
        if not remaining:
            return cat_data(starts), cat_data(histories)
    raise RuntimeError("经验池中满足预测长度的起点不足; 请增加预采集步数或缩短预测长度")


##############################################################################
#                         世界模型与短期预测轨迹                              #
##############################################################################


def train_imagination(agent, replay, batch_size):
    """采样真实起点, 保留预测计算图更新策略, 再固定目标更新 Critic."""
    cfg, mdp = agent.config, agent.mdp
    start, past = sample_starts(replay, mdp.motion, batch_size, cfg.history_length, cfg.imagination_horizon, agent.device)
    history = mdp.observation(past["state"], past["previous_action"], past["clip_id"], past["frame_id"], past["context"])
    history = mdp.add_observation_noise(history)
    state, previous, context, clip, frame = (start[key] for key in ("state", "previous_action", "context", "clip_id", "frame_id"))
    rollout = ImaginedRollout(state, previous, history, context, clip, frame, horizon=cfg.imagination_horizon,
                              history_length=cfg.history_length, goal_frames=cfg.goal_frames)
    try:
        with agent.policy_phase():
            for _ in range(cfg.imagination_horizon):
                goal = mdp.goal_features(clip, frame)
                action = agent.policy.predict_actions(history, goal)[:, 0]
                state = agent.world_model.step(state, action, context)
                frame = frame + 1
                reward = mdp.reward(state, clip, frame, action, previous)
                termination = mdp.soft_termination(state, clip, frame)
                rollout.append(state, action, reward, termination, frame)
                observation = mdp.add_observation_noise(mdp.observation(state, action, clip, frame, context))
                history = torch.cat((history[:, 1:], observation.unsqueeze(1)), dim=1)
                previous = action
            batch = rollout.as_dict()
            policy_metrics = agent.update_policy(batch)
        critic_batch = agent.prepare_critic_batch(rollout.detached())
        critic_metrics = agent.update_critic(critic_batch)
        return policy_metrics, critic_metrics
    finally:
        rollout.clear()


##############################################################################
#                         保存、评估与日志                                     #
##############################################################################


def save_checkpoint(run_dir, agent, iteration, keep_last=2):
    """先原子保存, 成功后仅保留本实验最近 keep_last 个模型文件."""
    if isinstance(keep_last, bool) or not isinstance(keep_last, int) or keep_last <= 0:
        raise ValueError("keep_last 必须为正整数")
    target = run_dir / f"epoch_{iteration:06d}.ckpt"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=run_dir, prefix=".checkpoint_", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(agent.state_dict(), stream)
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    checkpoints = sorted(run_dir.glob("epoch_*.ckpt"), key=lambda path: int(path.stem.split("_")[-1]))
    for old in checkpoints[:-keep_last]:
        old.unlink()
    return target


def evaluate_checkpoint(checkpoint, motion_dir, device, show_viewer):
    """独立进程复用 eval.py 的评估函数, 避免创建第二个仿真上下文."""
    with tempfile.TemporaryDirectory(prefix="supermimic_eval_") as directory:
        result = Path(directory) / "result.json"
        command = [sys.executable, "-m", "supermimic.workflows.eval", "--checkpoint", str(checkpoint),
                   "--device", device, "--show-viewer", str(show_viewer).lower(), "--motion-dir", motion_dir,
                   "--result-file", str(result)]
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if completed.returncode:
            raise RuntimeError(f"评估进程失败:\n{completed.stdout}")
        return json.loads(result.read_text(encoding="utf-8"))


def progress(done, total):
    width = 20
    filled = width * done // total
    print(f"\r训练 {done}/{total} [{'#' * filled}{' ' * (width - filled)}]", end="", flush=True)


def make_logger(args, run_dir, config):
    """W&B 使用本机登录信息; 初始化失败时警告并继续本地训练."""
    if not args.wandb:
        return None
    try:
        from supermimic.utils.wandb import make_wandb_logger
        logger = make_wandb_logger(project=args.wandb_project, variant=config, wandb_output_dir=run_dir, run_name=run_dir.name)
        if logger.run.url:
            print(f"W&B 训练曲线: {logger.run.url}", flush=True)
        return logger
    except Exception as error:
        warnings.warn(f"W&B 初始化失败, 继续训练: {error}")
        return None


##############################################################################
#                         启动与训练循环                                      #
##############################################################################


def main():
    parser = argparse.ArgumentParser(description="SuperMimic 训练入口")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--replay-device", default="cuda:0")
    parser.add_argument("--eval-device", default="cpu")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=12700)
    parser.add_argument("--replay-steps", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=64)
    parser.add_argument("--utd", type=float, default=0.4)
    parser.add_argument("--wm-updates", type=int, default=10)
    parser.add_argument("--wm-batch", type=int, default=2048)
    parser.add_argument("--policy-batch", type=int, default=4096)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--action-noise", type=float, default=0.1)
    parser.add_argument("--control-dt", type=float, default=0.02)
    parser.add_argument("--wm-hidden", default="2048,2048,2048")
    parser.add_argument("--policy-hidden", default="2048,1024,512,256")
    parser.add_argument("--critic-hidden", default="256,128")
    parser.add_argument("--history-length", type=int, default=10)
    parser.add_argument("--goal-frames", type=int, default=10)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--wm-window", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--critic-epochs", type=int, default=5)
    parser.add_argument("--critic-minibatches", type=int, default=4)
    parser.add_argument("--lr-decay-iterations", type=int, default=250000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-root", default="checkpoints/supermimic")
    parser.add_argument("--task", default="tracking")
    parser.add_argument("--motion-dir", default="")
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    parser.add_argument("--checkpoint-keep", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--memory-every", type=int, default=100)
    parser.add_argument("--memory-limit", type=float, default=0.9, help="整卡占用阈值, 0 表示只记录")
    parser.add_argument("--show-viewer", action="store_true")
    parser.add_argument("--eval-viewer", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="supermimic")
    args = parser.parse_args()
    for name in ("num_envs", "iterations", "replay_steps", "warmup_steps", "wm_updates", "wm_batch", "policy_batch", "max_episode_steps", "checkpoint_every", "checkpoint_keep", "eval_every", "log_every", "memory_every"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} 必须为正整数")
    if not math.isfinite(args.utd) or args.utd <= 0 or not 0 <= args.memory_limit <= 1 or args.action_noise < 0:
        parser.error("utd 必须为正, memory_limit 必须在 [0,1], action_noise 不能为负")
    torch.manual_seed(args.seed)
    try:
        hidden = lambda value: tuple(int(part) for part in value.split(","))
        algorithm = SuperMimicConfig(control_dt=args.control_dt, wm_hidden_dims=hidden(args.wm_hidden),
            policy_hidden_dims=hidden(args.policy_hidden), critic_hidden_dims=hidden(args.critic_hidden),
            history_length=args.history_length, goal_frames=args.goal_frames, latent_dim=args.latent_dim,
            wm_window=args.wm_window, imagination_horizon=args.horizon, learning_rate=args.learning_rate,
            gamma=args.gamma, gae_lambda=args.gae_lambda, critic_epochs=args.critic_epochs,
            critic_minibatches=args.critic_minibatches, lr_decay_iterations=args.lr_decay_iterations)
    except ValueError as error:
        parser.error(str(error))
    if args.replay_steps < algorithm.wm_window or args.warmup_steps < algorithm.wm_window:
        parser.error("经验池容量与预采集步数必须至少等于世界模型窗口长度")
    mdp_config = TrackingMDPConfig(control_dt=algorithm.control_dt, goal_frames=algorithm.goal_frames)
    config = {"workflow": vars(args), "algorithm": asdict(algorithm), "mdp": asdict(mdp_config)}
    run_dir = create_run_dir(args.checkpoint_root, args.task, "supermimic")
    save_run_config(run_dir, config)
    logger = make_logger(args, run_dir, config)
    from isaaclab.app import launch_simulation
    from sim_platform.assets.robots.g1.robot import G1_JOINT_NAMES, MJCF_PATH, default_joint_pos_list
    from sim_platform.sim.env import RobotSimEnv, RobotSimEnvConfig, make_sim_config
    env = None
    try:
        kin = G1Kinematics(MJCF_PATH, G1_JOINT_NAMES, mdp_config.tracked_bodies, device=args.device)
        motion = MotionLib(args.motion_dir or None, control_dt=args.control_dt, device=args.device, kinematics=kin)
        if not bool((motion.clip_lengths > algorithm.imagination_horizon).any()):
            raise ValueError("动作库中没有足够长的片段供世界模型预测展开")
        mdp = TrackingMDP(kin, motion, torch.tensor(default_joint_pos_list(), device=args.device), mdp_config)
        agent = SuperMimicAgent(WorldModel(algorithm).to(args.device), Policy(algorithm).to(args.device), Critic(mdp.critic_dim, algorithm).to(args.device), mdp, algorithm)
        replay = ReplayBuffer(args.replay_steps, args.num_envs, args.replay_device)
        env_config = RobotSimEnvConfig(num_envs=args.num_envs, device=args.device, control_dt=args.control_dt, show_viewer=args.show_viewer)
        sim_config = make_sim_config(env_config)
        with launch_simulation(sim_config):
            try:
                env = RobotSimEnv(env_config, sim_cfg=sim_config)
                collector = Collector(env, mdp, agent, replay, action_noise=args.action_noise, max_episode_steps=args.max_episode_steps)
                collector.collect(args.warmup_steps)
                is_cuda = torch.device(args.device).type == "cuda"
                if is_cuda:
                    reset_gpu_memory_peak(args.device)
                accumulator, eval_count, latest_memory = 0.0, 0, {}
                progress(0, args.iterations)
                for iteration in range(1, args.iterations + 1):
                    agent.set_iteration(iteration - 1)
                    accumulator += 1.0 / args.utd
                    steps = int(accumulator)
                    accumulator -= steps
                    reward = collector.collect(steps)
                    for _ in range(args.wm_updates):
                        windows, mask = replay.sample_windows(args.wm_batch, algorithm.wm_window)
                        wm = agent.update_world_model(move_data(windows, agent.device), mask.to(agent.device))
                    policy, critic = train_imagination(agent, replay, args.policy_batch)
                    progress(iteration, args.iterations)
                    metrics = {"train/reward": reward, "loss/world_model": float(wm["loss"]), "loss/policy": float(policy["loss"]), "loss/critic": float(critic["loss"]), "train/learning_rate": agent.policy_optimizer.param_groups[0]["lr"]}
                    if is_cuda and iteration % args.memory_every == 0:
                        memory = read_gpu_memory(args.device)
                        latest_memory = {f"gpu/{key}": value for key, value in memory.items()}
                        reset_gpu_memory_peak(args.device)
                        if args.memory_limit and memory["used_fraction"] >= args.memory_limit:
                            print("\n显存占用达到设定阈值, 保存模型后结束训练。", flush=True)
                            save_checkpoint(run_dir, agent, iteration, keep_last=args.checkpoint_keep)
                            break
                    checkpoint = None
                    if iteration % args.checkpoint_every == 0 or iteration == args.iterations:
                        checkpoint = save_checkpoint(run_dir, agent, iteration, keep_last=args.checkpoint_keep)
                    if iteration % args.eval_every == 0 or iteration == args.iterations:
                        eval_count += 1
                        print(f"\n第 {eval_count} 次测评", flush=True)
                        # 仅评估时用临时策略快照, 不增加持久 checkpoint 或影响保留数量.
                        with tempfile.TemporaryDirectory(prefix=".evaluation_", dir=run_dir) as directory:
                            if checkpoint is None:
                                checkpoint = Path(directory) / "policy.ckpt"
                                torch.save({"config": asdict(agent.config), "policy": agent.policy.state_dict()}, checkpoint)
                                save_run_config(Path(directory), config)
                            result = evaluate_checkpoint(checkpoint, args.motion_dir, args.eval_device, args.eval_viewer)
                        print(f"第 {eval_count} 次测评 | 归一化回报: {result['normalized_return']:.4f}", flush=True)
                        metrics.update({"eval/normalized_return": result["normalized_return"], "eval/failed_count": result["failed_count"]})
                        progress(iteration, args.iterations)
                    if logger is not None and (iteration % args.log_every == 0 or "eval/normalized_return" in metrics):
                        try:
                            logger.log({**metrics, **latest_memory}, step=iteration)
                        except Exception as error:
                            warnings.warn(f"W&B 记录失败: {error}")
                print(flush=True)
            finally:
                if env is not None:
                    env.close()
    finally:
        if logger is not None:
            try:
                logger.finish()
            except Exception as error:
                warnings.warn(f"W&B 结束失败: {error}")


if __name__ == "__main__":
    main()
