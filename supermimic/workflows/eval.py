"""SuperMimic 动作跟踪评估, 独立启动或由训练流程复用评估函数."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import torch

from sim_platform.assets.train_data.motion import MotionLib
from sim_platform.assets.train_data.kinematics import G1Kinematics
from sim_platform.sim.mdp import TrackingMDP, TrackingMDPConfig
from supermimic.agent.algorithm.supermimic_repro import Policy, SuperMimicConfig
from supermimic.utils.checkpoint_utils import find_latest_checkpoint, find_latest_run_dir, load_run_config


##############################################################################
#                         模型和评估配置                                      #
##############################################################################


def load_policy(checkpoint: str | Path, device: str | torch.device) -> tuple[Policy, SuperMimicConfig]:
    """从算法状态加载策略及归一化统计; 不加载世界模型或优化器到设备."""

    saved = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    # 训练入口可以直接保存 agent.state_dict(), 或放在 agent 字段中.
    if isinstance(saved, dict) and "agent" in saved:
        saved = saved["agent"]
    if not isinstance(saved, dict) or not isinstance(saved.get("config"), dict) or "policy" not in saved:
        raise ValueError("checkpoint 必须包含算法 config 和 policy 状态")
    parameters = dict(saved["config"])
    for name in ("wm_hidden_dims", "policy_hidden_dims", "critic_hidden_dims", "policy_betas"):
        if name in parameters:
            parameters[name] = tuple(parameters[name])
    config = SuperMimicConfig(**parameters)
    policy = Policy(config).to(device)
    policy.load_state_dict(saved["policy"], strict=True)
    policy.eval()
    return policy, config


def resolve_checkpoint(checkpoint: str, run_dir: str, root: str, task: str) -> Path:
    """优先使用指定 checkpoint 或实验目录, 均为空时查找最新实验."""

    if checkpoint and run_dir:
        raise ValueError("CHECKPOINT_PATH 和 RUN_DIR 只能填写一个")
    if checkpoint:
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint 文件不存在: {path}")
        return path
    directory = Path(run_dir).expanduser().resolve() if run_dir else find_latest_run_dir(root, task, "supermimic")
    return find_latest_checkpoint(directory)


##############################################################################
#                         统一评估                                            #
##############################################################################


@torch.no_grad()
def evaluate_policy(env, policy: Policy, mdp: TrackingMDP, *, verbose: bool = True, print_actions: bool = False) -> dict:
    """逐段评估全部动作; 环境由调用方创建和关闭, 不更新策略或训练状态.

    每段从首帧开始, 无动作和观测噪声, 跟踪失败时结束; 分母始终为完整片段
    最大回报, 与 vibed 的逐动作平均归一化回报一致. 环境必须是独立的单环境.
    """

    motion = mdp.motion
    if env.num_envs != 1:
        raise ValueError("逐动作评估需要 num_envs=1 的独立环境")
    if env.device != motion.device or next(policy.parameters()).device != motion.device:
        raise ValueError("环境、动作库和策略必须使用相同设备")
    if policy.config.goal_frames != mdp.config.goal_frames:
        raise ValueError("策略和 MDP 的未来参考帧数必须一致")
    if abs(env.config.control_dt - mdp.config.control_dt) > 1e-10:
        raise ValueError("环境和 MDP 的控制周期必须一致")

    training = policy.training
    policy.eval()
    results = []
    try:
        for index, name in enumerate(motion.names):
            clip = torch.tensor([index], device=motion.device, dtype=torch.long)
            frame = torch.zeros(1, device=motion.device, dtype=torch.long)
            state = env.reset(clip.new_zeros(1), motion.get_reference(clip, frame, include_body=False))
            previous = torch.zeros((1, 29), device=motion.device, dtype=state["joint_pos"].dtype)
            observation = mdp.observation(state, previous, clip, frame)
            history = observation.unsqueeze(1).repeat(1, policy.config.history_length, 1)
            total, steps, failed = 0.0, 0, False
            length = int(motion.clip_lengths[index].item())

            # 环境每步推进一次, 最后一帧不再生成下一个动作.
            for _ in range(length - 1):
                goal = mdp.goal_features(clip, frame)
                action = policy.predict_actions(history, goal)[:, 0, :]
                if verbose and print_actions:
                    print(f"动作 {name} 第 {steps + 1} 步: {action[0].tolist()}", flush=True)
                state = env.step(action)
                frame = frame + 1
                reward = mdp.reward(state, clip, frame, action, previous)
                total += float(reward.item())
                steps += 1
                failed = bool(mdp.termination(state, clip, frame).item())
                if failed:
                    break
                previous = action
                observation = mdp.observation(state, previous, clip, frame)
                history = torch.cat((history[:, 1:], observation.unsqueeze(1)), dim=1)

            normalized = total / ((length - 1) * mdp.max_step_reward)
            result = {"name": name, "return": total, "normalized_return": normalized, "steps": steps, "failed": failed}
            results.append(result)
            if verbose:
                print(f"测评 {index + 1}/{motion.num_clips} | {name} | 失败: {str(failed).lower()} | 回报: {total:.4f} | 归一化回报: {normalized:.4f}", flush=True)
    finally:
        policy.train(training)

    mean = sum(item["normalized_return"] for item in results) / len(results)
    failed_count = sum(item["failed"] for item in results)
    if verbose:
        print(f"测评平均归一化回报: {mean:.4f} | 失败动作: {failed_count}/{len(results)}", flush=True)
    return {"normalized_return": mean, "failed_count": failed_count, "num_clips": len(results), "per_clip": results}


##############################################################################
#                         独立评估入口                                        #
##############################################################################


def main() -> None:
    """读取脚本配置和训练参数, 在独立仿真上下文中评估保存的策略."""

    parser = argparse.ArgumentParser(description="评估 SuperMimic checkpoint")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--checkpoint-root", default="checkpoints/supermimic")
    parser.add_argument("--task", default="tracking")
    parser.add_argument("--motion-dir", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--show-viewer", choices=("true", "false"), default="false")
    parser.add_argument("--print-actions", choices=("true", "false"), default="false")
    parser.add_argument("--result-file", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()

    checkpoint = resolve_checkpoint(args.checkpoint, args.run_dir, args.checkpoint_root, args.task)
    policy, algorithm_config = load_policy(checkpoint, args.device)
    # 训练配置存在时读取 MDP 设置; 未提供时使用当前 MDP 默认配置.
    config_path = checkpoint.parent / "run_config.json"
    run_config = load_run_config(checkpoint.parent) if config_path.is_file() else {}
    if "mdp" not in run_config:
        warnings.warn("实验配置未包含 mdp 参数, 评估将使用当前默认任务配置; 结果可能与训练时不一致")
    mdp_values = dict(run_config.get("mdp", {}))
    for name in ("tracked_bodies", "tracking_stds", "tracking_weights", "observation_noise"):
        if name in mdp_values:
            mdp_values[name] = tuple(mdp_values[name])
    if "control_dt" in mdp_values and mdp_values["control_dt"] != algorithm_config.control_dt:
        raise ValueError("保存的 MDP 控制周期与算法配置不一致")
    if "goal_frames" in mdp_values and mdp_values["goal_frames"] != algorithm_config.goal_frames:
        raise ValueError("保存的 MDP 目标帧数与策略配置不一致")
    mdp_values.update(control_dt=algorithm_config.control_dt, goal_frames=algorithm_config.goal_frames)
    mdp_config = TrackingMDPConfig(**mdp_values)

    # Isaac Lab 只在独立入口中导入, 便于训练流程复用 evaluate_policy.
    from isaaclab.app import launch_simulation
    from sim_platform.assets.robots.g1.robot import G1_JOINT_NAMES, MJCF_PATH, default_joint_pos_list
    from sim_platform.sim.env import RobotSimEnv, RobotSimEnvConfig, make_sim_config

    kin = G1Kinematics(MJCF_PATH, G1_JOINT_NAMES, mdp_config.tracked_bodies, device=args.device)
    motion = MotionLib(args.motion_dir or None, control_dt=algorithm_config.control_dt, device=args.device, kinematics=kin)
    mdp = TrackingMDP(kin, motion, torch.tensor(default_joint_pos_list(), device=args.device), mdp_config)
    env_config = RobotSimEnvConfig(num_envs=1, device=args.device, control_dt=algorithm_config.control_dt, show_viewer=args.show_viewer == "true")
    sim_config = make_sim_config(env_config)
    if not args.result_file:
        print(f"评估 checkpoint: {checkpoint}", flush=True)
    with launch_simulation(sim_config):
        env = RobotSimEnv(env_config, sim_cfg=sim_config)
        try:
            result = evaluate_policy(env, policy, mdp, verbose=not bool(args.result_file), print_actions=args.print_actions == "true")
            if args.result_file:
                Path(args.result_file).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        finally:
            env.close()


if __name__ == "__main__":
    main()
