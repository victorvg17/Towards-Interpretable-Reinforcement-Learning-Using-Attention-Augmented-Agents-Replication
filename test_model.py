import gym
import torch
import argparse
import numpy as np
import attention
from main import Policy

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--num_episodes", type=int, default=1_000)
    parser.add_argument("--num_repeat_action", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=10_000)
    parser.add_argument("--reward_threshold", type=int, default=1_000)
    parser.add_argument("--render", action="store_true", help="render the environment")
    parser.add_argument(
        "--debug_print",
        action="store_true",
        help="debug flag for printing verbose outputs",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=1,
        metavar="N",
        help="interval between training status logs (default: 10)",
    )
    config = parser.parse_args()

    env = gym.make("Seaquest-v0")
    num_actions = env.action_space.n
    agent = attention.Agent(num_actions=num_actions)
    agent.load_state_dict(torch.load(config.model_path, map_location=device))
    agent.eval()
    # print agent module parameters details
    if config.debug_print:
        print("Agent's state_dict: ")
        for param_tensor in agent.state_dict():
            print(param_tensor, "\t", agent.state_dict()[param_tensor].size())
        params_size = sum([p.numel() for p in agent.parameters() if p.requires_grad])
        print(f"agent total parameters size: \t {params_size}")

    policy = Policy(agent=agent).to(device)
    running_reward = 10.0
    for i_episode in range(config.num_episodes):
        observation = env.reset()
        ep_reward = 0
        for t in range(config.max_steps):
            action = policy(observation)
            reward = 0.0
            for _ in range(config.num_repeat_action):
                if config.render:
                    env.render()
                observation, _reward, done, _ = env.step(action)
                reward += _reward
                if done:
                    break
            ep_reward += reward
            if t % config.log_interval == 0:
                print(f"timestep {t} \tReward: {ep_reward:.2f}")
            if done:
                running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
                if i_episode % config.log_interval == 0:
                    print(
                        f"Episode {i_episode} \tLast reward: {ep_reward:.2f}\tAverage reward: {running_reward:.2f}"
                    )
                if running_reward > config.reward_threshold:
                    print(
                        f"Solved! Running reward is now {running_reward} and "
                        f"the last episode runs to {t} time steps!"
                    )
                break
