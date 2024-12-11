import gym
import numpy as np
from td3_torch import Agent
import mujoco
import torch


def simulate(m,d,action,time_step):
    d.ctrl[0] = action[0]
    #print(action[0])
    mujoco.mj_step(m,d)
    done = False
    if (d.qpos[1] < -0.24 or d.qpos[1] > 0.24) or (d.qpos[2]+d.qpos[1] < -0.24 or d.qpos[2]+d.qpos[1] > 0.24):
        done = True
    penalty = 0
    x_tip = np.cos(d.qpos[1]) + np.cos(d.qpos[2])
    y_tip = np.sin(d.qpos[1]) + np.sin(d.qpos[2])
    #print(x_tip, y_tip)
    penalty += 0.5 * y_tip**2 + (x_tip - 2) ** 2
    #print(penalty)
    vel_pen = 0.05 * d.qvel[0]**2
    new_pos = [d.qpos[0], np.cos(d.qpos[1]), np.sin(d.qpos[1]), np.cos(d.qpos[2]), np.sin(d.qpos[2])]
    next_state = np.concatenate([new_pos, d.qvel])
    reward = 1 - penalty - vel_pen
    # print(reward)
    time_step = time_step + 1
    if time_step > 1000:
        done = True
        reward += 500

    return next_state, reward, done, time_step

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If you're using GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set the seed
set_seed(99)
if __name__ == '__main__':
    filename="double_cartpole.xml"
    m = mujoco.MjModel.from_xml_path(filename)
    d = mujoco.MjData(m)
    max_timesteps = 1000
    initial_state = np.array([0,0,0,0,0,0,0,0])
    action_space = 1
    agent = Agent(alpha=3e-3, beta=3e-3,
        input_dims=initial_state.shape, tau=0.005, batch_size=100, layer1_size=512, layer2_size=256,layer3_size=256,
        n_actions=action_space)
    score_history = []
    while True:
        d.qpos = np.random.uniform(low=-0.01, high=0.01, size=d.qpos.shape)
        d.qvel[0] = 0
        d.qvel[1] = 0
        d.qvel[2] = 0
        observation = [d.qpos[0],np.cos(d.qpos[1]),np.sin(d.qpos[1]),np.cos(d.qpos[2]),np.sin(d.qpos[2]),0,0,0]

        # d.qpos[:] = 0.01
        # d.qvel[:] = 0
        # # d.qpos[3] = 1
        # # d.qpos[2] = 0.01
        # observation = np.concatenate((d.qpos, d.qvel))
        time_step = 0
        score = 0
        done = False

        # Run simulation for max_timesteps or until done
        while not done and time_step < max_timesteps:
            action = agent.choose_action(observation)
            next_observation, reward, done, time_step = simulate(m, d, action, time_step)
            score += reward

            # Store transition in replay buffer
            # print(next_observation)
            agent.remember(observation, action, reward, next_observation, done)

            observation = next_observation

            # Periodically update the policy
            agent.learn()
        
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        print("average Score: " , avg_score)
        print("most recent score: ", score)

    # env = gym.make('BipedalWalker-v3')
    # agent = Agent(alpha=0.001, beta=0.001,
    #         input_dims=env.observation_space.shape, tau=0.005,
    #         batch_size=100, layer1_size=400, layer2_size=300,
    #         n_actions=env.action_space.shape[0])
    # n_games = 3000
    # # filename = 'plots/' + 'LunarLanderContinuous_' + str(n_games) + '_games.png'

    # # best_score = env.reward_range[0]
    # best_score = -np.inf
    # score_history = []

    # # agent.load_models()

    # for i in range(n_games):
    #     observation, _ = env.reset()
    #     done = False
    #     score = 0
    #     while not done:
    #         action = agent.choose_action(observation)

    #         observation_, reward, terminated, truncated, info = env.step(action)

    #         # Combine `terminated` and `truncated` to get `done`
    #         done = terminated or truncated
            
    #         agent.remember(observation, action, reward, observation_, done)
    #         agent.learn()
    #         score += reward
    #         observation = observation_
    #     score_history.append(score)
    #     avg_score = np.mean(score_history[-100:])

    #     # if avg_score > best_score:
    #     #     best_score = avg_score
    #     #     agent.save_models()

    #     print('episode ', i, 'score %.1f' % score,
    #             'average score %.1f' % avg_score)

    # # x = [i+1 for i in range(n_games)]
    # # plot_learning_curve(x, score_history, filename)

        
