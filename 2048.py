import numpy as np
from td3_torch import Agent
# import mujoco
import torch
from game_engine import Game2048


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If you're using GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set the seed
set_seed(123)
if __name__ == '__main__':
    game = Game2048()
    initial_state = game.reset()
    action_space = 1
    agent = Agent(alpha=3e-3, beta=3e-3,
        input_dims=initial_state.shape, tau=0.005, batch_size=100, layer1_size=1024, layer2_size=512,layer3_size=256,
        n_actions=action_space)
    score_history = []
    while True:
        
        time_step = 0
        score = 0
        done = False
        observation = game.reset()

        # Run simulation for max_timesteps or until done
        while not done:
            action = agent.choose_action(observation)
            next_observation, reward, done = game.step(action)
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

        
