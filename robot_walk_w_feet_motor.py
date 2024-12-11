import gym
import numpy as np
from td3_torch import Agent
import mujoco
import torch
import mujoco.viewer
import concurrent.futures
import time

def quaternion_angle_deviation(q1, q2):
    """Calculate the angle (in radians) between two quaternions."""
    dot_product = np.dot(q1, q2)
    dot_product = np.clip(dot_product, -1.0, 1.0)  # Clip for numerical stability
    angle = 2 * np.arccos(dot_product)
    return angle

def simulate_just_forward(m,d,action,time_step, viewer=None):
    dt=0.001
    healthy_reward= 2.0
    forward_reward_weight=2.0
    ctrl_cost_weight=0.7
    contact_cost_weight=0.01
    action = np.clip(action, -0.6, 0.6)

    np.copyto(d.ctrl, action)
    mujoco.mj_step(m, d)

    left_foot_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'left_feet')
    right_foot_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'right_feet')
    left_feet_joint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "left_feet_joint")
    right_feet_joint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "right_feet_joint")
    # left_foot_quat = d.xquat[left_foot_id]  # [w, x, y, z]
    # right_foot_quat = d.xquat[right_foot_id]
    # ground_quat = np.array([1, 0, 0, 0])  # Identity quaternion
    # left_deviation = quaternion_angle_deviation(left_foot_quat, ground_quat)
    # right_deviation = quaternion_angle_deviation(right_foot_quat, ground_quat)

    left_feet_angle = d.qpos[left_feet_joint_id]
    right_feet_angle = d.qpos[right_feet_joint_id]
    # print(left_foot_id, right_foot_id)

    # left_foot_height = d.xpos[left_foot_id][2]  # z-axis for vertical height
    # right_foot_height = d.xpos[right_foot_id][2]
    # foot_lift_threshold=0.1
    # foot_lift_reward_weight = 0.5

    # # Reward for lifting feet
    # reward_foot_lift = 0
    # if left_foot_height > foot_lift_threshold and right_foot_height < foot_lift_threshold:
    #     reward_foot_lift += foot_lift_reward_weight
    # elif right_foot_height > foot_lift_threshold and left_foot_height < foot_lift_threshold:
    #     reward_foot_lift += foot_lift_reward_weight

    # Get initial and updated states
    new_state = d.qpos.copy()
    new_velocity = d.qvel.copy()
    next_state = np.concatenate((new_state, new_velocity))


    # foot_speed_pen =  -np.abs(left_feet_joint_speed) *feet_scalar - np.abs(right_feet_joint_speed)*feet_scalar
    # Reward components
    # 1. Healthy reward (constant)
    reward_healthy = healthy_reward


    # 2. Forward reward based on head's x-coordinate CoM movement
    head_initial_x = d.qpos[0] # Head x-position before action
    mujoco.mj_step(m, d)
    if viewer!=None:
        # robot_pos = d.qpos[:3]
        # viewer.cam.lookat[:] = robot_pos
        viewer.sync()
    time_step+=1
    head_final_x = d.qpos[0]  # Head x-position after action
    if head_final_x - head_initial_x < 0:
        forward_reward = 0
    else:
        forward_reward = forward_reward_weight * min(((head_final_x - head_initial_x) / dt), 2.0)

    
    # 3. Control cost (penalizes large control forces)
    ctrl_cost = ctrl_cost_weight * np.sum(np.square(action))

    # 4. Contact cost (penalizes large external contact forces)
    contact_forces = np.linalg.norm(d.cfrc_ext, axis=1)
    contact_cost = contact_cost_weight * np.sum(np.square(contact_forces))
    #contact_cost = np.clip(contact_cost, contact_cost_range[0], contact_cost_range[1])
    # Get foot heights to check if they're lifted
    # if foot_lift_reward_weight >= 0.5:
        # print('lifted foot!')

    # Early termination conditions if desired
    # joint_angles = d.qpos[7:]
    # print(joint_angles)
    done = False

    ground_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'floor')


    #print(right_foot_id,ground_id,left_foot_id)
    for i in range(d.ncon):
        contact = d.contact[i]

        # Geoms involved in the contact
        geom1 = contact.geom1
        geom2 = contact.geom2

        # Check if the contact involves the ground and a non-foot body part
        if (geom1 == ground_id and geom2 not in [left_foot_id, right_foot_id]) or \
           (geom2 == ground_id and geom1 not in [left_foot_id, right_foot_id]):
            # print('other parts touching')
            done=True


    time_step+=1
    if time_step > 100000:
        done = True

    
    reward = reward_healthy + forward_reward - ctrl_cost - contact_cost
    if np.abs(left_feet_angle) >0.6:
         reward -= 1.5
    if np.abs(right_feet_angle)>0.6:
         reward -= 1.5

    return next_state, reward, done, time_step 

def set_seed(seed=57):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If you're using GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set the seed
set_seed(991)
nthreads=8
if __name__ == '__main__':
    filename = "robot_assm/meshes/mjmodel.xml"
    m = mujoco.MjModel.from_xml_path(filename)
    d = mujoco.MjData(m)
    state_space = np.concatenate((d.qpos, d.qvel))
    action_space = len(d.ctrl)
    print(state_space.shape, action_space)
    agent = Agent(alpha=3e-4, beta=3e-4,
        input_dims=state_space.shape, tau=0.005, batch_size=256, layer1_size=1024, layer2_size=1024,layer3_size=1024,
        n_actions=action_space, warmup=2000, chkpt_dir='tmp_feet/test3')
    # agent.load_models()
    
    n_games = 100000
    # uncomment this line and do a mkdir tmp && mkdir video if you want to
    # record video of the agent playing the game.
    # #env = wrappers.Monitor(env, 'tmp/video', video_callable=lambda episode_id: True, force=True)
    # filename = 'inverted_pendulum.png'

    # figure_file = 'plots/' + filename
    def run_simulation(agent, filename): 
        m = mujoco.MjModel.from_xml_path(filename)
        d = mujoco.MjData(m)
        d.qpos[:] = 0
        d.qvel[:] = 0
        d.qpos[2] = 0.01
        d.qpos[3] = 1
        observation = np.concatenate((d.qpos, d.qvel))
        time_step = 0 

        done = False
        score = 0


        while not done:
                action = agent.choose_action(observation)
                observation_, reward, done, time_step = simulate_just_forward(m, d, action, time_step)
                score += reward

                agent.remember(observation, action, reward, observation_, done)
                agent.learn()
                observation = observation_

        return score

                    
                    
    score_history = []
    load_checkpoint = False
    best_score = -np.inf
    def worker(thread_id):
        #print(f"Thread {thread_id} starting")
        return run_simulation(agent, filename)
    for i in range(n_games):

        with concurrent.futures.ThreadPoolExecutor(max_workers=nthreads) as executor:
            # Submit multiple jobs to the thread pool
            futures = [executor.submit(worker, i) for i in range(nthreads)]
            
            # Gather the results from all threads
            for future in concurrent.futures.as_completed(futures):
                score_history.append(future.result())
                        
        avg_score = np.mean(score_history[-100:])


        d.qpos[:] = 0
        d.qvel[:] = 0
        d.qpos[3] = 1
        d.qpos[2] = 0.01
        observation = np.concatenate((d.qpos, d.qvel))
        time_step = 0 

        done = False
        score = 0


        with mujoco.viewer.launch_passive(m, d) as viewer:
            print('start')
            
            while not done:
         
                    com_z = d.subtree_com[0][2]
                    #print(com_z)
                    action = agent.choose_action(observation)
                    observation_, reward, done, time_step = simulate_just_forward(m, d, action, time_step,viewer)
                    
                    score += reward

                    agent.remember(observation, action, reward, observation_, done)
                    agent.learn()
                    observation = observation_
                    # time.sleep(0.1)
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        # if avg_score> 100:

        #     break


        print('episode ', i, 'score %.1f' % score_history[-1], 'avg_score %.1f' % avg_score)
