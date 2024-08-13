import cv2 as cv
import numpy as np
from multiagents import MultiAgent
from replay_buffer import Replay
from pettingzoo.mpe import simple_speaker_listener_v4


# test how the agent is doing
if __name__ == '__main__':

    env = simple_speaker_listener_v4.parallel_env(continuous_actions=True, render_mode="human")


    # define video recording params
    video_path = './thumbnails/video.mp4'  
    frame_width = 640
    frame_height = 480
    frame_rate = 30.0  

    _, _ = env.reset()
    n_agents = env.max_num_agents
    actor_dims = []
    n_actions = []
    for agent in env.agents:
        actor_dims.append(env.observation_space(agent).shape[0])
        n_actions.append(env.action_space(agent).shape[0])
    critic_dims = sum(actor_dims) + sum(n_actions) # take action and obs from all the agents
    agents = MultiAgent(actor_dims=actor_dims, critic_dims=critic_dims, n_agents=n_agents, n_actions=n_actions, env=env, ckp_dir='tmp/', gamma=0.95, lr_actor=1e-4, lr_critic=1e-3)
    critic_dims = sum(actor_dims) # only use observations
    memory = Replay(mem_size=1_000_000, critic_dims=critic_dims, actor_dims=actor_dims, n_actions=n_actions, n_agents=n_agents, batch_size=1024)
 

    n_games = 3
    best_score = 0
    agents.load_checkpoint()

    # initialize video writter
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  
    out = cv.VideoWriter(video_path, fourcc, frame_rate, (frame_width, frame_height))

    # training loop
    for i in range(n_games):
        obs, _ = env.reset()
        terminal = [False] * n_agents
        score = 0

        while not any(terminal):
            actions = agents.choose_action(obs)
            obs_, reward, done, trunc, info = env.step(actions)
            list_trunc = list(trunc.values())
            list_reward = list(reward.values())
            list_done = list(done.values())

            terminal = [d or t for d, t in zip(list_done, list_trunc)]
            env.render()

            # capture the current screen render and write out
            frame = env.render()
            frame_bgr = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            out.write(frame_bgr)

            score += sum(list_reward)
            obs = obs_
            #time.sleep(0.03)

        print(f'episode: {i}, score: {score}')

    out.release()
