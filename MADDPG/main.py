import numpy as np
from multiagents import MultiAgent
from replay_buffer import Replay
from pettingzoo.mpe import simple_speaker_listener_v4

def obs_list_to_state_vector(observation):
    # convert observation to numpy arrays
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state

def main():
    env = simple_speaker_listener_v4.parallel_env(continuous_actions=True)
    _, _ = env.reset()
    n_agents = env.max_num_agents
    print(f"Number of agents: {n_agents}")

    actor_dims = []
    n_actions = []
    for agent in env.agents:
        actor_dims.append(env.observation_space(agent).shape[0])
        n_actions.append(env.action_space(agent).shape[0])
    critic_dims = sum(actor_dims) + sum(n_actions) # take action and obs from all the agents
    agents = MultiAgent(actor_dims=actor_dims, critic_dims=critic_dims, n_agents=n_agents, n_actions=n_actions, env=env, ckp_dir='tmp/', gamma=0.95, lr_actor=1e-4, lr_critic=1e-3)
    critic_dims = sum(actor_dims) # only use observations
    memory = Replay(mem_size=1_000_000, critic_dim=critic_dims, actor_dim=actor_dims, n_actions=n_actions, n_agents=n_agents, batch_size=1024)

    load_checkpoint = False
    if load_checkpoint:
        agent.load_models()

    EVAL_INTERVAL = 1000
    MAX_STEPS = 1_000_000

    n_steps = 0
    episode = 0
    eval_scores, eval_steps = [], []

    score = evaluate(agents, env, episode, n_steps)
    eval_scores.append(score)
    eval_steps.append(n_steps)

    while n_steps < MAX_STEPS:
        obs, _ = env.reset()
        terminal = [False] * n_agents
        while not any(terminal):
            actions = agents.choose_action(obs)

            obs_, reward, done, trunc, info = env.step(actions)
            # convert dictionary values to list
            list_done = list(done.values())
            list_obs = list(obs.values())
            list_reward = list(reward.values())
            list_actions = list(actions.values())
            list_obs_ = list(obs_.values())
            list_trunc = list(trunc.values())

            state = obs_list_to_state_vector(list_obs)
            state_ = obs_list_to_state_vector(list_obs_)

            terminal = [d or t for d, t in zip(list_done, list_trunc)]
            memory.store_transition(list_obs, state, list_actions, list_reward,
                                    list_obs_, state_, terminal)

            if n_steps % 100 == 0:
                agents.learn(memory)
            obs = obs_
            n_steps += 1

        if n_steps % EVAL_INTERVAL == 0:
            score = evaluate(agents, env, episode, n_steps)
            eval_scores.append(score)
            eval_steps.append(n_steps)

        if not load_checkpoint:
            agents.save_checkpoint()


        episode += 1
    np.save('data/maddpg_scores.npy', np.array(eval_scores))
    np.save('data/maddpg_steps.npy', np.array(eval_steps))


def evaluate(agents, env, episode, step, n_eval=3):
    # no saving or learning
    score_history = []
    for i in range(n_eval):
        obs, _ = env.reset()
        score = 0
        terminal = [False] * env.max_num_agents
        while not any(terminal):
            actions = agents.choose_action(obs, evaluate=True)
            obs_, reward, done, trunc, info = env.step(actions)

            list_trunc = list(trunc.values())
            list_reward = list(reward.values())
            list_done = list(done.values())

            terminal = [d or t for d, t in zip(list_done, list_trunc)]
            obs = obs_
            score += sum(list_reward) # for cooperative domains
        score_history.append(score)
    avg_score = np.mean(score_history)
    print(f'Evaluation episode {episode} train steps {step}'
          f' average score {avg_score:.1f}')
    return avg_score


if __name__ == '__main__':
    main()

    