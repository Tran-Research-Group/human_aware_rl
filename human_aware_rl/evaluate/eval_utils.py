import numpy as np
import itertools
import time
import pickle
from os import makedirs

from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.agents.agent import AgentPair, GreedyHumanModel
from overcooked_ai_py.mdp.layout_generator import LayoutGenerator

from human_aware_rl.rllib.rllib import load_agent_pair, load_agent

# Whether or not to display the game during evaluation process
DISPLAY = False

# Whether to print information during evaluation
PRINTING = True

# Where to save the data
SAVE_DIR = "data/"

makedirs(SAVE_DIR, exist_ok=True)


def single_layout_stats(evaluator, agent_pair, num_games=5):
    """
    Arguments:
        evaluator (AgentEvaluator): the evaluator
        agent_pair (AgentPair): the pair of agents being
        num_games (int): the number of games
    """
    trajectory = evaluator.evaluate_agent_pair(agent_pair=agent_pair, num_games=num_games, native_eval=True, display=DISPLAY)
    sparse_reward_arr = trajectory["ep_returns"]

    if PRINTING:
        print("sparse rewards", sparse_reward_arr)
    return sparse_reward_arr, [], [], []


def agent_pair_lst_compare_stats(evaluator, agent_pair_lst, num_layouts=1, num_games_per_layout=40):
    """
        Arguments:
        evaluator (AgentEvaluator)
        agent_pair_lst (python list of AgentPair)
        num_layouts (int)
        num_games (int): number of games per layouts
    """
    num_agent_pairs = len(agent_pair_lst)
    layout_namelst = []
    spares_reward_lst = np.zeros((num_layouts, num_agent_pairs, num_games_per_layout))
    sparse_reward_mean_lst = np.zeros((num_layouts, num_agent_pairs))
    sparse_reward_std_lst = np.zeros((num_layouts, num_agent_pairs))

    for i in range(num_layouts):
        if PRINTING:
            print("layout", i)
            # print(evaluator.env.mdp.layout_name)
        layout_namelst.append(evaluator.env.mdp.layout_name)
        for j in range(num_agent_pairs):
            if PRINTING:
                print("starting pair", j)
            agent_pair = agent_pair_lst[j]
            # to ensure that starting position for each agent pair is the same
            assert evaluator.env.mdp.layout_name == layout_namelst[-1]
            sr, c1, c2, c3 = single_layout_stats(evaluator, agent_pair, num_games_per_layout)
            spares_reward_lst[i][j] = np.array(sr)
            sparse_reward_mean_lst[i][j] = np.mean(sr)
            sparse_reward_std_lst[i][j] = np.std(sr)
        evaluator.env.reset(regen_mdp=True)

    if PRINTING:
        print('sparse reward mean', sparse_reward_mean_lst)
        print('sparse reward std', sparse_reward_std_lst)

    current_time = str(time.time())

    file_name = current_time

    with open(SAVE_DIR + 'layouts' + file_name + '.pkl', 'wb') as f:
        pickle.dump(layout_namelst, f)

    np.savez(SAVE_DIR + 'data' + file_name + ".npz",
             sparse_reward_lst=spares_reward_lst,
             sparse_reward_mean_lst=sparse_reward_mean_lst,
             sparse_reward_std_lst=sparse_reward_std_lst,
             )
    return file_name


def from_params_stats_ppo_agent_pair_lst_self_play(mdp_gen_param, checkpoint_path_lst, outer_shape, num_layouts=1, num_games=40):
    mdp_gen_fn = LayoutGenerator.mdp_gen_fn_from_dict(mdp_gen_param, outer_shape=outer_shape)
    eva = AgentEvaluator(env_params={"horizon": 400}, mdp_fn=mdp_gen_fn, force_compute=True)
    agent_pair_lst = [load_agent_pair(checkpoint_path, featurize_fn=lambda state: eva.env.lossless_state_encoding_mdp(state))
                      for checkpoint_path in checkpoint_path_lst]
    return agent_pair_lst_compare_stats(eva, agent_pair_lst, num_layouts, num_games)


def from_params_stats_ppo_agent_pair_lst_mixed_play(mdp_gen_param, checkpoint_path_lst, outer_shape, num_layouts=1, num_games=40):
    mdp_gen_fn = LayoutGenerator.mdp_gen_fn_from_dict(mdp_gen_param, outer_shape=outer_shape)
    eva = AgentEvaluator(env_params={"horizon": 400}, mdp_fn=mdp_gen_fn, force_compute=True)

    agent_pair_lst = []

    for checkpoint_path_0, checkpoint_path_1 in itertools.product(checkpoint_path_lst, checkpoint_path_lst):
        agent_0 = load_agent(checkpoint_path_0, agent_index=0, featurize_fn=lambda state: eva.env.lossless_state_encoding_mdp(state))
        agent_1 = load_agent(checkpoint_path_1, agent_index=1, featurize_fn=lambda state: eva.env.lossless_state_encoding_mdp(state))
        agent_pair = AgentPair(agent_0, agent_1)
        agent_pair_lst.append(agent_pair)
    return agent_pair_lst_compare_stats(eva, agent_pair_lst, num_layouts, num_games)


def from_params_stats_ppo_agent_pair_lst_human_play(mdp_gen_param, include_greedy_human, ai_checkpoint_path_lst,
                                                    h_checkpoint_path_lst, outer_shape, num_layouts=1, num_games=40):
    mdp_gen_fn = LayoutGenerator.mdp_gen_fn_from_dict(mdp_gen_param, outer_shape=outer_shape)
    eva = AgentEvaluator(env_params={"horizon": 400}, mdp_fn=mdp_gen_fn, force_compute=True)

    agent_pair_lst = []

    for ai_checkpoint_path in ai_checkpoint_path_lst:
        if include_greedy_human:
            agent_0 = load_agent(ai_checkpoint_path, agent_index=0, featurize_fn=lambda state: eva.env.lossless_state_encoding_mdp(state))
            agent_g0 = GreedyHumanModel(eva.env.mlam)
            agent_pair_0 = AgentPair(agent_0, agent_g0)
            agent_pair_lst.append(agent_pair_0)

            agent_g1 = GreedyHumanModel(eva.env.mlam)
            agent_1 = load_agent(ai_checkpoint_path, agent_index=1, featurize_fn=lambda state: eva.env.lossless_state_encoding_mdp(state))
            agent_pair_1 = AgentPair(agent_g1, agent_1)
            agent_pair_lst.append(agent_pair_1)


            agent_2 = load_agent(ai_checkpoint_path, agent_index=0, featurize_fn=lambda state: eva.env.lossless_state_encoding_mdp(state))
            agent_g2 = GreedyHumanModel(eva.env.mlam, hl_boltzmann_rational=True, hl_temp=0.2)
            agent_pair_2 = AgentPair(agent_2, agent_g2)
            agent_pair_lst.append(agent_pair_2)

            agent_g3 = GreedyHumanModel(eva.env.mlam, hl_boltzmann_rational=True, hl_temp=0.2)
            agent_3 = load_agent(ai_checkpoint_path, agent_index=1, featurize_fn=lambda state: eva.env.lossless_state_encoding_mdp(state))
            agent_pair_3 = AgentPair(agent_g3, agent_3)
            agent_pair_lst.append(agent_pair_3)


            agent_4 = load_agent(ai_checkpoint_path, agent_index=0, featurize_fn=lambda state: eva.env.lossless_state_encoding_mdp(state))
            agent_g4 = GreedyHumanModel(eva.env.mlam, hl_boltzmann_rational=True, hl_temp=1)
            agent_pair_4 = AgentPair(agent_4, agent_g4)
            agent_pair_lst.append(agent_pair_4)

            agent_g5 = GreedyHumanModel(eva.env.mlam, hl_boltzmann_rational=True, hl_temp=1)
            agent_5 = load_agent(ai_checkpoint_path, agent_index=0, featurize_fn=lambda state: eva.env.lossless_state_encoding_mdp(state))
            agent_pair_5 = AgentPair(agent_g5, agent_5)
            agent_pair_lst.append(agent_pair_5)


            agent_6 = load_agent(ai_checkpoint_path, agent_index=0, featurize_fn=lambda state: eva.env.lossless_state_encoding_mdp(state))
            agent_g6 = GreedyHumanModel(eva.env.mlam, ll_boltzmann_rational=True, ll_temp=0.2)
            agent_pair_6 = AgentPair(agent_6, agent_g6)
            agent_pair_lst.append(agent_pair_6)

            agent_g7 = GreedyHumanModel(eva.env.mlam, ll_boltzmann_rational=True, ll_temp=0.2)
            agent_7 = load_agent(ai_checkpoint_path, agent_index=0, featurize_fn=lambda state: eva.env.lossless_state_encoding_mdp(state))
            agent_pair_7 = AgentPair(agent_g7, agent_7)
            agent_pair_lst.append(agent_pair_7)


            agent_8 = load_agent(ai_checkpoint_path, agent_index=0, featurize_fn=lambda state: eva.env.lossless_state_encoding_mdp(state))
            agent_g8 = GreedyHumanModel(eva.env.mlam, ll_boltzmann_rational=True, ll_temp=1)
            agent_pair_8 = AgentPair(agent_8, agent_g8)
            agent_pair_lst.append(agent_pair_8)

            agent_g9 = GreedyHumanModel(eva.env.mlam, ll_boltzmann_rational=True, ll_temp=1)
            agent_9 = load_agent(ai_checkpoint_path, agent_index=0, featurize_fn=lambda state: eva.env.lossless_state_encoding_mdp(state))
            agent_pair_9 = AgentPair(agent_g9, agent_9)
            agent_pair_lst.append(agent_pair_9)

        for h_checkpoint_path in h_checkpoint_path_lst:
            agent_2i = load_agent(ai_checkpoint_path, agent_index=0,
                                 featurize_fn=lambda state: eva.env.lossless_state_encoding_mdp(state))
            agent_bc_2i = load_agent(h_checkpoint_path, policy_id='bc', agent_index=1,
                                 featurize_fn=lambda state: eva.env.featurize_state(state))
            agent_pair_2i = AgentPair(agent_2i, agent_bc_2i)
            agent_pair_lst.append(agent_pair_2i)

            agent_bc_2i_plus_one = load_agent(h_checkpoint_path, policy_id='bc', agent_index=0,
                                 featurize_fn=lambda state: eva.env.featurize_state(state))
            agent_2i_plus_one = load_agent(ai_checkpoint_path, agent_index=1,
                                 featurize_fn=lambda state: eva.env.lossless_state_encoding_mdp(state))
            agent_pair_2i_plus_one = AgentPair(agent_bc_2i_plus_one, agent_2i_plus_one)
            agent_pair_lst.append(agent_pair_2i_plus_one)

    if include_greedy_human:
        agent_gg0 = GreedyHumanModel(eva.env.mlam)
        agent_gg1 = GreedyHumanModel(eva.env.mlam)
        agent_pair_g_01 = AgentPair(agent_gg0, agent_gg1)
        agent_pair_lst.append(agent_pair_g_01)

        agent_gg2 = GreedyHumanModel(eva.env.mlam, hl_boltzmann_rational=True, hl_temp=0.2)
        agent_gg3 = GreedyHumanModel(eva.env.mlam, hl_boltzmann_rational=True, hl_temp=0.2)
        agent_pair_g23 = AgentPair(agent_gg2, agent_gg3)
        agent_pair_lst.append(agent_pair_g23)

        agent_gg4 = GreedyHumanModel(eva.env.mlam, hl_boltzmann_rational=True, hl_temp=1)
        agent_gg5 = GreedyHumanModel(eva.env.mlam, hl_boltzmann_rational=True, hl_temp=1)
        agent_pair_g45 = AgentPair(agent_gg4, agent_gg5)
        agent_pair_lst.append(agent_pair_g45)

        agent_gg6 = GreedyHumanModel(eva.env.mlam, ll_boltzmann_rational=True, ll_temp=0.2)
        agent_gg7 = GreedyHumanModel(eva.env.mlam, ll_boltzmann_rational=True, ll_temp=0.2)
        agent_pair_g67 = AgentPair(agent_gg6, agent_gg7)
        agent_pair_lst.append(agent_pair_g67)

        agent_gg8 = GreedyHumanModel(eva.env.mlam, ll_boltzmann_rational=True, ll_temp=1)
        agent_gg9 = GreedyHumanModel(eva.env.mlam, ll_boltzmann_rational=True, ll_temp=1)
        agent_pair_g89 = AgentPair(agent_gg8, agent_gg9)
        agent_pair_lst.append(agent_pair_g89)

    for h_checkpoint_path in h_checkpoint_path_lst:
        agent_bc_i = load_agent(h_checkpoint_path, policy_id='bc', agent_index=0,
                              featurize_fn=lambda state: eva.env.featurize_state(state))
        agent_bc_i_plus_one = load_agent(h_checkpoint_path, policy_id='bc', agent_index=1,
                                 featurize_fn=lambda state: eva.env.featurize_state(state))
        agent_pair_i = AgentPair(agent_bc_i, agent_bc_i_plus_one)
        agent_pair_lst.append(agent_pair_i)

    return agent_pair_lst_compare_stats(eva, agent_pair_lst, num_layouts, num_games)