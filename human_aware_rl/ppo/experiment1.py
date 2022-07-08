import os, pickle, copy, argparse, sys
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from human_aware_rl.rllib.rllib import RlLibAgent, get_base_ae
from tensorflow import keras
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.reset_default_graph()

from overcooked_ai_py.agents.agent import AgentPair, AgentFromPolicy
from overcooked_ai_py.agents.benchmarking import AgentEvaluator

from human_aware_rl.human.process_dataframes import train_test_split
from human_aware_rl.static import *
from human_aware_rl.data_dir import DATA_DIR

#import ray
from ray.tune.result import DEFAULT_RESULTS_DIR
from human_aware_rl.ppo.ppo_rllib import RllibPPOModel, RllibLSTMPPOModel
from human_aware_rl.rllib.rllib import OvercookedMultiAgent, save_trainer, gen_trainer_from_params, get_agent_from_trainer, load_trainer, load_agent, evaluate
from human_aware_rl.imitation.behavior_cloning_tf2 import BehaviorCloningPolicy, load_bc_model

SAVE_DIR = os.path.join(DATA_DIR)

# environment variable that tells us whether this code is running on the server or not
LOCAL_TESTING = True

# Dummy wrapper to pass rllib type checks
def _env_creator(env_config):
    # Re-import required here to work with serialization
    from human_aware_rl.rllib.rllib import OvercookedMultiAgent 
    return OvercookedMultiAgent.from_config(env_config)

def my_config():
    ### Model params ###

    # Whether dense reward should come from potential function or not
    use_phi = True

    # whether to use recurrence in ppo model
    use_lstm = False

    # Base model params
    NUM_HIDDEN_LAYERS = 3
    SIZE_HIDDEN_LAYERS = 64
    NUM_FILTERS = 25
    NUM_CONV_LAYERS = 3

    # LSTM memory cell size (only used if use_lstm=True)
    CELL_SIZE = 256

    # whether to use D2RL https://arxiv.org/pdf/2010.09163.pdf (concatenation the result of last conv layer to each hidden layer); works only when use_lstm is False
    D2RL = False
    ### Training Params ###

    num_workers = 30 if not LOCAL_TESTING else 2

    # list of all random seeds to use for experiments, used to reproduce results
    seeds = [1]

    # Placeholder for random for current trial
    seed = None

    # Number of gpus the central driver should use
    num_gpus = 0 if LOCAL_TESTING else 1

    # How many environment timesteps will be simulated (across all environments)
    # for one set of gradient updates. Is divided equally across environments
    train_batch_size = 12000 if not LOCAL_TESTING else 800

    # size of minibatches we divide up each batch into before
    # performing gradient steps
    sgd_minibatch_size = 2000 if not LOCAL_TESTING else 800

    # Rollout length
    rollout_fragment_length = 400
    
    # Whether all PPO agents should share the same policy network
    shared_policy = True

    # Number of training iterations to run
    num_training_iters = 420 if not LOCAL_TESTING else 10000

    # Stepsize of SGD.
    lr = 5e-5

    # Learning rate schedule.
    lr_schedule = None

    # If specified, clip the global norm of gradients by this amount
    grad_clip = 0.1

    # Discount factor
    gamma = 0.99

    # Exponential decay factor for GAE (how much weight to put on monte carlo samples)
    # Reference: https://arxiv.org/pdf/1506.02438.pdf
    lmbda = 0.98

    # Whether the value function shares layers with the policy model
    vf_share_layers = True

    # How much the loss of the value network is weighted in overall loss
    vf_loss_coeff = 1e-4

    # Entropy bonus coefficient, will anneal linearly from _start to _end over _horizon steps
    entropy_coeff_start = 0.2
    entropy_coeff_end = 0.1
    entropy_coeff_horizon = 3e5

    # Initial coefficient for KL divergence.
    kl_coeff = 0.2

    # PPO clipping factor
    clip_param = 0.05

    # Number of SGD iterations in each outer loop (i.e., number of epochs to
    # execute per train batch).
    num_sgd_iter = 8 if not LOCAL_TESTING else 1

    # How many trainind iterations (calls to trainer.train()) to run before saving model checkpoint
    save_freq = 100

    # How many training iterations to run between each evaluation
    evaluation_interval = 50 if not LOCAL_TESTING else 0

    # How many timesteps should be in an evaluation episode
    evaluation_ep_length = 400

    # Number of games to simulation each evaluation
    evaluation_num_games = 1

    # Whether to display rollouts in evaluation
    evaluation_display = False

    # Where to log the ray dashboard stats
    temp_dir = os.path.join(os.path.abspath(os.sep), "tmp", "ray_tmp")

    # Where to store model checkpoints and training stats
    results_dir = DEFAULT_RESULTS_DIR

    # Whether tensorflow should execute eagerly or not
    eager = False

    # Whether to log training progress and debugging info
    verbose = True

    ### BC Params ###
    # path to pickled policy model for behavior cloning
    # bc_model_dir = os.path.join(SAVE_DIR, "train")
    bc_model_dir = '/Users/rupaln/Documents/uiuc/research/human_aware_rl/human_aware_rl/imitation/bc_runs/not9_train'

    # Whether bc agents should return action logit argmax or sample
    bc_stochastic = True

    ### Environment Params ###
    # Which overcooked level to use
    layout_name = "cramped_room"

    params_str = str(use_phi) + "_nw=%d_vf=%f_es=%f_en=%f_kl=%f" % (
        num_workers,
        vf_loss_coeff,
        entropy_coeff_start,
        entropy_coeff_end,
        kl_coeff
    )

    experiment_name = "{0}_{1}_{2}".format("PPO", layout_name, params_str)

    # Rewards the agent will receive for intermediate actions
    rew_shaping_params = {
        "PLACEMENT_IN_POT_REW": 3,
        "DISH_PICKUP_REWARD": 3,
        "SOUP_PICKUP_REWARD": 5,
        "DISH_DISP_DISTANCE_REW": 0,
        "POT_DISTANCE_REW": 0,
        "SOUP_DISTANCE_REW": 0,
    }

    # Max episode length
    horizon = 400

    # Constant by which shaped rewards are multiplied by when calculating total reward
    reward_shaping_factor = 1.0

    # Linearly anneal the reward shaping factor such that it reaches zero after this number of timesteps
    reward_shaping_horizon = float('inf')
    bc_schedule = OvercookedMultiAgent.self_play_bc_schedule


    # To be passed into rl-lib model/custom_options config
    model_params = {
        "use_lstm" : use_lstm,
        "NUM_HIDDEN_LAYERS" : NUM_HIDDEN_LAYERS,
        "SIZE_HIDDEN_LAYERS" : SIZE_HIDDEN_LAYERS,
        "NUM_FILTERS" : NUM_FILTERS,
        "NUM_CONV_LAYERS" : NUM_CONV_LAYERS,
        "CELL_SIZE" : CELL_SIZE,
        "D2RL": D2RL
    }

    # to be passed into the rllib.PPOTrainer class
    training_params = {
        "num_workers" : num_workers,
        "train_batch_size" : train_batch_size,
        "sgd_minibatch_size" : sgd_minibatch_size,
        "rollout_fragment_length" : rollout_fragment_length,
        "num_sgd_iter" : num_sgd_iter,
        "lr" : lr,
        "lr_schedule" : lr_schedule,
        "grad_clip" : grad_clip,
        "gamma" : gamma,
        "lambda" : lmbda,
        "vf_share_layers" : vf_share_layers,
        "vf_loss_coeff" : vf_loss_coeff,
        "kl_coeff" : kl_coeff,
        "clip_param" : clip_param,
        "num_gpus" : num_gpus,
        "seed" : seed,
        "evaluation_interval" : evaluation_interval,
        "entropy_coeff_schedule" : [(0, entropy_coeff_start), (entropy_coeff_horizon, entropy_coeff_end)],
        "eager" : eager,
        "log_level" : "WARN" if verbose else "ERROR"
    }

    # To be passed into AgentEvaluator constructor and _evaluate function
    evaluation_params = {
        "ep_length" : evaluation_ep_length,
        "num_games" : evaluation_num_games,
        "display" : evaluation_display
    }


    environment_params = {
        # To be passed into OvercookedGridWorld constructor

        "mdp_params" : {
            "layout_name": layout_name,
            "rew_shaping_params": rew_shaping_params
        },
        # To be passed into OvercookedEnv constructor
        "env_params" : {
            "horizon" : horizon
        },

        # To be passed into OvercookedMultiAgent constructor
        "multi_agent_params" : {
            "reward_shaping_factor" : reward_shaping_factor,
            "reward_shaping_horizon" : reward_shaping_horizon,
            "use_phi" : use_phi,
            "bc_schedule" : bc_schedule
        }
    }

    bc_params = {
        "bc_policy_cls" : BehaviorCloningPolicy,
        "bc_config" : {
            "model_dir" : bc_model_dir,
            "stochastic" : bc_stochastic,
            "eager" : eager
        }
    }

    ray_params = {
        "custom_model_id" : "MyPPOModel",
        "custom_model_cls" : RllibLSTMPPOModel if model_params['use_lstm'] else RllibPPOModel,
        "temp_dir" : temp_dir,
        "env_creator" : _env_creator
    }

    params = {
        "model_params" : model_params,
        "training_params" : training_params,
        "environment_params" : environment_params,
        "bc_params" : bc_params,
        "shared_policy" : shared_policy,
        "num_training_iters" : num_training_iters,
        "evaluation_params" : evaluation_params,
        "experiment_name" : experiment_name,
        "save_every" : save_freq,
        "seeds" : seeds,
        "results_dir" : results_dir,
        "ray_params" : ray_params,
        "verbose" : verbose
    }

    return params

def run(params):
    # Retrieve the tune.Trainable object that is used for the experiment
    trainer = gen_trainer_from_params(params)
    result = {}

    # Training loop
    for i in range(params['num_training_iters']):
        if params['verbose']:
            print("Starting training iteration", i)
        result = trainer.train()

        if i % params['save_every'] == 0:
            save_path = save_trainer(trainer, params)

    # Save the state of the experiment at end
    save_path = save_trainer(trainer, params)
    print("saved trainer at", save_path)

    return result

def run_trainer(trainer, params):
    # Retrieve the tune.Trainable object that is used for the experiment
    result = {}

    # Training loop
    for i in range(params['num_training_iters']):
        if params['verbose']:
            print("Starting training iteration", i)
        result = trainer.train()

        if i % params['save_every'] == 0:
            save_path = save_trainer(trainer, params)

    # Save the state of the experiment at end
    save_path = save_trainer(trainer, params)
    print("saved trainer at", save_path)

    return result

def load_data(file):
    data = pd.read_pickle(file)
    print(data['layout_name'])
    split = train_test_split(data, print_stats=True)
    layouts = np.unique(data['layout_name'])
    train_trials = pd.concat([split[layout]["train"] for layout in layouts])
    test_trials = pd.concat([split[layout]["test"] for layout in layouts])
    train_trials.to_pickle("2020_hh_trials_playertrain.pickle")
    test_trials.to_pickle("2020_hh_trials_playertest.pickle")
    return train_trials, test_trials

def _get_base_ae(bc_params):
    return get_base_ae(bc_params['mdp_params'], bc_params['env_params'])

def evaluate_ppo_and_bc_models_for_layout(layout='asymmetric_advantages_tomato'):
    ppo_bc_performance = defaultdict(lambda: defaultdict(list))
    num_rounds = 1
    seeds = [0]

    agent_bc_test, bc_params = load_bc_model('/Users/rupaln/Documents/uiuc/research/human_aware_rl/human_aware_rl/imitation/bc_runs/not9_test')
    agent_bc_new, bc_params_new = load_bc_model('/Users/rupaln/Documents/uiuc/research/human_aware_rl/human_aware_rl/imitation/bc_runs/9')
    agent_bc_train, bc_params_train = load_bc_model('/Users/rupaln/Documents/uiuc/research/human_aware_rl/human_aware_rl/imitation/bc_runs/not9_train')
    
    base_ae = _get_base_ae(bc_params)
    base_env = base_ae.env
    def featurize_fn(state):
        return base_env.featurize_state_mdp(state)
    
    ppo_bc_holdout_path = '/Users/rupaln/ray_results/PPO_asymmetric_advantages_tomato_True_nw=2_vf=0.000100_es=0.200000_en=0.100000_kl=0.200000_1_2022-06-09_00-27-30mur9pciv/checkpoint_10000/config.pkl'
    # ppo_bc_all_path = '/Users/rupaln/ray_results/PPO_asymmetric_advantages_tomato_True_nw=2_vf=0.000100_es=0.200000_en=0.100000_kl=0.200000_0_2022-04-18_19-47-151845fyll/checkpoint_10000/'
    evaluator = AgentEvaluator.from_layout_name(mdp_params=bc_params["mdp_params"], env_params=bc_params["env_params"])
    
    for seed_idx in range(num_rounds):
        agent_ppo_bc_holdout_trainer = load_trainer(ppo_bc_holdout_path)
        print(agent_ppo_bc_holdout_trainer._evaluate())
        agent_ppo_bc_holdout = get_agent_from_trainer(agent_ppo_bc_holdout_trainer)
        # agent_ppo_bc_all = get_agent_from_trainer(load_trainer(ppo_bc_all_path))

        # how well does agent do with itself?
        #ppo_and_ppo = evaluator.evaluate_agent_pair(AgentPair(agent_ppo_bc_holdout, agent_ppo_bc_holdout, allow_duplicate_agents=True), num_games=max(int(num_rounds/2), 1), display=False, native_eval=True)
        #ppo_and_ppo = evaluate(bc_params['evaluation_params'], bc_params['mdp_params'], None, agent_ppo_bc_holdout_trainer.get_policy('ppo'), agent_ppo_bc_holdout_trainer.get_policy('ppo'), None, None, False)
        #avg_ppo_and_ppo = np.mean(ppo_and_ppo['ep_returns'])
        #ppo_bc_performance[layout]["PPO_BC_holdout+PPO_BC_holdout"].append(avg_ppo_and_ppo)

        # How well it generalizes to new agent in simulation?
        pair = AgentPair(agent_ppo_bc_holdout, RlLibAgent(BehaviorCloningPolicy.from_model(agent_bc_new, bc_params_new), agent_index=1, featurize_fn=featurize_fn))
        ppo_and_bc_new = evaluator.evaluate_agent_pair(pair, num_games=num_rounds, display=False)
        avg_ppo_and_bc = np.mean(ppo_and_bc_new['ep_returns'])
        ppo_bc_performance[layout]["PPO_BC_holdout+BC_new_0"].append(avg_ppo_and_bc)

        #pair = AgentPair(RlLibAgent(BehaviorCloningPolicy.from_model(agent_bc_new, bc_params_new), agent_index=0, featurize_fn=featurize_fn), agent_ppo_bc_holdout)
        #bc_and_ppo_new = evaluator.evaluate_agent_pair(pair, num_games=num_rounds, display=False)
        #avg_bc_and_ppo = np.mean(bc_and_ppo_new['ep_returns'])
        #ppo_bc_performance[layout]["PPO_BC_holdout+BC_new_1"].append(avg_bc_and_ppo)
        
        # How well could we do if we knew true model BC?
        #pair = AgentPair(agent_ppo_bc_holdout, RlLibAgent(BehaviorCloningPolicy.from_model(agent_bc_train, bc_params), agent_index=1, featurize_fn=featurize_fn))
        #ppo_and_bc = evaluator.evaluate_agent_pair(pair, num_games=num_rounds, display=False)
        #avg_ppo_and_bc = np.mean(ppo_and_bc['ep_returns'])
        #ppo_bc_performance[layout]["PPO_BC_holdout+BC_train_0"].append(avg_ppo_and_bc)

        #pair = AgentPair(RlLibAgent(BehaviorCloningPolicy.from_model(agent_bc_train, bc_params), agent_index=0, featurize_fn=featurize_fn), agent_ppo_bc_holdout)
        #bc_and_ppo = evaluator.evaluate_agent_pair(pair, num_games=num_rounds, display=False)
        #avg_bc_and_ppo = np.mean(bc_and_ppo['ep_returns'])
        #ppo_bc_performance[layout]["PPO_BC_holdout+BC_train_1"].append(avg_bc_and_ppo)
    
    return ppo_bc_performance

def plot_runs_training_curves(ppo_bc_model_paths, seeds, single=False):
    # Plot PPO BC models
    for run_type, type_dict in ppo_bc_model_paths.items():
        print(run_type)
        for layout, layout_model_path in type_dict.items():
            print(layout)
            plt.figure(figsize=(8,5))
            # plot_ppo_run(layout_model_path, sparse=True, print_config=False, single=single, seeds=seeds[run_type])
            plt.xlabel("Environment timesteps")
            plt.ylabel("Mean episode reward")
            plt.savefig("rew_ppo_bc_{}_{}".format(run_type, layout), bbox_inches='tight')
            plt.show()

def main(params):
    # List of each random seed to run
    seeds = params['seeds']
    del params['seeds']
    # print(params['bc_params']['bc_config']['model_dir'])
    # params['bc_params']['bc_config']['model_dir'] = '/home/rupaln/dev/human_aware_rl/human_aware_rl/imitation/bc_runs/9'
    results = []
    # ppo_bc_holdout_path = '/home/rupaln/dev/human_aware_rl/human_aware_rl/ppo/model_not9/checkpoint_10000/config.pkl'
    # agent_ppo_bc_holdout = load_trainer(ppo_bc_holdout_path)

    # Train an agent to completion for each random seed specified
    for seed in seeds:
        # Override the seed
        params['training_params']['seed'] = seed
        # Do the thing
        result = run(params)
        #result = run_trainer(agent_ppo_bc_holdout, params)
        results.append(result)

    average_sparse_reward = np.mean([res['custom_metrics']['sparse_reward_mean'] for res in results])
    average_episode_reward = np.mean([res['episode_reward_mean'] for res in results])
    print("average_sparse_reward: " + str(average_sparse_reward), " average_total_reward: " + str(average_episode_reward))

if __name__ == "__main__":
    main(my_config())
    # load_data('/Users/rupaln/Documents/uiuc/research/human_aware_rl/human_aware_rl/static/human_data/cleaned/2020_hh_trials_player9.pickle')
    # print(evaluate_ppo_and_bc_models_for_layout())
