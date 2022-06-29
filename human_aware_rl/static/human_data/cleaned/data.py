import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import pickle
import os
from human_aware_rl.static import *

def split_data():
    data = pd.read_pickle('2020_hh_trials_all.pickle')
    print("unique players: ", len(pd.unique(data['player_0_id'])))
    print("number of trials: ", len(pd.unique(data['trial_id'])))
    print("total number of datapoints: ", len(data['trial_id']))
    print("layout names: ", pd.unique(data['layout_name']))

    grouped = data.groupby(data.player_0_id)
    trials, players = [], []
    asym, counter, cramped, soup, nopass = [], [], [], [], []

    for idx, i in enumerate(pd.unique(data['player_0_id'])):
        players.append(i)
        group = grouped.get_group(i)
        trials.append(len(group))
        layout_sort = group.groupby(group.layout_name)
        layouts = pd.unique(group['layout_name'])
        if 'asymmetric_advantages_tomato' in layouts:
            asym.append(len(layout_sort.get_group('asymmetric_advantages_tomato')))
        if 'counter_circuit' in layouts:
            counter.append(len(layout_sort.get_group('counter_circuit')))
        if 'cramped_corridor' in layouts:
            cramped.append(len(layout_sort.get_group('cramped_corridor')))
        if 'soup_coordination' in layouts:
            soup.append(len(layout_sort.get_group('soup_coordination')))
        if 'you_shall_not_pass' in layouts:
            nopass.append(len(layout_sort.get_group('you_shall_not_pass')))
        #group.to_pickle('2020_hh_trials_player' + str(idx) + '.pickle')

def combine_train_curves(first, second):
    # load json results data
    data1 = [json.loads(line) for line in open(first, 'r')]
    data2 = [json.loads(line) for line in open(second, 'r')]
    # combine data
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    df = pd.concat([df1, df2])
    # plot data
    df.plot(x='timesteps_total', y='episode_reward_mean')
    plt.axvline(x=8000800, color='k', linestyle='--')
    plt.show()

def plot_players():
    acc = [66.27, 78.73, 65.08, 71.62, 69.39, 72.87, 77.86, 69.54, 80.68, 58.31, 79.49, 80.83, 73.29,
        63.79, 67.81, 71.32, 76.26, 70.58, 70.88, 75.82, 66.86, 81.60, 70.77, 70.88, 74.29, 78.19,
        73.03, 65.53, 55.42, 79.49, 76.85, 72.81, 67.61, 72.36]
    val_acc = [63.03, 74.17, 77.31, 49.58, 57.98, 59.66, 67.23, 60.50, 60.50, 41.67, 77.31, 63.87,
                64.17, 45.38, 52.94, 59.66, 75.00, 68.91, 63.87, 75.00, 52.94, 77.50, 40.00, 67.23,
                79.83, 57.50, 89.08, 84.03, 55.46, 87.39, 70.00, 91.60, 73.11, 40.34]

    fig, ax = plt.subplots()
    ax.scatter(np.arange(0, 34, step=1), acc, label='train acc')
    ax.scatter(np.arange(0, 34, step=1), val_acc, label='val acc')
    ax.legend()
    ax.set_xlabel('player')
    ax.set_ylabel('percentage')
    ax.set_xticks(np.arange(0, 34, step=1))
    plt.grid(True)
    plt.show()

def combine_players(players, kind):
    combined = pd.DataFrame()
    for i in players:
        filename = os.path.join(CLEAN_HUMAN_DATA_DIR, "2020_hh_trials_player" + str(i) + ".pickle")
        data = pd.read_pickle(filename)
        combined = combined.append(data)
    newfile = os.path.join(CLEAN_HUMAN_DATA_DIR, "2020_hh_trials_player" + kind + ".pickle")
    combined.to_pickle(newfile)

if __name__ == "__main__":
    # split_data()
    # plot_players()
    # players = [9]
    # all_players = np.arange(34)
    # combine_players(np.delete(all_players, players), 'base9')
    first = '/Users/rupaln/ray_results/PPO_asymmetric_advantages_tomato_True_nw=2_vf=0.000100_es=0.200000_en=0.100000_kl=0.200000_1_2022-06-09_00-27-30mur9pciv/result.json'
    second = '/Users/rupaln/ray_results/PPO_asymmetric_advantages_tomato_True_nw=2_vf=0.000100_es=0.200000_en=0.100000_kl=0.200000_1_2022-06-09_10-55-38ajobgph9/result.json'
    combine_train_curves(first, second)