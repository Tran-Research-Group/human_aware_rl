#import tbplots
#iFilter = ["XXX1","XXX2","XXX3"]
#tbplots.PlotTensorflowData(path="/path/to/tblogs",gFilter="XXX",iFilter=iFilter, metric="winrate",saveName="FileName")

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import pickle
import os

def plot_averaged_curves(first, second, third, fourth, fifth):
      # load json results data
      data1 = [json.loads(line) for line in open(first, 'r')]
      data2 = [json.loads(line) for line in open(second, 'r')]
      data3 = [json.loads(line) for line in open(third, 'r')]
      data4 = [json.loads(line) for line in open(fourth, 'r')]
      data5 = [json.loads(line) for line in open(fifth, 'r')]
      
      # combine data
      df1 = pd.DataFrame(data1)
      df2 = pd.DataFrame(data2)
      df3 = pd.DataFrame(data3)
      df4 = pd.DataFrame(data4)
      df5 = pd.DataFrame(data5)
      df = pd.concat([df1, df2, df3, df4, df5])
      by_row_index = df.groupby(df.index)
      df_means = by_row_index.mean()
      # df = df[(df['timesteps_total'] > 7000000) & (df['timesteps_total'] < 10000000)]
      
      # plot data
      print(df_means.iloc[-1])
      df_means.plot(x='timesteps_total', y='episode_reward_mean')
      plt.show()

      '''
      # save plot of learning curve
      fig, ax = plt.subplots(1, 1)
      size = 50
      size = size if len(data['total_reward']) > size else 1
      payoff = pd.Series(data['total_reward'])
      payoff_mean = payoff.rolling(size).mean()
      payoff_std = payoff.rolling(size).std()
      ax.plot(data['step'], data['total_reward'], '.', markersize=1, alpha=0.1, color='C0')
      ax.plot(data['step'], payoff_mean, '-', linewidth=2, label='average reward per step', alpha=1.0, color='C1')
      ax.fill_between(data['step'], payoff_mean - payoff_std, payoff_mean + payoff_std, alpha=0.6, color='C1')
      ax.grid()
      ax.set_xlabel('total number of simulation steps')
      ax.legend()
      plt.tight_layout()
      plt.savefig(os.path.join(data['options']['dirname'], f'learningcurve.png'))
      plt.close()
      '''

if __name__ == "__main__":
      plot_averaged_curves('/Users/rupaln/Documents/uiuc/ray_results/asym/PPO_asymmetric_advantages_tomato_True_nw=2_vf=0.000100_es=0.200000_en=0.100000_kl=0.200000_1_2022-07-11_12-56-04_6tgn7a_/result.json',
                           '/Users/rupaln/Documents/uiuc/ray_results/asym/PPO_asymmetric_advantages_tomato_True_nw=2_vf=0.000100_es=0.200000_en=0.100000_kl=0.200000_1_2022-07-11_14-34-195zgifd78/result.json',
                           '/Users/rupaln/Documents/uiuc/ray_results/asym/PPO_asymmetric_advantages_tomato_True_nw=2_vf=0.000100_es=0.200000_en=0.100000_kl=0.200000_1_2022-07-11_16-38-00n5h9q8ab/result.json',
                           '/Users/rupaln/Documents/uiuc/ray_results/asym/PPO_asymmetric_advantages_tomato_True_nw=2_vf=0.000100_es=0.200000_en=0.100000_kl=0.200000_1_2022-07-11_20-05-46gn6dsxe7/result.json',
                           '/Users/rupaln/Documents/uiuc/ray_results/asym/PPO_asymmetric_advantages_tomato_True_nw=2_vf=0.000100_es=0.200000_en=0.100000_kl=0.200000_1_2022-07-12_00-21-23kmwtl20x/result.json')