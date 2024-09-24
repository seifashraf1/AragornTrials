from typing import Tuple, List
import numpy as np
from heroes import Heroes
from helpers import run_trials, save_results_plots

def ucb(
    heroes: Heroes, 
    c: float, 
    init_value: float = .0
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Perform Upper Confidence Bound (UCB) action selection for a bandit problem.

    :param heroes: A bandit problem, instantiated from the Heroes class.
    :param c: The exploration coefficient that balances exploration vs. exploitation.
    :param init_value: Initial estimation of each hero's value.
    :return: 
        - rew_record: The record of rewards at each timestep.
        - avg_ret_record: TThe average of rewards up to step t. For example: If 
    we define `ret_T` = \sum^T_{t=0}{r_t}, `avg_ret_record` = ret_T / (1+T).
        - tot_reg_record: The total regret up to step t.
        - opt_action_record: Percentage of optimal actions selected.
    """

    num_heroes = len(heroes.heroes)
    values = [init_value] * num_heroes   # Initial action values
    rew_record = []                      # Rewards at each timestep
    avg_ret_record = []                  # Average reward up to each timestep
    tot_reg_record = []                  # Total regret up to each timestep
    opt_action_record = []               # Percentage of optimal actions selected
    
    total_rewards = 0
    total_regret = 0
    optimal_action_count = 0

    optimal_hero_index = np.argmax([hero['true_success_probability'] for hero in heroes.heroes])
    optimal_reward = heroes.heroes[optimal_hero_index]['true_success_probability']
    
    #counts = how many times the hero was selected, init to 0
    counts = [0] * num_heroes

    for t in range(heroes.total_quests):
        ucb_values = []
        
        #compute ucb value for heroes
        for i in range(num_heroes):
            if counts[i] == 0:
                #choose hero that we didnt select before
                ucb_value = float('inf')
            else:
                #or calculate the ucb value through the formula
                ucb_value = values[i] + c * np.sqrt(np.log(t + 1) / counts[i])
            
            ucb_values.append(ucb_value)

        #select and get reward of the hero with max ucb
        selected_hero = np.argmax(ucb_values)
        reward = heroes.attempt_quest(selected_hero)
        rew_record.append(reward)
        
        counts[selected_hero] += 1
        values[selected_hero] += (reward - values[selected_hero]) / counts[selected_hero]
        
        #update total reward and regret
        total_rewards += reward
        total_regret += (optimal_reward - reward)
        
        avg_ret_record.append(total_rewards / (t + 1))
        tot_reg_record.append(total_regret)

        if selected_hero == optimal_hero_index:
            optimal_action_count += 1

        opt_action_record.append(optimal_action_count / (t + 1))
        

    return rew_record, avg_ret_record, tot_reg_record, opt_action_record

if __name__ == "__main__":
    # Define the bandit problem
    heroes = Heroes(total_quests=3000, true_probability_list=[0.35, 0.6, 0.1])

    # Test various c values
    c_values = [0.0, 0.5, 2.0]
    results_list = []
    for c in c_values:
        rew_rec, avg_ret_rec, tot_reg_rec, opt_act_rec = run_trials(30, 
                                                                    heroes=heroes, bandit_method=ucb, 
                                                                    c=c, init_value=0.0)
        
        results_list.append({
            'exp_name': f'c={c}',
            'reward_rec': rew_rec,
            'average_rew_rec': avg_ret_rec,
            'tot_reg_rec': tot_reg_rec,
            'opt_action_rec': opt_act_rec
        })

    save_results_plots(results_list, plot_title='UCB Experiment Results On Various C Values', 
                       results_folder='results', pdf_name='ucb_various_c_values.pdf')
