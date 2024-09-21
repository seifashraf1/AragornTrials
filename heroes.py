import numpy as np

class Heroes: ## The Fellowship class 
    def __init__(self,
                 total_quests: int = 2000,
                 true_probability_list: list = [0.4, 0.6]):
        """
        Initialize the Heroes class with a list of true success probabilities and the total number of quests.
        
        :param total_quests: Total number of quests to be performed.
        :param true_probability_list: List of true success probabilities for each hero.
        """
        self.heroes = [{
            'name': f"Hero_{i+1}",            # hero's name
            'true_success_probability': p,    # hero's true success probabilities
            'successes': 0,
            'n_quests': 0                     # hero's total number of quests
        } for i, p in enumerate(true_probability_list)]
        self.total_quests = total_quests

    def init_heroes(self):
        """
        Initialize the heroes' performance for a new simulation.
        """
        for hero in self.heroes:
            hero['successes'] = 0
            hero['n_quests'] = 0

    def attempt_quest(self, hero_index: int):
        """
        Attempt a single quest for a specified hero and update their performance.
        (This should be equivalent to pulling a single arm from a multi-armed bandit.)

        Make sure to update the number of quests and the number of successes for the specified hero.
        
        :param hero_index: Index of the hero in the self.heroes list.
        :return: Reward of the quest (1 for success, 0 for failure).
        """
        if hero_index < 0 or hero_index >= len(self.heroes):
            raise IndexError("Hero index out of range.")
        
        hero = self.heroes[hero_index]

        """
        the probability of the hero succeeding in this quest is assumed to be random, if the random number is less than the actual true success probability, the quest is considered a success.
        why? because if it's within the true success probability then it means the hero succeeded, and if its more then it means the hero failed as he can't have more than p_i success probability
        """
        success = np.random.rand() < hero['true_success_probability']       

        hero['n_quests'] += 1

        if success:
            hero['successes'] += 1
        
        if success:
            reward = 1
        else:
            reward = 0

        return reward
