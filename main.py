import time

import gymnasium as gym
import numpy as np

def make_simulation(env, nr_of_episodes : int, max_episode_steps : int, learning_rate : float, gamma : float, epsilon : float, epsilon_decay : float, show_animation : bool):
    """
    Q-Table training instance on Cliffwalking gymnasium environment

    Args:
        nr_of_episodes (int):
        max_episode_steps (int): 
        learning_rate (float):
        gamma (float):
        epsilon (float):
        epsilon_decay (float):
        show_animation (bool):
    """
    
    Qtable = np.zeros((4*12, 4))

    output_file = open("single_sim_data.txt", "w")


    for episode_nr in range(0, nr_of_episodes):
        
        if(show_animation):
            print(f"Episode number {episode_nr}:")
        state, _ = env.reset()


        for episode_step_nr in range(0, max_episode_steps):
            single_line = ""
            single_line += str(episode_nr) + ";"
            single_line += str(episode_step_nr) + ";"
            single_line += str(state) + ";"
            if(np.random.random() < epsilon):
                # Exploration -> Random action
                action = np.random.randint(0, 4)
            else:
                # Exploitation -> Action with best value
                action = np.argmax(Qtable[state])

            single_line += str(action) + ";"
            # Next step, taking:
            # 1. New state
            # 2. Value of the new state
            # 3. Is the new state a final state?
            # 4. Did it took more steps, than maximum number specified in gym.make(max_episode_steps=...)?
            next_state, value_reward, is_final_state, is_out_of_steps, _ = env.step(action)

            single_line += str(value_reward) + ";"

            best_next_action = np.argmax(Qtable[next_state])
            Qtable[state, action] += learning_rate * (
                value_reward + gamma * Qtable[next_state, best_next_action] - Qtable[state, action]
            )

            single_line += str(epsilon) + ";"
            single_line += str(Qtable[state, action]) + "\n"
            output_file.write(single_line)

            state = next_state

            if(is_final_state or is_out_of_steps):
                break
        epsilon = max(0.1, epsilon*0.999)
        
    return Qtable

def test_qtable(env, Qtable, max_episode_steps):
    state, _ = env.reset()
    for i in range(max_episode_steps):
        action = np.argmax(Qtable[state])
        next_state, _, is_final_state, is_out_of_steps, _ = env.step(action)
        state = next_state
        if(is_final_state):
            return True
    return False

def automated_test():
    env = gym.make("CliffWalking-v0")

    """
    arr_nr_of_episodes = np.array([100, 1000, 5000])
    arr_max_steps = np.array([13, 14, 15, 30])
    arr_learning_rate = np.array([0.01, 0.1])
    arr_gamma = np.array([0.99])
    arr_epsilon = np.array([0.1, 0.9, 1])
    arr_epsilon_decay = np.array([0.999, 1])
    """

    arr_nr_of_episodes = np.array([1000])
    arr_max_steps = np.array([30])
    arr_learning_rate = np.array([0.1])
    arr_gamma = np.array([0.99])
    arr_epsilon = np.array([0.9])
    arr_epsilon_decay = np.array([0.999])

    test_iterations = 1#5


    for nr_of_episodes in arr_nr_of_episodes:
        #output_file = open(f"{nr_of_episodes}.txt", "w")
        #output_file_string = ""
        #print(f"nr_of_episodes: {nr_of_episodes}")
        for max_steps in arr_max_steps:
            #print(f"max_steps: {max_steps}")
            for learning_rate in arr_learning_rate:
                #print(f"learning_rate: {learning_rate}")
                for gamma in arr_gamma:
                    for epsilon in arr_epsilon:
                        for epsilon_decay in arr_epsilon_decay:
                            successes = 0
                            for i in range(test_iterations):
                                Qtable = make_simulation(env, nr_of_episodes, max_steps, learning_rate, gamma, epsilon, epsilon_decay, False)
                                success = test_qtable(env, Qtable, max_steps)
                                if(success):
                                    successes += 1
                            #output_file_string += f"{nr_of_episodes};{max_steps};{learning_rate};{gamma};{epsilon};{epsilon_decay};{successes}\n"
                            
                        #break
                    #break
                #break
            #break
        #break
        #output_file.write(output_file_string)
        #output_file.close()

if __name__ == "__main__":
    automated_test()