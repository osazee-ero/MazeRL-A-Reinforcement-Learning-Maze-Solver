#%% import libraries
import os
working_dir = "C:/Users/Ero/Desktop/ErosAIPortfolio/QLearning"
os.chdir(working_dir)
import tkinter as tk
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

#%% class Qlearn

class Qlearn:
    
    def __init__(self,goal_reward,goal_reward2,discount_reward,loss_reward,fire_reward,number_of_episodes,epochs,tolerance):
        self.goal_reward = goal_reward
        self.goal_reward2 = goal_reward2
        self.fire_reward = fire_reward
        self.discount_reward = discount_reward
        self.loss_reward = loss_reward
        self.number_of_episodes = number_of_episodes
        self.epochs = epochs
        self.tolerance = tolerance
       
       
        
    def create_env(self, start_index,goal_index, goal2_index, wall_index,fire_index,number_of_states):
         #number_of_states = int(math.sqrt(number_of_states))
         environment_ = -self.loss_reward * np.ones((number_of_states,number_of_states))
         
         goal_row_position, goal_col_position = math.floor(goal_index/number_of_states), goal_index%number_of_states
         goal2_row_position, goal2_col_position = math.floor(goal2_index/number_of_states), goal2_index%number_of_states
         agent_row_position, agent_col_position = math.floor(start_index/number_of_states), start_index%number_of_states
         
         environment_[goal_row_position, goal_col_position] = self.goal_reward
         environment_[goal2_row_position, goal2_col_position] = self.goal_reward2
         for j in wall_index:
              wall_row_position, wall_col_position = math.floor(j/number_of_states), j%number_of_states
              environment_[wall_row_position, wall_col_position] = -1000
         for j in fire_index:
             fire_row_position, fire_col_position = math.floor(j/number_of_states), j%number_of_states
             environment_[fire_row_position, fire_col_position] = -self.fire_reward
         #print(environment_)
         return np.array(environment_),agent_row_position,agent_col_position,goal_row_position,goal_col_position,goal2_row_position,goal2_col_position
     
    
    def agent_observations(self,environment,agent_row_position,agent_col_position):
        
        if(agent_row_position==0 and agent_col_position == 0):
                env = np.array([environment[agent_row_position,agent_col_position],environment[agent_row_position,agent_col_position+1],environment[agent_row_position,agent_col_position],environment[agent_row_position+1,agent_col_position]])
        
        elif(agent_row_position==len(environment)-1 and agent_col_position == 0):
                env = np.array([environment[agent_row_position,agent_col_position],environment[agent_row_position,agent_col_position+1],environment[agent_row_position-1,agent_col_position],environment[agent_row_position,agent_col_position]])
        
        elif(agent_row_position==len(environment)-1 and agent_col_position == len(environment)-1):
                env = np.array([environment[agent_row_position,agent_col_position-1],environment[agent_row_position,agent_col_position],environment[agent_row_position-1,agent_col_position],environment[agent_row_position,agent_col_position]])
                
        elif(agent_row_position==0 and agent_col_position == len(environment)-1):
                env = np.array([environment[agent_row_position,agent_col_position-1],environment[agent_row_position,agent_col_position],environment[agent_row_position,agent_col_position],environment[agent_row_position+1,agent_col_position]])
                
        elif(agent_row_position==0 and (agent_col_position > 0  and agent_col_position < len(environment)-1)):
                env = np.array([environment[agent_row_position,agent_col_position-1],environment[agent_row_position,agent_col_position+1],environment[agent_row_position,agent_col_position],environment[agent_row_position+1,agent_col_position]])
        
        elif(agent_row_position==len(environment)-1 and (agent_col_position > 0 and agent_col_position < len(environment)-1)):
                env = np.array([environment[agent_row_position,agent_col_position-1],environment[agent_row_position,agent_col_position+1],environment[agent_row_position-1,agent_col_position],environment[agent_row_position,agent_col_position]])
                
        elif(agent_col_position==len(environment)-1 and (agent_row_position > 0 and agent_row_position < len(environment)-1)):
                env = np.array([environment[agent_row_position,agent_col_position-1],environment[agent_row_position,agent_col_position],environment[agent_row_position-1,agent_col_position],environment[agent_row_position+1,agent_col_position]])
              
        
        elif(agent_col_position==0 and (agent_row_position > 0 and agent_row_position < len(environment)-1)):
                env = np.array([environment[agent_row_position,agent_col_position],environment[agent_row_position,agent_col_position+1],environment[agent_row_position-1,agent_col_position],environment[agent_row_position+1,agent_col_position]])
                
        else:
                env = np.array([environment[agent_row_position,agent_col_position-1],environment[agent_row_position,agent_col_position+1],environment[agent_row_position-1,agent_col_position],environment[agent_row_position+1,agent_col_position]])
            
              
        return env 
    
    
    
    def agent_action(self, number_of_actions):
        return np.random.randint(0, number_of_actions)
    
    
    def generate_trans_prob(self, number_of_actions):
        number_of_left_actions = 0
        number_of_right_actions = 0
        number_of_up_actions = 0
        number_of_down_actions = 0
        for k in range(0,number_of_actions):
            action = self.agent_action(4)
            if(action == 0):
                number_of_left_actions += 1
            elif(action == 1):
                number_of_right_actions += 1
            elif(action == 2):
                number_of_up_actions += 1
            elif(action == 3):
                number_of_down_actions += 1
        total_outcome = number_of_left_actions + number_of_right_actions + number_of_up_actions + number_of_down_actions 
        trans_proba = np.array([round(number_of_left_actions/total_outcome, 3), round(number_of_right_actions/total_outcome, 3), round(number_of_up_actions/total_outcome, 3), round(number_of_down_actions/total_outcome, 3)])
        return trans_proba  
    
    def generate_trans_proba_matrix(self, number_of_states, number_of_random_actions):
        trans_proba_matrix = list()
        trans_proba_list = list()
        for k in range(0, number_of_states):
            for l in range(0, number_of_states):
                trans_proba = self.generate_trans_prob(number_of_random_actions)
                trans_proba_list.append(trans_proba)
            index = 0
        for i in range(0, number_of_states):
           buttons_ = list()
           for j in range(0, number_of_states):
              buttons_.append(trans_proba_list[index])
              index = index + 1
           trans_proba_matrix.append(buttons_)
        return np.array(trans_proba_matrix)
                
        
    
    # def create_prob_matrix(self,action,number_of_states):
    #     trans_proba = np.zeros((number_of_states,number_of_states))
    #     i=0
    #     for j in range(0, number_of_states):
    #         if(action == "left"):
    #             if(j==0 or j%math.sqrt(number_of_states)==0):
    #                 trans_proba[i,j] = 1
    #             else:
    #                 trans_proba[i,j-1] = 1.
                    
    #             i=i+1
    #         elif(action == "right"):
    #             if(j==number_of_states - 1 or (j+1)%math.sqrt(number_of_states)==0):
    #                 trans_proba[i,j] = 1
    #             else:
    #                 trans_proba[i,j+1] = 1
    #             i=i+1
    #         elif(action == "up"):
    #             if(j < math.sqrt(number_of_states)):
    #                 trans_proba[i,j] = 1
    #             else:
    #                 trans_proba[i,j-math.sqrt(number_of_states)] = 1
    #             i=i+1
    #         elif(action == "down"):
    #             if(j >= (number_of_states - math.sqrt(number_of_states))):
    #                 trans_proba[i,j] = 1
    #             else:
    #                 trans_proba[i,j+math.sqrt(number_of_states)] = 1
    #             i=i+1
    #     return trans_proba
    
    
    # def create_reward_prob(self,action,number_of_states,environment):
        
    #     env = environment.reshape(number_of_states)
    #     reward_proba = np.zeros((number_of_states,number_of_states))
    #     i=0
    #     for j in range(0, number_of_states):
    #         if(action == "left"):
    #             if(j==0 or j%math.sqrt(number_of_states)==0):
    #                 reward_proba[i,j] = env(j)
    #             else:
    #                 reward_proba[i,j-1] = env(j-1)
    #             i=i+1
    #         elif(action == "right"):
    #             if(j==number_of_states - 1 or (j+1)%math.sqrt(number_of_states)==0):
    #                 reward_proba[i,j] = env(j)
    #             else:
    #                 reward_proba[i,j+1] = env(j+1)
    #             i=i+1
    #         elif(action == "up"):
    #             if(j < math.sqrt(number_of_states)):
    #                 reward_proba[i,j] = env(j)
    #             else:
    #                 reward_proba[i,j-math.sqrt(number_of_states)] = env(j-math.sqrt(number_of_states))
    #             i=i+1
    #         elif(action == "down"):
    #             if(j >= (number_of_states - math.sqrt(number_of_states))):
    #                 reward_proba[i,j] = env(j)
    #             else:
    #                 reward_proba[i,j+math.sqrt(number_of_states)] = env(j+math.sqrt(number_of_states))
    #             i=i+1
    #     return reward_proba
    
    def agent_value_observations(self,environment,agent_row_position,agent_col_position):
        
        if(agent_row_position==0 and agent_col_position == 0):
                env = np.array([-1e10000,environment[agent_row_position,agent_col_position+1],-1e10000,environment[agent_row_position+1,agent_col_position]])
        
        elif(agent_row_position==len(environment)-1 and agent_col_position == 0):
                env = np.array([-1e10000,environment[agent_row_position,agent_col_position+1],environment[agent_row_position-1,agent_col_position],-1e10000])
        
        elif(agent_row_position==len(environment)-1 and agent_col_position == len(environment)-1):
                env = np.array([environment[agent_row_position,agent_col_position-1],-1e10000,environment[agent_row_position-1,agent_col_position],-1e10000])
                
        elif(agent_row_position==0 and agent_col_position == len(environment)-1):
                env = np.array([environment[agent_row_position,agent_col_position-1],-1e10000,-1e10000,environment[agent_row_position+1,agent_col_position]])
                
        elif(agent_row_position==0 and (agent_col_position > 0  and agent_col_position < len(environment)-1)):
                env = np.array([environment[agent_row_position,agent_col_position-1],environment[agent_row_position,agent_col_position+1],-1e10000,environment[agent_row_position+1,agent_col_position]])
        
        elif(agent_row_position==len(environment)-1 and (agent_col_position > 0 and agent_col_position < len(environment)-1)):
                env = np.array([environment[agent_row_position,agent_col_position-1],environment[agent_row_position,agent_col_position+1],environment[agent_row_position-1,agent_col_position],-1e10000])
                
        elif(agent_col_position==len(environment)-1 and (agent_row_position > 0 and agent_row_position < len(environment)-1)):
                env = np.array([environment[agent_row_position,agent_col_position-1],-1e10000,environment[agent_row_position-1,agent_col_position],environment[agent_row_position+1,agent_col_position]])
              
        
        elif(agent_col_position==0 and (agent_row_position > 0 and agent_row_position < len(environment)-1)):
                env = np.array([-1e10000,environment[agent_row_position,agent_col_position+1],environment[agent_row_position-1,agent_col_position],environment[agent_row_position+1,agent_col_position]])
                
        else:
                env = np.array([environment[agent_row_position,agent_col_position-1],environment[agent_row_position,agent_col_position+1],environment[agent_row_position-1,agent_col_position],environment[agent_row_position+1,agent_col_position]])
            
              
        return env 
    
    
    def agent_current_position(self,agent_row_position,agent_col_position,action_type,environment):
        if(agent_row_position==0 and agent_col_position == 0):
            if(action_type==0): #left movement
                     agent_row_position,agent_col_position = agent_row_position,agent_col_position
            elif(action_type == 1):
                    agent_row_position,agent_col_position = agent_row_position,agent_col_position+1
            elif(action_type == 2):
                    agent_row_position,agent_col_position = agent_row_position,agent_col_position
            elif(action_type == 3):
                    agent_row_position,agent_col_position = agent_row_position+1,agent_col_position
        
        elif(agent_row_position==len(environment)-1 and agent_col_position == 0):
            if(action_type==0): #left movement
                    agent_row_position,agent_col_position = agent_row_position,agent_col_position
            elif(action_type == 1):
                    agent_row_position,agent_col_position = agent_row_position,agent_col_position+1
            elif(action_type == 2):
                    agent_row_position,agent_col_position = agent_row_position-1,agent_col_position
            elif(action_type == 3):
                    agent_row_position,agent_col_position = agent_row_position,agent_col_position
        
        elif(agent_row_position==len(environment)-1 and agent_col_position == len(environment)-1):
            if(action_type==0): #left movement
                    agent_row_position,agent_col_position = agent_row_position,agent_col_position-1
            elif(action_type == 1):
                    agent_row_position,agent_col_position = agent_row_position,agent_col_position
            elif(action_type == 2):
                    agent_row_position,agent_col_position = agent_row_position-1,agent_col_position
            elif(action_type == 3):
                    agent_row_position,agent_col_position = agent_row_position,agent_col_position

                
        elif(agent_row_position==0 and agent_col_position == len(environment)-1):
            if(action_type==0): #left movement
                    agent_row_position,agent_col_position = agent_row_position,agent_col_position-1
            elif(action_type == 1):
                    agent_row_position,agent_col_position = agent_row_position,agent_col_position
            elif(action_type == 2):
                    agent_row_position,agent_col_position = agent_row_position,agent_col_position
            elif(action_type == 3):
                    agent_row_position,agent_col_position = agent_row_position+1,agent_col_position
                
        elif(agent_row_position==0 and (agent_col_position > 0  and agent_col_position < len(environment)-1)):
            if(action_type==0): #left movement
                    agent_row_position,agent_col_position = agent_row_position,agent_col_position-1
            elif(action_type == 1):
                    agent_row_position,agent_col_position = agent_row_position,agent_col_position+1
            elif(action_type == 2):
                    agent_row_position,agent_col_position = agent_row_position,agent_col_position
            elif(action_type == 3):
                    agent_row_position,agent_col_position = agent_row_position+1,agent_col_position
        
        elif(agent_row_position==len(environment)-1 and (agent_col_position > 0 and agent_col_position < len(environment)-1)):
            if(action_type==0): #left movement
                    agent_row_position,agent_col_position = agent_row_position,agent_col_position-1
            elif(action_type == 1):
                    agent_row_position,agent_col_position = agent_row_position,agent_col_position+1
            elif(action_type == 2):
                    agent_row_position,agent_col_position = agent_row_position-1,agent_col_position
            elif(action_type == 3):
                    agent_row_position,agent_col_position = agent_row_position,agent_col_position
                
        elif(agent_col_position==len(environment)-1 and (agent_row_position > 0 and agent_row_position < len(environment)-1)):
            if(action_type==0): #left movement
                    agent_row_position,agent_col_position = agent_row_position,agent_col_position-1
            elif(action_type == 1):
                    agent_row_position,agent_col_position = agent_row_position,agent_col_position
            elif(action_type == 2):
                    agent_row_position,agent_col_position = agent_row_position-1,agent_col_position
            elif(action_type == 3):
                    agent_row_position,agent_col_position = agent_row_position+1,agent_col_position
              
        
        elif(agent_col_position==0 and (agent_row_position > 0 and agent_row_position < len(environment)-1)):
            if(action_type==0): #left movement
                    agent_row_position,agent_col_position = agent_row_position,agent_col_position
            elif(action_type == 1):
                    agent_row_position,agent_col_position = agent_row_position,agent_col_position+1
            elif(action_type == 2):
                    agent_row_position,agent_col_position = agent_row_position-1,agent_col_position
            elif(action_type == 3):
                    agent_row_position,agent_col_position = agent_row_position+1,agent_col_position
                
        else:
            if(action_type==0): #left movement
                    agent_row_position,agent_col_position = agent_row_position,agent_col_position-1
            elif(action_type == 1):
                    agent_row_position,agent_col_position = agent_row_position,agent_col_position+1
            elif(action_type == 2):
                    agent_row_position,agent_col_position = agent_row_position-1,agent_col_position
            elif(action_type == 3):
                    agent_row_position,agent_col_position = agent_row_position+1,agent_col_position
        
        return agent_row_position,agent_col_position


    def value_learning(self,environment,number_of_states,number_of_random_actions,goal_row_position,goal_col_position,goal2_row_position,goal2_col_position):
        
        #optimal_actions = np.zeros_like(environment)
        optimal_trans_proba = list()
        episodes_return = list()
        value_of_states = np.zeros_like(environment)
        
        
        for k in range(0, self.number_of_episodes):
            trans_proba_matrix_left = self.generate_trans_proba_matrix(number_of_states, number_of_random_actions)
            trans_proba_matrix_right = self.generate_trans_proba_matrix(number_of_states, number_of_random_actions)
            trans_proba_matrix_up = self.generate_trans_proba_matrix(number_of_states, number_of_random_actions)
            trans_proba_matrix_down = self.generate_trans_proba_matrix(number_of_states, number_of_random_actions)
            previous_value_of_states = 0

            for l in range(0, self.epochs):
                for i in range(0,number_of_states):
                    for j in range(0, number_of_states):
                        
                        if((i==goal_row_position and j==goal_col_position) or (i==goal2_row_position and j==goal2_col_position) ):
                            continue
                        else:
                            observations = self.agent_observations(environment,i,j)
                            value_left = round(np.dot(trans_proba_matrix_left[i,j],np.add(observations, self.discount_reward * value_of_states[i,j])),3)
                            value_right = round(np.dot(trans_proba_matrix_right[i,j],np.add(observations, self.discount_reward * value_of_states[i,j])),3)
                            value_up = round(np.dot(trans_proba_matrix_up[i,j],np.add(observations, self.discount_reward * value_of_states[i,j])),3)
                            value_down = round(np.dot(trans_proba_matrix_down[i,j],np.add(observations, self.discount_reward * value_of_states[i,j])),3)
                            values= np.array([value_left,value_right,value_up,value_down])
                            value_of_states[i,j] = np.max(values)
                            #optimal_actions[i,j] = np.argmax(values)
                if(l > 1):
                    error =  np.fabs(value_of_states-previous_value_of_states)
                    sum_error =np.sum(error)
                    print(round(sum_error,3))

                    if(sum_error <= self.tolerance):
                        break
                    
                previous_value_of_states = value_of_states.copy()
            print(value_of_states)
        return value_of_states
            #print(optimal_actions)
            #print(f"episode: {l} completed")
       
                
            
        
    def agent_navigate(self,agent_row_position,agent_col_position,goal_row_position,goal_col_position,goal2_row_position,goal2_col_position,number_of_states,environment,value_of_states,buttons,root):
            
           
            reshape_buttons = list()
            index = 0
            for i in range(0, number_of_states):
                buttons_ = list()
                for j in range(0, number_of_states):
                   buttons_.append(buttons[index])
                   index = index + 1
                reshape_buttons.append(buttons_)
                    
            while((agent_row_position is not goal_row_position and agent_col_position is not goal_col_position) or (agent_row_position is not goal2_row_position and agent_col_position is not goal2_col_position)):
                observ = self.agent_value_observations(value_of_states,agent_row_position,agent_col_position)
                actions  = np.argwhere(observ  == np.max( observ ))
                actions = actions.flatten()
                print(actions)
                if(len(actions) > 1):
                    rand_action = self.agent_action(len(actions))
                    select_action = actions[rand_action]
                else:
                    select_action = actions[0]
                    
                agent_row_position,agent_col_position = self.agent_current_position(agent_row_position,agent_col_position,select_action,environment)
                #print(agent_row_position,agent_col_position)
                reshape_buttons[agent_row_position][agent_col_position].config(bg="#B10DC9")
                root.update()
                if((agent_row_position == goal_row_position and agent_col_position == goal_col_position) or (agent_row_position == goal2_row_position and agent_col_position == goal2_col_position)):
                    break
              
                
    def reset_colors(self,start_index,goal_index,goal2_index,wall_index,fire_index,buttons,root):
        
        for i in range(0, len(buttons)):
            buttons[i].config(bg="white")
            root.update()
        buttons[start_index].config(bg="#B10DC9")
        root.update()
        buttons[goal_index].config(bg="green")
        root.update()
        buttons[goal2_index].config(bg="green")
        root.update()
        for k in wall_index:
            buttons[k].config(bg="black")
            root.update()
        for j in fire_index:
            buttons[j].config(bg="red")
            root.update()
        
               
    
    
    def explore_with_val_func(self,goal_index, goal2_index, start_index,wall_index, fire_index,number_of_states,buttons,root,number_of_random_actions):
        number_of_states = int(math.sqrt(number_of_states))
        self.reset_colors(start_index,goal_index,goal2_index,wall_index,fire_index,buttons,root)
        environment,agent_row_position,agent_col_position,goal_row_position,goal_col_position,goal2_row_position,goal2_col_position = self.create_env(start_index,goal_index, goal2_index, wall_index,fire_index,number_of_states)
        
        value_of_states = self.value_learning(environment,number_of_states,number_of_random_actions,goal_row_position,goal_col_position,goal2_row_position,goal2_col_position)
        self.agent_navigate(agent_row_position,agent_col_position,goal_row_position,goal_col_position,goal2_row_position,goal2_col_position,number_of_states,environment,value_of_states,buttons,root)
        
        #average_returns = np.sum(np.array(episodes_return))/len(episodes_return)
        #print(optimal_rewards)
        #print(optimal_actions)
        #print(average_returns)
        return value_of_states
        
   
        
        
   
      
        
       
   
        
        
     
        
     
    
        
        
               
    
        
        
        
        
        
        
        
        
        