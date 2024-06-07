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
    
    def agent_current_position(self,action_type,agent_row_position,agent_col_position,environment):
        if(action_type == 0):
            agent_row_position,agent_col_position = agent_row_position,agent_col_position-1
        elif(action_type == 1):
            agent_row_position,agent_col_position = agent_row_position,agent_col_position+1
        elif(action_type == 2):
            agent_row_position,agent_col_position = agent_row_position-1,agent_col_position
        elif(action_type == 3):
            agent_row_position,agent_col_position = agent_row_position+1,agent_col_position
        if(agent_row_position > len(environment)-1):
            agent_row_position = agent_row_position - 1
        if(agent_row_position < 0):
            agent_row_position = 0
        if(agent_col_position > len(environment)-1):
            agent_col_position = agent_col_position - 1
        if(agent_col_position < 0):
            agent_col_position = 0
        return agent_row_position, agent_col_position
            
    

    def agent_action(self, number_of_actions):
        return np.random.randint(0, number_of_actions)
    
    def label_states(self, environment):
        states = np.zeros_like(environment)
        index = 0
        for i in range(0, len(environment)):
            for j in range(0, len(environment)):
                states[i,j] = index
                index += 1
        return states
    
    
    # def play_random_actions(self,states,number_of_random_actions,environment,agent_row_position,agent_col_position,goal_row_position,goal_col_position,goal2_row_position,goal2_col_position):
    #     observations = list()
    #     for k in range(0, number_of_random_actions):
    #         if((agent_row_position == goal_row_position and agent_col_position == goal_col_position) or (agent_row_position == goal2_row_position and agent_col_position == goal2_col_position)):
    #             continue
    #         else:
    #             current_state =states[agent_row_position,agent_col_position]
    #             action = self.agent_action(4)
    #             agent_row_position,agent_col_position = self.agent_current_position(action,agent_row_position,agent_col_position,environment)
    #             rewards = environment[agent_row_position,agent_col_position]
    #             next_state = states[agent_row_position,agent_col_position]
    #             observations.append((current_state,next_state,action,rewards))
    #     return observations
        
    
   
    
    
    def play_episodes(self,environment,agent_row_position,agent_col_position,goal_row_position,goal_col_position,goal2_row_position,goal2_col_position):
        episodes = list()
        states = self.label_states(environment)
        start_row_position, start_col_position = agent_row_position,agent_col_position
        
        for i in range(0, self.number_of_episodes):
            index = 1
            agent_row_position,agent_col_position =  start_row_position, start_col_position
            action = self.agent_action(4)
            episode = list()
            while(index > 0):
                current_state =states[agent_row_position,agent_col_position]
                if(index > 1):
                    if(action == 0):
                        action = self.agent_action(4)
                        while(action==1):
                            action = self.agent_action(4)
                    elif(action == 1):
                        action = self.agent_action(4)
                        while(action==0):
                            action = self.agent_action(4)
                    elif(action == 2):
                        action = self.agent_action(4)
                        while(action==3):
                            action = self.agent_action(4)
                    elif(action == 3):
                        action = self.agent_action(4)
                        while(action==2):
                            action = self.agent_action(4)
                agent_row_position,agent_col_position = self.agent_current_position(action,agent_row_position,agent_col_position,environment)
                rewards = environment[agent_row_position,agent_col_position]
                next_state = states[agent_row_position,agent_col_position]
                episode.append([current_state,next_state,action,rewards])
                if((agent_row_position == goal_row_position and agent_col_position == goal_col_position) or (agent_row_position == goal2_row_position and agent_col_position == goal2_col_position)):
                    break
                index += 1
            episodes.append(episode)
        return episodes
    
    def generate_trans_proba_matrix(self,observations):
        trans_proba = list()
        observ= list()
        visited_states = list()
        number_of_left_actions = 0
        number_of_right_actions = 0
        number_of_up_actions = 0
        number_of_down_actions = 0
        reward_left = 0
        reward_right = 0
        reward_up = 0
        reward_down = 0
        for i in observations:  
            for j in i:
                observ.append([j[0],j[2],j[3]])
                
                
        for i in observ:
            if(i[0] not in visited_states):
                for j in observ:
                    if i[0] == j[0]:
                        if(j[1] == 0):
                            number_of_left_actions += 1
                            reward_left = j[2]
                        elif(j[1]==1):
                            number_of_right_actions += 1
                            reward_right = j[2]
                        elif(j[1]==2):
                            number_of_up_actions += 1
                            reward_up = j[2]
                        elif(j[1]==3):
                            number_of_down_actions += 1
                            reward_down = j[2]
                total_proba = number_of_left_actions + number_of_right_actions + number_of_up_actions + number_of_down_actions
                state_trans_proba = [number_of_left_actions/total_proba,number_of_right_actions/total_proba, number_of_up_actions/total_proba, number_of_down_actions/total_proba]
                state_rewards = [reward_left,reward_right,reward_up,reward_down]
                trans_proba.append([i[0],state_trans_proba,state_rewards])
            visited_states.append(i[0])
                
        return trans_proba
    
    
        
    
    def trans_proba_matrix(self,environment,agent_row_position,agent_col_position,goal_row_position,goal_col_position,goal2_row_position,goal2_col_position):
       
        episodes = self.play_episodes(environment,agent_row_position,agent_col_position,goal_row_position,goal_col_position,goal2_row_position,goal2_col_position)
        trans_proba = self.generate_trans_proba_matrix(episodes)
        return trans_proba
            


    def value_learning(self,agent_row_position,agent_col_position,environment,number_of_states,goal_row_position,goal_col_position,goal2_row_position,goal2_col_position):
        
        #optimal_actions = np.zeros_like(environment)
        
        value_of_states = np.zeros(number_of_states * number_of_states)
        previous_value_of_states = 0
        trans_proba= self.trans_proba_matrix(environment,agent_row_position,agent_col_position,goal_row_position,goal_col_position,goal2_row_position,goal2_col_position)
        
        for k in range(0, self.epochs):
            for i in trans_proba:
                values = round(np.dot(i[1],np.add(i[2], self.discount_reward * value_of_states[int(i[0])])),3)
                values= np.array(values)
                value_of_states[int(i[0])] = np.max(values)
                            #optimal_actions[i,j] = np.argmax(values)
            if(k > 1):
                error =  np.fabs(value_of_states-previous_value_of_states)
                sum_error =np.sum(error)
               

                if(sum_error <= self.tolerance):
                    break
                
            previous_value_of_states = value_of_states.copy()
        
        return value_of_states
            #print(optimal_actions)
            #print(f"episode: {l} completed")
    def train_agent(self,start_index,goal_index,goal2_index,wall_index,fire_index,agent_row_position,agent_col_position,goal_row_position,goal_col_position,goal2_row_position,goal2_col_position,number_of_states,environment,value_of_states,buttons,root):
        reshape_buttons = list()
        index = 0
        start_row_position, start_col_position =  agent_row_position,agent_col_position
        previous_optimal_action =list()
        prev_episodes_return = 0
        for i in range(0, number_of_states):
            buttons_ = list()
            for j in range(0, number_of_states):
               buttons_.append(buttons[index])
               index = index + 1
            reshape_buttons.append(buttons_)
        for i in range(0, 10):
            self.reset_colors(start_index,goal_index,goal2_index,wall_index,fire_index,buttons,root)
            agent_row_position,agent_col_position =  start_row_position, start_col_position
            episode_reward = 0
            optimal_actions = list()
            while((agent_row_position is not goal_row_position and agent_col_position is not goal_col_position) or (agent_row_position is not goal2_row_position and agent_col_position is not goal2_col_position)):
                observ = self.agent_observations(value_of_states,agent_row_position,agent_col_position)
                actions  = np.argwhere(observ  == np.max( observ ))
                actions = actions.flatten()
                if(len(actions) > 1):
                    rand_action = self.agent_action(len(actions))
                    select_action = actions[rand_action]
                else:
                    select_action = actions[0]
                optimal_actions.append(select_action)
                agent_row_position,agent_col_position = self.agent_current_position(agent_row_position,agent_col_position,select_action,environment)
                episode_reward += value_of_states[agent_row_position,agent_col_position]
                #print(agent_row_position,agent_col_position)
                reshape_buttons[agent_row_position][agent_col_position].config(bg="#B10DC9")
                root.update()
                if((agent_row_position == goal_row_position and agent_col_position == goal_col_position) or (agent_row_position == goal2_row_position and agent_col_position == goal2_col_position)):
                     break
            print(episode_reward)
            if(episode_reward >= prev_episodes_return):
                prev_episodes_return = episode_reward
                previous_optimal_action = optimal_actions
            
           # print(prev_episodes_return)
            #print(previous_optimal_action)
        self.reset_colors(start_index,goal_index,goal2_index,wall_index,fire_index,buttons,root)
                
         
                
            
        
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
                observ = self.agent_observations(value_of_states,agent_row_position,agent_col_position)
                actions  = np.argwhere(observ  == np.max( observ ))
                actions = actions.flatten()
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
        
               
    
    
    def explore_with_val_func(self,goal_index, goal2_index, start_index,wall_index, fire_index,number_of_states,buttons,root):
        number_of_states = int(math.sqrt(number_of_states))
        self.reset_colors(start_index,goal_index,goal2_index,wall_index,fire_index,buttons,root)
        environment,agent_row_position,agent_col_position,goal_row_position,goal_col_position,goal2_row_position,goal2_col_position = self.create_env(start_index,goal_index, goal2_index, wall_index,fire_index,number_of_states)
        
        value_of_states = self.value_learning(agent_row_position,agent_col_position,environment,number_of_states,goal_row_position,goal_col_position,goal2_row_position,goal2_col_position)
        
        value_of_states = value_of_states.reshape(number_of_states,number_of_states)
        value_of_states[goal_row_position,goal_col_position] = self.goal_reward
        value_of_states[goal2_row_position,goal2_col_position] = self.goal_reward2
        self.train_agent(start_index,goal_index,goal2_index,wall_index,fire_index,agent_row_position,agent_col_position,goal_row_position,goal_col_position,goal2_row_position,goal2_col_position,number_of_states,environment,value_of_states,buttons,root)
        #self.agent_navigate(agent_row_position,agent_col_position,goal_row_position,goal_col_position,goal2_row_position,goal2_col_position,number_of_states,environment,value_of_states,buttons,root)
        print(value_of_states)
        #average_returns = np.sum(np.array(episodes_return))/len(episodes_return)
        #print(optimal_rewards)
        #print(optimal_actions)
        #print(average_returns)
        return value_of_states
        
   
        
        
   
      
        
       
   
        
        
     
        
     
    
        
        
               
    
        
        
        
        
        
        
        
        
        