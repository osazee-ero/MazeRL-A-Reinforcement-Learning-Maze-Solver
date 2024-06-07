#%% import libraries
import os
# working_dir = "C:/Users/Ero/Desktop/ErosAIPortfolio/QLearning"
# os.chdir(working_dir)
import tkinter as tk
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

#%% class Qlearn

class Qlearn:
    
    def __init__(self,goal_reward,goal_reward2,discount_reward,loss_reward,fire_reward,number_of_episodes,epochs,tolerance,learning_rate):
        self.goal_reward = goal_reward
        self.goal_reward2 = goal_reward2
        self.fire_reward = fire_reward
        self.discount_reward = discount_reward
        self.loss_reward = loss_reward
        self.number_of_episodes = number_of_episodes
        self.epochs = epochs
        self.tolerance = tolerance
        self.learning_rate  = learning_rate
       
       
        
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
              environment_[wall_row_position, wall_col_position] = -10000
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
    
    
    def check_available_actions(self,environment,agent_row_position,agent_col_position):
        
        if(agent_row_position==0 and agent_col_position == 0):
             return np.array([1,3])
        
        elif(agent_row_position==len(environment)-1 and agent_col_position == 0):
             return np.array([1,2])
        
        elif(agent_row_position==len(environment)-1 and agent_col_position == len(environment)-1):
             return np.array([0,2])

        elif(agent_row_position==0 and agent_col_position == len(environment)-1):
             return np.array([0,3])                
        
        elif(agent_row_position==0 and (agent_col_position > 0  and agent_col_position < len(environment)-1)):
             return np.array([0,1,3])   
         
        elif(agent_row_position==len(environment)-1 and (agent_col_position > 0 and agent_col_position < len(environment)-1)):
             return np.array([0,1,2])  
              
        elif(agent_col_position==len(environment)-1 and (agent_row_position > 0 and agent_row_position < len(environment)-1)):
             return np.array([0,2,3])              
        
        elif(agent_col_position==0 and (agent_row_position > 0 and agent_row_position < len(environment)-1)):
             return np.array([1,2,3])                
        else:
             return np.array([0,1,2,3])            
              
    
    
    
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
            


    def value_learning(self,environment,number_of_states,goal_row_position,goal_col_position,goal2_row_position,goal2_col_position):
        
        values_of_states = np.zeros((number_of_states,number_of_states))
        previous_value_of_states = 0
        value_of_states_left = np.zeros((number_of_states,number_of_states))
        value_of_states_right = np.zeros((number_of_states,number_of_states))
        value_of_states_up = np.zeros((number_of_states,number_of_states))
        value_of_states_down = np.zeros((number_of_states,number_of_states))
        
        for k in range(0, self.epochs):
            for i in range(0, len(environment)):
                for j in range(0, len(environment)):
                    if((i==goal_row_position and j==goal_col_position) or (i==goal2_row_position and j==goal2_col_position)):
                        continue
                    else:
                       
                        rewards = self.agent_observations(environment,i,j)
                        value_of_states_left_ = round(rewards[0] + self.discount_reward * values_of_states[i,j],2)
                        value_of_states_right_ = round(rewards[1] + self.discount_reward *  values_of_states[i,j],2)
                        value_of_states_up_ = round(rewards[2] + self.discount_reward * values_of_states[i,j],2)
                        value_of_states_down_ = round(rewards[3] + self.discount_reward * values_of_states[i,j],2)
                        
                      
                        values_of_states[i,j] = np.max(np.array([value_of_states_left_,value_of_states_right_,value_of_states_up_,value_of_states_down_]))
                    
                    
                            #optimal_actions[i,j] = np.argmax(values)
            if(k > 1):
                error =  np.fabs(values_of_states-previous_value_of_states)
                sum_error =np.sum(error)
                print(sum_error)
               

                if(sum_error <= self.tolerance):
                    for i in range(0, len(environment)):
                        for j in range(0, len(environment)):
                          
                            rewards = self.agent_observations(environment,i,j)
                            value_of_states_left[i,j] = round(rewards[0] + self.discount_reward * values_of_states[i,j],2)
                            value_of_states_right[i,j] = round(rewards[1] + self.discount_reward *  values_of_states[i,j],2)
                            value_of_states_up[i,j] = round(rewards[2] + self.discount_reward * values_of_states[i,j],2)
                            value_of_states_down[i,j] = round(rewards[3] + self.discount_reward * values_of_states[i,j],2)
                    break
                
            previous_value_of_states = values_of_states.copy()
        print(value_of_states_left)
        print(value_of_states_right)
        print(value_of_states_up)
        print(value_of_states_down)
        return value_of_states_left, value_of_states_right, value_of_states_up, value_of_states_down
    
    
    def td_value_learning(self,environment,number_of_states,goal_row_position,goal_col_position,goal2_row_position,goal2_col_position):
       
        value_of_states_left = np.zeros((number_of_states,number_of_states))
        value_of_states_right = np.zeros((number_of_states,number_of_states))
        value_of_states_up = np.zeros((number_of_states,number_of_states))
        value_of_states_down = np.zeros((number_of_states,number_of_states))
        values_of_states = np.zeros((number_of_states,number_of_states))
        previous_value_of_states = 0
        
        
        for k in range(0, self.epochs):
            for i in range(0, number_of_states):
                for j in range(0, number_of_states):
                     rewards = self.agent_observations(environment,i,j)
                     next_states_left = self.agent_observations(value_of_states_left,i,j)
                     next_states_right = self.agent_observations(value_of_states_right,i,j)
                     next_states_up = self.agent_observations(value_of_states_up,i,j)
                     next_states_down = self.agent_observations(value_of_states_down,i,j)
                     value_of_states_left[i,j] = round(values_of_states[i,j] + self.learning_rate *(rewards[0] + self.discount_reward * next_states_left[0] - values_of_states[i,j]),4)
                     value_of_states_right[i,j] = round(values_of_states[i,j] + self.learning_rate *(rewards[1] + self.discount_reward * next_states_right[1] - values_of_states[i,j]),4)
                     value_of_states_up[i,j] = round(values_of_states[i,j] + self.learning_rate *(rewards[2] + self.discount_reward * next_states_up[2] - values_of_states[i,j]),4)        
                     value_of_states_down[i,j] = round(values_of_states[i,j] + self.learning_rate *(rewards[3] + self.discount_reward * next_states_down[3] - values_of_states[i,j]),4)

                     values_of_states[i,j] = np.max(np.array([value_of_states_left[i,j],value_of_states_right[i,j],value_of_states_up[i,j],value_of_states_down[i,j]]))

                      
            if(k > 1):
                    error =  np.fabs(values_of_states-previous_value_of_states)
                    sum_error =np.sum(error)
                    #print(round(sum_error,3))
                    if(sum_error <= self.tolerance):
                        break
                
            previous_value_of_states = values_of_states.copy()
       

        return value_of_states_left, value_of_states_right, value_of_states_up, value_of_states_down
    
    
    def td_Q_learning(self,environment,number_of_states,goal_row_position,goal_col_position,goal2_row_position,goal2_col_position):
       
        Qvalue_of_states_left = np.zeros((number_of_states,number_of_states))
        Qvalue_of_states_right = np.zeros((number_of_states,number_of_states))
        Qvalue_of_states_up = np.zeros((number_of_states,number_of_states))
        Qvalue_of_states_down = np.zeros((number_of_states,number_of_states))
        #Qvalues_of_states = np.zeros((number_of_states,number_of_states))
        previous_Qvalue_of_states = 0
        
        
        for k in range(0, self.epochs):
            for i in range(0, number_of_states):
                for j in range(0, number_of_states):
                     rewards = self.agent_observations(environment,i,j)
                     agent_row,agent_col = i,j
                     row,col = self.agent_current_position(0,agent_row,agent_col,environment)  
                     next_states_left = np.max(np.array([Qvalue_of_states_left[row,col],Qvalue_of_states_right[row,col],Qvalue_of_states_up[row,col],Qvalue_of_states_down[row,col]]))
                     
                     row,col = self.agent_current_position(1,agent_row,agent_col,environment)
                     next_states_right = np.max(np.array([Qvalue_of_states_left[row,col],Qvalue_of_states_right[row,col],Qvalue_of_states_up[row,col],Qvalue_of_states_down[row,col]]))
                     
                     row,col = self.agent_current_position(2,agent_row,agent_col,environment)
                     next_states_up = np.max(np.array([Qvalue_of_states_left[row,col],Qvalue_of_states_right[row,col],Qvalue_of_states_up[row,col],Qvalue_of_states_down[row,col]]))
                     
                     row,col = self.agent_current_position(3,agent_row,agent_col,environment)
                     next_states_down = np.max(np.array([Qvalue_of_states_left[row,col],Qvalue_of_states_right[row,col],Qvalue_of_states_up[row,col],Qvalue_of_states_down[row,col]]))
                     
                     Qvalue_of_states_left[i,j] = round(Qvalue_of_states_left[i,j] + self.learning_rate *(rewards[0] + self.discount_reward * next_states_left - Qvalue_of_states_left[i,j]),4)
                     Qvalue_of_states_right[i,j] = round(Qvalue_of_states_right[i,j] + self.learning_rate *(rewards[1] + self.discount_reward * next_states_right - Qvalue_of_states_right[i,j]),4)
                     Qvalue_of_states_up[i,j] = round(Qvalue_of_states_up[i,j] + self.learning_rate *(rewards[2] + self.discount_reward * next_states_up - Qvalue_of_states_up[i,j]),4)        
                     Qvalue_of_states_down[i,j] = round(Qvalue_of_states_down[i,j] + self.learning_rate *(rewards[3] + self.discount_reward * next_states_down - Qvalue_of_states_down[i,j]),4)

                     # Qvalues_of_states[i,j] = np.max(np.array([value_of_states_left[i,j],value_of_states_right[i,j],value_of_states_up[i,j],value_of_states_down[i,j]]))

                      
            # if(k > 1):
            #         error =  np.fabs(values_of_states-previous_value_of_states)
            #         sum_error =np.sum(error)
            #         #print(round(sum_error,3))
            #         if(sum_error <= self.tolerance):
            #             break
                
            # previous_value_of_states = values_of_states.copy()
       

        return Qvalue_of_states_left, Qvalue_of_states_right, Qvalue_of_states_up, Qvalue_of_states_down
    def Q_learning(self,environment,number_of_states,goal_row_position,goal_col_position,goal2_row_position,goal2_col_position):
        
        Qvalue_of_states_left = np.zeros((number_of_states,number_of_states))
        Qvalue_of_states_right = np.zeros((number_of_states,number_of_states))
        Qvalue_of_states_up = np.zeros((number_of_states,number_of_states))
        Qvalue_of_states_down = np.zeros((number_of_states,number_of_states))
        Qvalue_of_states = np.zeros((number_of_states,number_of_states))
        previous_Qvalue_of_states = 0
        
        for k in range(0, self.epochs):
            for i in range(0, len(environment)):
                for j in range(0, len(environment)):
                    if((i==goal_row_position and j==goal_col_position) or (i==goal2_row_position and j==goal2_col_position)):
                        continue
                    else:
                       
                        rewards = self.agent_observations(environment,i,j)
                      
                        max_Q_values = np.max(self.agent_observations(Qvalue_of_states,i,j))
                        Qvalue_of_states_left_= (1-self.learning_rate)*Qvalue_of_states[i,j] + self.learning_rate * (round(rewards[0] + self.discount_reward * max_Q_values,2))
                        Qvalue_of_states_right_ = (1-self.learning_rate)*Qvalue_of_states[i,j] + self.learning_rate * (round(rewards[1] + self.discount_reward * max_Q_values,2))
                        Qvalue_of_states_up_= (1-self.learning_rate)*Qvalue_of_states[i,j] + self.learning_rate * (round(rewards[2] + self.discount_reward * max_Q_values,2))
                        Qvalue_of_states_down_ = (1-self.learning_rate)*Qvalue_of_states[i,j] + self.learning_rate * (round(rewards[3] + self.discount_reward * max_Q_values,2))
                        
                        Qvalue_of_states[i,j] = np.max(np.array([Qvalue_of_states_left_,Qvalue_of_states_right_,Qvalue_of_states_up_,Qvalue_of_states_down_]))
                        
                    
                    
                #optimal_actions[i,j] = np.argmax(values)
            if(k > 1):
                error =  np.fabs(Qvalue_of_states - previous_Qvalue_of_states)
                sum_error =np.sum(error)
                print(round(sum_error,2))
                
                if(sum_error <= self.tolerance):
                    for i in range(0, len(environment)):
                        for j in range(0, len(environment)):
                               
                            rewards = self.agent_observations(environment,i,j)
                          
                            Qvalue_of_states_left[i,j]= (1-self.learning_rate)*Qvalue_of_states[i,j] + self.learning_rate * (round(rewards[0] + self.discount_reward * max_Q_values,2))
                            Qvalue_of_states_right[i,j] = (1-self.learning_rate)*Qvalue_of_states[i,j] + self.learning_rate * (round(rewards[1] + self.discount_reward * max_Q_values,2))
                            Qvalue_of_states_up[i,j]= (1-self.learning_rate)*Qvalue_of_states[i,j] + self.learning_rate * (round(rewards[2] + self.discount_reward * max_Q_values,2))
                            Qvalue_of_states_down[i,j] = (1-self.learning_rate)*Qvalue_of_states[i,j] + self.learning_rate * (round(rewards[3] + self.discount_reward * max_Q_values,2))
                    break
                
            previous_Qvalue_of_states = Qvalue_of_states.copy()
           
        print(Qvalue_of_states_left)
        print(Qvalue_of_states_right)
        print(Qvalue_of_states_up)
        print(Qvalue_of_states_down)
        return Qvalue_of_states_left, Qvalue_of_states_right, Qvalue_of_states_up, Qvalue_of_states_down
        
    
    
    def agent_navigate(self,start_index,goal_index,goal2_index,wall_index,fire_index,agent_row_position,agent_col_position,goal_row_position,goal_col_position,goal2_row_position,goal2_col_position,number_of_states,environment,value_of_states_left, value_of_states_right, value_of_states_up, value_of_states_down,buttons,root):
            
            reshape_buttons = list()
            prev_listed_actions=list()
            
            prev_rewards = -1e1000000
            index = 0
            start_row_position,start_col_position = agent_row_position,agent_col_position
            for i in range(0, number_of_states):
                buttons_ = list()
                for j in range(0, number_of_states):
                   buttons_.append(buttons[index])
                   index = index + 1
                reshape_buttons.append(buttons_)
                    
            for i in range(0, self.number_of_episodes):
                exploration = 1
                explotation = 0
                self.reset_colors(start_index,goal_index,goal2_index,wall_index,fire_index,buttons,root)
                selected_actions = list()
                rewards = 0
                agent_row_position,agent_col_position = start_row_position,start_col_position
                for count in range(0, 100):
                    
                    if( i < (0.9*self.number_of_episodes)):
                        exploration = 1
                    else:
                        exploration = 0.05
                    explotation = round(np.random.rand(),2)
                        
                    if(exploration > explotation):
                        select_action = self.agent_action(4)
                    else:
                        left_action_value = value_of_states_left[agent_row_position,agent_col_position]
                        right_action_value = value_of_states_right[agent_row_position,agent_col_position]
                        up_action_value = value_of_states_up[agent_row_position,agent_col_position]
                        down_action_value = value_of_states_down[agent_row_position,agent_col_position]
                        observ = np.array([left_action_value,right_action_value,up_action_value, down_action_value])
                        actions  = np.argwhere(observ  == np.max( observ ))
                        actions = actions.flatten()
                        if(len(actions) > 1):
                            rand_action = self.agent_action(len(actions))
                            select_action = actions[rand_action]
                        else:
                            select_action = actions[0]
                    
                    selected_actions.append(select_action)
                    agent_row_position,agent_col_position = self.agent_current_position(select_action,agent_row_position,agent_col_position,environment)
                    rewards += environment[agent_row_position,agent_col_position]
                    
                    reshape_buttons[agent_row_position][agent_col_position].config(bg="#B10DC9")
                    root.update()
                    
                    if((agent_row_position == goal_row_position and agent_col_position == goal_col_position) or (agent_row_position == goal2_row_position and agent_col_position == goal2_col_position)):
                        #print(rewards)
                        break
                if(rewards > prev_rewards):
                    prev_listed_actions = selected_actions
                    prev_rewards = rewards
            #print(prev_rewards)        
            self.reset_colors(start_index,goal_index,goal2_index,wall_index,fire_index,buttons,root)
            agent_row_position,agent_col_position = start_row_position,start_col_position 
            for i in prev_listed_actions:
                agent_row_position,agent_col_position = self.agent_current_position(i,agent_row_position,agent_col_position,environment)
                reshape_buttons[agent_row_position][agent_col_position].config(bg="#B10DC9")
                
                
    def td_agent_navigate(self,start_index,goal_index,goal2_index,wall_index,fire_index,agent_row_position,agent_col_position,goal_row_position,goal_col_position,goal2_row_position,goal2_col_position,number_of_states,environment,value_of_states_left, value_of_states_right, value_of_states_up, value_of_states_down,buttons,root):
            
            reshape_buttons = list()
            prev_listed_actions=list()
            prev_rewards = -1e1000000
            index = 0
            start_row_position,start_col_position = agent_row_position,agent_col_position
            for i in range(0, number_of_states):
                buttons_ = list()
                for j in range(0, number_of_states):
                   buttons_.append(buttons[index])
                   index = index + 1
                reshape_buttons.append(buttons_)
                    
            for i in range(0, self.number_of_episodes):
                self.reset_colors(start_index,goal_index,goal2_index,wall_index,fire_index,buttons,root)
                selected_actions = list()
                rewards = 0
                agent_row_position,agent_col_position = start_row_position,start_col_position
                value_of_states_left_ = value_of_states_left.copy()
                value_of_states_right_ = value_of_states_right.copy()
                value_of_states_up_ = value_of_states_up.copy()
                value_of_states_down_ = value_of_states_down.copy()
                available_actions = self.check_available_actions(environment,agent_row_position,agent_col_position)

                for count in range(0, 100):
                    exploration = 0.4
                    explotation = round(np.random.rand(),2)
                        
                    if(exploration > explotation):
                        select_action = self.agent_action(4)
                        while(select_action not in available_actions):
                             select_action = self.agent_action(4)     
                    else:
                        left_action_value = value_of_states_left_[agent_row_position,agent_col_position]
                        right_action_value = value_of_states_right_[agent_row_position,agent_col_position]
                        up_action_value = value_of_states_up_[agent_row_position,agent_col_position]
                        down_action_value = value_of_states_down_[agent_row_position,agent_col_position]
                        if(0 not in available_actions):
                            left_action_value = -1e1000
                        if(1 not in available_actions):
                            right_action_value = -1e1000
                        if(2 not in available_actions):
                            up_action_value = -1e1000
                        if(3 not in available_actions):
                            down_action_value = -1e1000
                            
                        observ = np.array([left_action_value,right_action_value,up_action_value, down_action_value])
                        
                        actions  = np.argwhere(observ  == np.max( observ ))
                        actions = actions.flatten()
                        if(len(actions) > 1):
                            rand_action = self.agent_action(len(actions))
                            select_action = actions[rand_action]
                        else:
                            select_action = actions[0]
                    # if(select_action == 0):
                    #     value_of_states_left_[agent_row_position,agent_col_position] = -1e1000
                    # elif(select_action == 1):
                    #     value_of_states_right_[agent_row_position,agent_col_position] = -1e1000
                    # elif(select_action == 2):
                    #     value_of_states_up_[agent_row_position,agent_col_position] = -1e1000
                    # elif(select_action == 3):
                    #     value_of_states_down_[agent_row_position,agent_col_position] = -1e1000
                    
                    selected_actions.append(select_action)
                    agent_row_position,agent_col_position = self.agent_current_position(select_action,agent_row_position,agent_col_position,environment)
                    rewards += environment[agent_row_position,agent_col_position]
                    
                    reshape_buttons[agent_row_position][agent_col_position].config(bg="#B10DC9")
                    root.update()
                
                    if((agent_row_position == goal_row_position and agent_col_position == goal_col_position) or (agent_row_position == goal2_row_position and agent_col_position == goal2_col_position)):
                        print(rewards)
                        break
                if(rewards > prev_rewards):
                    prev_listed_actions = selected_actions
                    prev_rewards = rewards
            print(prev_rewards)        
            self.reset_colors(start_index,goal_index,goal2_index,wall_index,fire_index,buttons,root)
            agent_row_position,agent_col_position = start_row_position,start_col_position 
            for i in prev_listed_actions:
                agent_row_position,agent_col_position = self.agent_current_position(i,agent_row_position,agent_col_position,environment)
                reshape_buttons[agent_row_position][agent_col_position].config(bg="#B10DC9")
                
                
    def buttons_create(self,start_index,goal_index,goal2_index,wall_index,fire_index,number_of_states,buttons):
            reshape_buttons = list()
            index = 0
            for i in range(0, number_of_states):
                buttons_ = list()
                for j in range(0, number_of_states):
                   buttons_.append(buttons[index])
                   index = index + 1
                reshape_buttons.append(buttons_)
            return reshape_buttons
        
    def move_agent(self,agent_row_position,agent_col_position,reshape_buttons,root):
        reshape_buttons[agent_row_position][agent_col_position].config(bg="#B10DC9")
        root.update()
                
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
        #%%value_learning      
        #value_of_states_left, value_of_states_right, value_of_states_up, value_of_states_down = self.value_learning(environment,number_of_states,goal_row_position,goal_col_position,goal2_row_position,goal2_col_position)
        #%%Qlearning
        #Qvalue_of_states_left, Qvalue_of_states_right, Qvalue_of_states_up, Qvalue_of_states_down = self.Q_learning(environment,number_of_states,goal_row_position,goal_col_position,goal2_row_position,goal2_col_position)
        value_of_states_left, value_of_states_right, value_of_states_up, value_of_states_down = self.td_Q_learning(environment,number_of_states,goal_row_position,goal_col_position,goal2_row_position,goal2_col_position)

        #%% td_value_learning
        #value_of_states_left, value_of_states_right, value_of_states_up, value_of_states_down= self.td_value_learning(environment,number_of_states,goal_row_position,goal_col_position,goal2_row_position,goal2_col_position)
        # print(value_of_states_left)
        # print(value_of_states_right)
        # print(value_of_states_up)
        # print(value_of_states_down)
       
        
        # print(f"Total returns: {returns}")
#%% navigation
        #self.agent_navigate(start_index,goal_index,goal2_index,wall_index,fire_index,agent_row_position,agent_col_position,goal_row_position,goal_col_position,goal2_row_position,goal2_col_position,number_of_states,environment,value_of_states_left, value_of_states_right, value_of_states_up, value_of_states_down,buttons,root)
        self.td_agent_navigate(start_index,goal_index,goal2_index,wall_index,fire_index,agent_row_position,agent_col_position,goal_row_position,goal_col_position,goal2_row_position,goal2_col_position,number_of_states,environment,value_of_states_left, value_of_states_right, value_of_states_up, value_of_states_down,buttons,root)

        #self.agent_navigate(start_index,goal_index,goal2_index,wall_index,fire_index,agent_row_position,agent_col_position,goal_row_position,goal_col_position,goal2_row_position,goal2_col_position,number_of_states,environment,Qvalue_of_states_left, Qvalue_of_states_right, Qvalue_of_states_up, Qvalue_of_states_down,buttons,root)
      
       
        
   
        
        
   
      
        
       
   
        
        
     
        
     
    
        
        
               
    
        
        
        
        
        
        
        
        
        