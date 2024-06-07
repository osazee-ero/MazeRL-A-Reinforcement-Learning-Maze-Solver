import os
working_dir = r"C:\Users\eroew\OneDrive\PersonalWorks\AIProjects\ErosAIPortfolio\QLearning"
os.chdir(working_dir)

import math
import numpy as np
import tkinter as tk

class Qlearn:
    def __init__(self,goal_reward,loss_reward,discount_reward,fire_reward,learning_rate,epochs):
        self.goal_reward = goal_reward
        self.loss_reward = loss_reward
        self.fire_reward = fire_reward
        self.discount_reward = discount_reward
        self.learning_rate = learning_rate
        self.epochs = epochs
        
    def generate_matrices(self,start_index, goal_index, wall_index,fire_index,routes):
        Q_matrix = np.zeros((routes,routes))
        reward_matrix = -self.loss_reward * np.ones((routes,routes))
        goal_row_position, goal_col_position = math.floor(goal_index/routes), goal_index%routes
        reward_matrix[goal_row_position,goal_col_position] = self.goal_reward
        for j in wall_index:
             wall_row_position, wall_col_position = math.floor(j/routes), j%routes
             reward_matrix[wall_row_position, wall_col_position] = -100000
        for j in fire_index:
            fire_row_position, fire_col_position = math.floor(j/routes), j%routes
            reward_matrix[fire_row_position, fire_col_position] = -self.fire_reward
        
        agent_row_position, agent_col_position = math.floor(start_index/routes), start_index%routes
        # print(reward_matrix)
        return agent_row_position, agent_col_position, reward_matrix,Q_matrix,goal_row_position, goal_col_position
    
    def select_random_actions(self,agent_row_position,agent_col_position,routes,random_action):
        if(agent_row_position == routes-1 and agent_col_position == routes-1):
            if(random_action == 0):
                agent_row_position,agent_col_position = agent_row_position, agent_col_position-1
            elif(random_action ==1):
                agent_row_position,agent_col_position = agent_row_position, agent_col_position
            elif(random_action ==2):
                agent_row_position,agent_col_position = agent_row_position-1, agent_col_position
            else:
                agent_row_position,agent_col_position = agent_row_position, agent_col_position
                
        elif(agent_row_position == 0 and agent_col_position == routes-1):
            if(random_action == 0):
                agent_row_position,agent_col_position = agent_row_position, agent_col_position-1
            elif(random_action ==1):
                agent_row_position,agent_col_position = agent_row_position, agent_col_position
            elif(random_action ==2):
                agent_row_position,agent_col_position = agent_row_position, agent_col_position
            else:
                agent_row_position,agent_col_position = agent_row_position+1, agent_col_position
        elif(agent_row_position == routes-1 and agent_col_position == 0):
            if(random_action == 0):
                agent_row_position,agent_col_position = agent_row_position, agent_col_position
            elif(random_action ==1):
                agent_row_position,agent_col_position = agent_row_position, agent_col_position+1
            elif(random_action ==2):
                agent_row_position,agent_col_position = agent_row_position-1, agent_col_position
            else:
                agent_row_position,agent_col_position = agent_row_position, agent_col_position
        elif(agent_row_position == 0 and agent_col_position == 0):
            if(random_action == 0):
                agent_row_position,agent_col_position = agent_row_position, agent_col_position
            elif(random_action ==1):
                agent_row_position,agent_col_position = agent_row_position, agent_col_position+1
            elif(random_action ==2):
                agent_row_position,agent_col_position = agent_row_position, agent_col_position
            else:
                agent_row_position,agent_col_position = agent_row_position+1, agent_col_position
        elif(agent_row_position == 0 and (agent_col_position > 0 and agent_col_position < routes-1)):
            if(random_action == 0):
                agent_row_position,agent_col_position = agent_row_position, agent_col_position-1
            elif(random_action ==1):
                agent_row_position,agent_col_position = agent_row_position, agent_col_position+1
            elif(random_action ==2):
                agent_row_position,agent_col_position = agent_row_position, agent_col_position
            else:
                agent_row_position,agent_col_position = agent_row_position+1, agent_col_position
        elif(agent_col_position == 0 and (agent_row_position > 0 and agent_row_position < routes-1)):
            if(random_action == 0):
                agent_row_position,agent_col_position = agent_row_position, agent_col_position
            elif(random_action ==1):
                agent_row_position,agent_col_position = agent_row_position, agent_col_position+1
            elif(random_action ==2):
                agent_row_position,agent_col_position = agent_row_position-1, agent_col_position
            else:
                agent_row_position,agent_col_position = agent_row_position+1, agent_col_position
        elif(agent_row_position == routes-1 and (agent_col_position > 0 and agent_col_position < routes-1)):
            if(random_action == 0):
                agent_row_position,agent_col_position = agent_row_position, agent_col_position-1
            elif(random_action ==1):
                agent_row_position,agent_col_position = agent_row_position, agent_col_position+1
            elif(random_action ==2):
                agent_row_position,agent_col_position = agent_row_position-1, agent_col_position
            else:
                agent_row_position,agent_col_position = agent_row_position, agent_col_position     
        elif(agent_col_position == routes-1 and (agent_row_position > 0 and agent_row_position < routes-1)):
            if(random_action == 0):
                agent_row_position,agent_col_position = agent_row_position, agent_col_position-1
            elif(random_action ==1):
                agent_row_position,agent_col_position = agent_row_position, agent_col_position
            elif(random_action ==2):
                agent_row_position,agent_col_position = agent_row_position-1, agent_col_position
            else:
                agent_row_position,agent_col_position = agent_row_position+1, agent_col_position 
        else:
            if(random_action == 0):
                agent_row_position,agent_col_position = agent_row_position, agent_col_position-1
            elif(random_action ==1):
                agent_row_position,agent_col_position = agent_row_position, agent_col_position+1
            elif(random_action ==2):
                agent_row_position,agent_col_position = agent_row_position-1, agent_col_position
            else:
                agent_row_position,agent_col_position = agent_row_position+1, agent_col_position 
                
        return agent_row_position,agent_col_position
    
        
    def select_action(self,agent_row_position, agent_col_position,routes,Q_matrix,reward_matrix,exploration,exploitation):
        #[left right up down]
        if(agent_row_position == routes-1 and agent_col_position == routes-1):
            
            actions = np.array([Q_matrix[agent_row_position, agent_col_position-1],
                                Q_matrix[agent_row_position, agent_col_position], 
                                Q_matrix[agent_row_position-1, agent_col_position],
                                Q_matrix[agent_row_position, agent_col_position]])
            rewards = np.array([reward_matrix[agent_row_position, agent_col_position-1],
                                reward_matrix[agent_row_position, agent_col_position], 
                                reward_matrix[agent_row_position-1, agent_col_position],
                                reward_matrix[agent_row_position, agent_col_position]])
            
        elif(agent_row_position == 0 and agent_col_position == routes-1):
            actions = np.array([Q_matrix[agent_row_position, agent_col_position-1],
                                Q_matrix[agent_row_position, agent_col_position], 
                                Q_matrix[agent_row_position, agent_col_position],
                                Q_matrix[agent_row_position+1, agent_col_position]])
            rewards = np.array([reward_matrix[agent_row_position, agent_col_position-1],
                                reward_matrix[agent_row_position, agent_col_position], 
                                reward_matrix[agent_row_position, agent_col_position],
                                reward_matrix[agent_row_position+1, agent_col_position]])
        elif(agent_row_position == routes-1 and agent_col_position == 0):
            actions = np.array([Q_matrix[agent_row_position, agent_col_position],
                                Q_matrix[agent_row_position, agent_col_position+1], 
                                Q_matrix[agent_row_position-1, agent_col_position],
                                Q_matrix[agent_row_position, agent_col_position]])
            rewards = np.array([reward_matrix[agent_row_position, agent_col_position],
                                reward_matrix[agent_row_position, agent_col_position+1], 
                                reward_matrix[agent_row_position-1, agent_col_position],
                                reward_matrix[agent_row_position, agent_col_position]])
        elif(agent_row_position == 0 and agent_col_position == 0):
            actions = np.array([Q_matrix[agent_row_position, agent_col_position],
                                Q_matrix[agent_row_position, agent_col_position+1], 
                                Q_matrix[agent_row_position, agent_col_position],
                                Q_matrix[agent_row_position+1, agent_col_position]])
            rewards = np.array([reward_matrix[agent_row_position, agent_col_position],
                                reward_matrix[agent_row_position, agent_col_position+1], 
                                reward_matrix[agent_row_position, agent_col_position],
                                reward_matrix[agent_row_position+1, agent_col_position]])
            
        elif(agent_row_position == 0 and (agent_col_position > 0 and agent_col_position < routes-1)):
            actions = np.array([Q_matrix[agent_row_position, agent_col_position-1],
                                Q_matrix[agent_row_position, agent_col_position+1], 
                                Q_matrix[agent_row_position, agent_col_position],
                                Q_matrix[agent_row_position+1, agent_col_position]])
            rewards = np.array([reward_matrix[agent_row_position, agent_col_position-1],
                                reward_matrix[agent_row_position, agent_col_position+1], 
                                reward_matrix[agent_row_position, agent_col_position],
                                reward_matrix[agent_row_position+1, agent_col_position]])
            
        elif(agent_col_position == 0 and (agent_row_position > 0 and agent_row_position < routes-1)):
            actions = np.array([Q_matrix[agent_row_position, agent_col_position],
                                Q_matrix[agent_row_position, agent_col_position+1], 
                                Q_matrix[agent_row_position-1, agent_col_position],
                                Q_matrix[agent_row_position+1, agent_col_position]])
            rewards = np.array([reward_matrix[agent_row_position, agent_col_position],
                                reward_matrix[agent_row_position, agent_col_position+1], 
                                reward_matrix[agent_row_position-1, agent_col_position],
                                reward_matrix[agent_row_position+1, agent_col_position]])
        
        elif(agent_row_position == routes-1 and (agent_col_position > 0 and agent_col_position < routes-1)):
            actions = np.array([Q_matrix[agent_row_position, agent_col_position-1],
                                Q_matrix[agent_row_position, agent_col_position+1], 
                                Q_matrix[agent_row_position-1, agent_col_position],
                                Q_matrix[agent_row_position, agent_col_position]])
            rewards = np.array([reward_matrix[agent_row_position, agent_col_position-1],
                                reward_matrix[agent_row_position, agent_col_position+1], 
                                reward_matrix[agent_row_position-1, agent_col_position],
                                reward_matrix[agent_row_position, agent_col_position]])
            
        elif(agent_col_position == routes-1 and (agent_row_position > 0 and agent_row_position < routes-1)):
            actions = np.array([Q_matrix[agent_row_position, agent_col_position-1],
                                Q_matrix[agent_row_position, agent_col_position], 
                                Q_matrix[agent_row_position-1, agent_col_position],
                                Q_matrix[agent_row_position+1, agent_col_position]])
            rewards = np.array([reward_matrix[agent_row_position, agent_col_position-1],
                                reward_matrix[agent_row_position, agent_col_position], 
                                reward_matrix[agent_row_position-1, agent_col_position],
                                reward_matrix[agent_row_position+1, agent_col_position]])
            
            
        else:
            actions = np.array([Q_matrix[agent_row_position, agent_col_position-1],
                                Q_matrix[agent_row_position, agent_col_position+1], 
                                Q_matrix[agent_row_position-1, agent_col_position],
                                Q_matrix[agent_row_position+1, agent_col_position]])
            rewards = np.array([reward_matrix[agent_row_position, agent_col_position-1],
                                reward_matrix[agent_row_position, agent_col_position+1], 
                                reward_matrix[agent_row_position-1, agent_col_position],
                                reward_matrix[agent_row_position+1, agent_col_position]])
            
        if(exploration >= exploitation):
            random_action = np.random.randint(0,4)
            select_action = actions[random_action] 
            select_reward = rewards[random_action]
        else:
            random_action = np.argmax(actions)
            select_action = np.max(actions)
            select_reward = rewards[random_action] 
        
        Q_matrix[agent_row_position, agent_col_position]   = Q_matrix[agent_row_position,agent_col_position] + self.learning_rate * (select_reward + self.discount_reward*select_action - Q_matrix[agent_row_position,agent_col_position])
       
        agent_row_position,agent_col_position = self.select_random_actions(agent_row_position,agent_col_position,routes,random_action)
                
            

            
        return Q_matrix,agent_row_position,agent_col_position
    
    def find_goal(self,agent_row_position, agent_col_position,goal_row_position, goal_col_position,Q_matrix,buttons,root,routes):
        while(agent_row_position != goal_row_position and agent_col_position != goal_col_position):
            actions = np.array([Q_matrix[agent_row_position, agent_col_position-1],
                                Q_matrix[agent_row_position, agent_col_position+1], 
                                Q_matrix[agent_row_position-1, agent_col_position],
                                Q_matrix[agent_row_position+1, agent_col_position]])
            random_action = np.argmax(actions)
            agent_row_position,agent_col_position = self.select_random_actions(agent_row_position,agent_col_position,routes,random_action)
            # print([agent_row_position,agent_col_position])
            buttons[agent_row_position*routes + agent_col_position].config(bg="#B10DC9")
            root.update()

                
        
    def explore(self,start_index, goal_index, wall_index,fire_index,routes,actions,buttons,root):
           agent_row_position, agent_col_position, reward_matrix,Q_matrix,goal_row_position, goal_col_position = self.generate_matrices(start_index, goal_index, wall_index,fire_index,routes)
           start_row_position, start_col_position = agent_row_position, agent_col_position
           for epoch in range(0, self.epochs):
               agent_row_position, agent_col_position = start_row_position, start_col_position
               action = 0
               if(epoch < 0.9 * self.epochs ):
                   exploration = 1.000
               else:
                   exploration = 0.5
               exploitation = round(np.random.rand(),3)    
               while(action < actions):
                   if((agent_row_position *routes + agent_col_position in wall_index)):
                       buttons[agent_row_position *routes + agent_col_position].config(bg="black")
                   elif((agent_row_position *routes + agent_col_position in fire_index)):
                       buttons[agent_row_position *routes + agent_col_position].config(bg="red")
                   elif(agent_row_position *routes + agent_col_position == start_row_position*routes + start_col_position ):
                       buttons[agent_row_position *routes + agent_col_position].config(bg="#B10DC9")
                   else:
                       buttons[agent_row_position *routes + agent_col_position].config(bg="white")
                   root.update()
                   Q_matrix,agent_row_position,agent_col_position = self.select_action(agent_row_position, agent_col_position,routes,Q_matrix,reward_matrix,exploration,exploitation)
                   buttons[agent_row_position *routes + agent_col_position].config(bg="#B10DC9")
                   root.update()
                   if(agent_row_position == goal_row_position and agent_col_position==goal_col_position):
                       buttons[agent_row_position *routes + agent_col_position].config(bg="green")
                       root.update()
                       break
                #    print(Q_matrix)
                   action = action + 1
               print("epochs completed")
           
           return Q_matrix
             
               
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        