
import os
working_dir = r"C:\Users\eroew\OneDrive\PersonalWorks\AIProjects\ErosAIPortfolio\QLearning"
os.chdir(working_dir)
import tkinter as tk
import numpy as np
import math
import matplotlib.pyplot as plt


class Qlearn:
    
    def __init__(self,goal_reward,loss_reward,discount_reward,fire_reward,learning_rate,epochs):
        self.goal_reward = goal_reward
        self.loss_reward = loss_reward
        self.fire_reward = fire_reward
        self.discount_reward = discount_reward
        self.learning_rate = learning_rate
        self.epochs = epochs
        
    def generate_matrices(self,start_index, goal_index, wall_index,fire_index,states_half):
         Q_matrix = np.zeros((states_half,states_half))
         reward_matrix = -self.loss_reward * np.ones((states_half,states_half))
         goal_row_position, goal_col_position = math.floor(goal_index/states_half), goal_index%states_half
         reward_matrix[goal_row_position,goal_col_position] = self.goal_reward
         for j in wall_index:
              wall_row_position, wall_col_position = math.floor(j/states_half), j%states_half
              reward_matrix[wall_row_position, wall_col_position] = -1000
         for j in fire_index:
             fire_row_position, fire_col_position = math.floor(j/states_half), j%states_half
             reward_matrix[fire_row_position, fire_col_position] = -self.fire_reward
         
         agent_row_position, agent_col_position = math.floor(start_index/states_half), start_index%states_half
         
         return agent_row_position, agent_col_position, goal_row_position, goal_col_position, reward_matrix,Q_matrix
     
        
     
    def get_Qmatrices(self,Q_matrix,reward_matrix,goal_row_position,goal_col_position):
        
        q_matrix_for_each_actions = np.zeros(4)
        Q_table = np.zeros((Q_matrix.shape[0]*Q_matrix.shape[0], 4))
        
        #[left,right,down,up]
        actions = np.array([0,1,2,3])
        index = 0
        for i in range(0, Q_matrix.shape[0]):
            for j in range(0, Q_matrix.shape[0]):
                
                if(i == 0 and j==0):
                    rewards = [reward_matrix[i,j],reward_matrix[i,j+1],reward_matrix[i+1,j],reward_matrix[i,j]]
                    q_actions = [Q_matrix[i,j],Q_matrix[i,j+1],Q_matrix[i+1,j],Q_matrix[i,j]]
                  
                    
                elif(i == 0 and j==Q_matrix.shape[0]-1):
                    rewards = [reward_matrix[i,j-1],reward_matrix[i,j],reward_matrix[i+1,j],reward_matrix[i,j]]
                    q_actions = [Q_matrix[i,j-1],Q_matrix[i,j],Q_matrix[i+1,j],Q_matrix[i,j]]
                 
                    
                elif(i == Q_matrix.shape[0]-1 and j==Q_matrix.shape[0]-1):
                    rewards = [reward_matrix[i,j-1],reward_matrix[i,j],reward_matrix[i,j],reward_matrix[i-1,j]]
                    q_actions = [Q_matrix[i,j-1],Q_matrix[i,j],Q_matrix[i,j],Q_matrix[i-1,j]]
                    
                elif(i == Q_matrix.shape[0]-1 and j==0):
                    rewards = [reward_matrix[i,j],reward_matrix[i,j+1],reward_matrix[i,j],reward_matrix[i-1,j]]
                    q_actions = [Q_matrix[i,j],Q_matrix[i,j+1],Q_matrix[i,j],Q_matrix[i-1,j]]
                    
                elif(i == 0 and (j < Q_matrix.shape[0]-1 and j > 0)):
                    rewards = [reward_matrix[i,j-1],reward_matrix[i,j+1],reward_matrix[i+1,j],reward_matrix[i,j]]
                    q_actions = [Q_matrix[i,j-1],Q_matrix[i,j+1],Q_matrix[i+1,j],Q_matrix[i,j]]
                    
                elif(j==0 and (i < Q_matrix.shape[0]-1 and i > 0)):
                    rewards = [reward_matrix[i,j],reward_matrix[i,j+1],reward_matrix[i+1,j],reward_matrix[i-1,j]]
                    q_actions =[Q_matrix[i,j],Q_matrix[i,j+1],Q_matrix[i+1,j],Q_matrix[i-1,j]]
                    
                elif(i == Q_matrix.shape[0]-1 and (j < Q_matrix.shape[0]-1 and j > 0) ):
                    rewards = [reward_matrix[i,j-1],reward_matrix[i,j+1],reward_matrix[i,j],reward_matrix[i-1,j]]
                    q_actions = [Q_matrix[i,j-1],Q_matrix[i,j+1],Q_matrix[i,j],Q_matrix[i-1,j]]
                    
                elif(j == Q_matrix.shape[0]-1 and (i < Q_matrix.shape[0]-1 and i > 0) ):
                     rewards = [reward_matrix[i,j-1],reward_matrix[i,j],reward_matrix[i+1,j],reward_matrix[i-1,j]]      
                     q_actions = [Q_matrix[i,j-1],Q_matrix[i,j],Q_matrix[i+1,j],Q_matrix[i-1,j]]      
                else:
                    rewards = [reward_matrix[i,j-1],reward_matrix[i,j+1],reward_matrix[i+1,j],reward_matrix[i-1,j]]
                    q_actions = [Q_matrix[i,j-1],Q_matrix[i,j+1],Q_matrix[i+1,j],Q_matrix[i-1,j]]
                
                rewards=np.array(rewards)
                q_actions = np.array(q_actions)
                max_states = np.max(q_actions)
                
                if(i==goal_row_position and j==goal_col_position):
                    Q_table[index,:] = np.array([i,j,Q_matrix[i,j],7])
                    
                else:
                    for action in actions:
                        q_matrix_for_each_actions[action] = Q_matrix[i,j] + self.learning_rate * (rewards[action]  + self.discount_reward * max_states - Q_matrix[i,j])
                        
                    max_qvalue = np.max(q_matrix_for_each_actions)
                    max_qaction = np.argmax(q_matrix_for_each_actions)
                    
                    Q_table[index,:] = np.array([i,j,max_qvalue,max_qaction])
                    #Q_matrix[i,j] = max_qvalue
                index = index + 1

        return Q_table
    
    def check_optimality(self,Q_table1,Q_table2,tolerance):
        error = Q_table2[2]-Q_table1[2]
        sum_error =np.sum(error)
        #avg_error =sum_error/len(error)
        
        if(sum_error <= tolerance):
            return True,sum_error
        else:
            return False,sum_error
        
    def reshape_Qmatrix(self,Q_table):
        Q_matrix = Q_table[:,2].reshape(int(math.sqrt(len(Q_table[:,2]))),int(math.sqrt(len(Q_table[:,2]))))
        actions_matrix = Q_table[:,3].reshape(int(math.sqrt(len(Q_table[:,3]))),int(math.sqrt(len(Q_table[:,3]))))
        return Q_matrix,actions_matrix
        
     
    def explore(self,start_index, goal_index, wall_index,fire_index,states_half,tolerance):
       agent_row_position, agent_col_position, goal_row_position, goal_col_position, reward_matrix,Q_matrix = self.generate_matrices(start_index, goal_index, wall_index,fire_index,states_half)
       print(reward_matrix)
       Q_table2=0
       average_error=list()
       for epoch in range(0, self.epochs):
           Q_table1 = self.get_Qmatrices(Q_matrix,reward_matrix,goal_row_position,goal_col_position)
           if epoch > 1:
               condition,avg_error = self.check_optimality(Q_table1,Q_table2,tolerance)
               average_error.append(avg_error)
               if(condition):
                   print("convergence acheived")
                   break
               else:
                   pass
           Q_table2 = Q_table1
           Q_matrix,actions_matrix = self.reshape_Qmatrix(Q_table1)
           print(f"epochs: {epoch}")
       plt.figure(figsize=(5,5))
       plt.plot(range(0,len(average_error)), average_error)
       plt.xlabel("number of epochs")
       plt.ylabel("Q_errors")
       plt.title("Convergence graph")
           
        
       return Q_table1,Q_matrix,actions_matrix
        
        
               
    
        
        
        
        
        
        
        
        
        