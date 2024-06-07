
import os
working_dir = r"C:\Users\eroew\OneDrive\PersonalWorks\AIProjects\ErosAIPortfolio\QLearning"
# os.chdir(working_dir)
import tkinter as tk
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd



class Qlearn:
    
    def __init__(self,goal_reward,goal_reward2,discount_reward,loss_reward,tolerance,epochs,history):
        self.goal_reward = goal_reward
        self.goal_reward2 = goal_reward2
        self.discount_reward = discount_reward
        self.loss_reward = loss_reward
        self.tolerance = tolerance
        self.epochs = epochs
        self.history = history
       
        
    def create_env(self, goal_index, goal2_index, number_of_states):
         environment_ = -self.loss_reward * np.ones(number_of_states)
         environment_[goal_index] = self.goal_reward
         environment_[goal2_index] = self.goal_reward2
         print(environment_)
         return np.array(environment_)
     
    def state_value_func(self,state, environment, policy):
        state_value_fun_sum = 0
        #states = []
        
        if(policy=="right"):
            env = iter(environment[state:])
            for i in range(0, len(environment[state:])):
                #states.append(environment[i+state]) 
                state_value_fun_sum += pow(self.discount_reward,i) * next(env)
        elif(policy == "left"):
            env = iter(environment[:state])
            for i in range(0, len(environment[:state])):
                #states.append(environment[i])
                state_value_fun_sum += pow(self.discount_reward,i) * next(env)
        return state_value_fun_sum
    
    def state_action_func(self,environment,state,action):
        Q_value_sum = 0
        #states =[]
        if(action == "right"):
            env = iter(environment[state:])
            for i in range(0, len(environment[state:])):
               # states.append(environment[i+state])
                Q_value_sum += pow(self.discount_reward,i) * next(env)
        elif(action == "left"):
            env = iter(environment[:state])
            for i in range(0, len(environment[:state])):
                #states.append(environment[i])
                Q_value_sum += pow(self.discount_reward,i) * next(env)
        return Q_value_sum
    
    def bellman_value_func(self,value_fun,environment,state,action):
       
        value_fun_sum = 0
        
        #states =[]
        if(action == "right"):
            #env = iter(environment[state+1:self.history+1])
            if(state < len(environment)-1):
                env = environment[state+1]
            else:
                env =0
            next_state = iter(value_fun[state+1:self.history+1])
            for i in range(0, len(environment[state+1:self.history+1])):
               # states.append(environment[i+state])
              
                value_fun_sum +=   next(next_state)
            value_fun_sum = env + self.discount_reward * value_fun_sum
                
                    
                
        elif(action == "left"):
            #env = iter(environment[state-self.history:state])
            if(state > 0): 
                env = environment[state-1]
            else:
                env = 0
                
            next_state = iter(value_fun[state-self.history:state])
            for i in range(0, len(environment[state-self.history:state])):
                #states.append(environment[i])
                 
                value_fun_sum +=  next(next_state)
            value_fun_sum = env + self.discount_reward * value_fun_sum
                
                            
        return value_fun_sum
    
    def check_optimality(self,Q_table1,Q_table2):
        error = np.fabs(Q_table2-Q_table1)
        sum_error =np.sum(error)
        #avg_error =sum_error/len(error)
        
        if(sum_error <= self.tolerance):
            return True,sum_error
        else:
            return False,sum_error
        
    def agent_navigate(self,start_index,goal_index,goal2_index, optimal_val_func,buttons,root):
            
            k = start_index
            for i in range(0, len(buttons)*2):
                try:
                    if(optimal_val_func[k,1] == 0):
                        buttons[k-1].config(bg="#B10DC9")
                        k=k-1
                    elif(optimal_val_func[k,1] == 1):
                        buttons[k+1].config(bg="#B10DC9")
                        k=k+1
                except IndexError:
                    print("Agent is confused")
                    break
                else:
                    root.update()
                if(goal2_index == k or  goal_index == k):
                    break
            
                  
                
    def reset_colors(self,start_index,goal_index,goal2_index,buttons,root):
        
        for i in range(0, len(buttons)):
            buttons[i].config(bg="white")
        buttons[start_index].config(bg="#B10DC9")
        buttons[goal_index].config(bg="green")
        buttons[goal2_index].config(bg="green")
        root.update()
                
               
    
    def explore_with_state_value_func(self,goal_index, goal2_index, start_index, number_of_states,buttons,root):
        self.reset_colors(start_index,goal_index,goal2_index,buttons,root)
        environment = self.create_env(goal_index, goal2_index, number_of_states)
        #states_value_func_policies = np.zeros((number_of_states, 5))
        #Q_table_ = np.zeros((number_of_states,5))
        value_fun_matrix_left = np.zeros(number_of_states)
        value_fun_matrix_right = np.zeros(number_of_states)
        value_fun_matrix = np.zeros(number_of_states)
        compare_value_fun_matrix = np.zeros(number_of_states)
        QValue_table = np.zeros((number_of_states,5))
        optimal_val_func = np.zeros((number_of_states,3))
        errors=list()
    
        # for i in range(0, len(environment)):
             
        #     states_value_func_policies[i,:]  = np.array([i, 0, self.state_value_func(i, environment, "left"), 1, self.state_value_func(i, environment, "right")])
        #     Q_table_[i,:]= np.array([i ,0, self.state_action_func(environment,i,"left"), 1, self.state_action_func(environment,i,"right")])
           
        # Q_table = pd.DataFrame(Q_table_, columns=["state","left","QvalueL","right","QvalueR"])
        # State_table = pd.DataFrame(states_value_func_policies, columns=["state","policy1","QvalueL","policy2","QvalueR"])
        # print(Q_table)
        #print(State_table)
        for epoch in range(0, self.epochs):
        
            for i in range(0, len(environment)):
               value_fun_matrix_left[i] = self.bellman_value_func(value_fun_matrix,environment,i,"left")
               value_fun_matrix_right[i] = self.bellman_value_func(value_fun_matrix,environment,i,"right")
               QValue_table[i,:] = np.array([i ,0, value_fun_matrix_left[i], 1, value_fun_matrix_right[i]])
               
            for k,l in enumerate(zip( value_fun_matrix_left, value_fun_matrix_right)):
                if(l[0] > l[1]):
                    value_fun_matrix[k] = l[0]
                    
                else:
                    value_fun_matrix[k] = l[1]
                    
                
            if(epoch > 5):
               condition,error = self.check_optimality(value_fun_matrix,compare_value_fun_matrix)
               errors.append(error)
               if(condition):
                   print("convergence acheived")
                   break
               else:
                   pass
            
            compare_value_fun_matrix = value_fun_matrix
            print(f"epochs: {epoch}")
        for k,l in enumerate(zip( value_fun_matrix_left, value_fun_matrix_right)):
            if(l[0] > l[1]):
                optimal_val_func[k,:] =  np.array([k ,0, l[0]])  
            else:
                optimal_val_func[k,:] =  np.array([k ,1, l[1]])  
            
        QValue_tables = pd.DataFrame(QValue_table, columns=["state","policy1","valueL","policy2","valueR"] )
        self.agent_navigate(start_index,goal_index,goal2_index,optimal_val_func,buttons,root)
        print(QValue_tables)
        print(optimal_val_func)
        #plt.figure(figsize=(5,5))
        #plt.plot(range(0,len(errors)), errors)
        #plt.xlabel("number of epochs")
        #plt.ylabel("Q_errors")
        #plt.title("Convergence graph")
        
       
   
        
        
     
        
     
    
        
        
               
    
        
        
        
        
        
        
        
        
        