import os
working_dir = r"C:\Users\eroew\OneDrive\PersonalWorks\AIProjects\ErosAIPortfolio\QLearning"
os.chdir(working_dir)
import tkinter as tk
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd



class Qlearn:
    
    def __init__(self,goal_reward,goal_reward2,discount_reward,loss_reward,fire_reward,tolerance,learning_rate,epochs,history):
        self.goal_reward = goal_reward
        self.goal_reward2 = goal_reward2
        self.fire_reward = fire_reward
        self.discount_reward = discount_reward
        self.loss_reward = loss_reward
        self.tolerance = tolerance
        self.epochs = epochs
        self.history = history
        self.learning_rate = learning_rate
       
        
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
     
    
    
    
    def states(self,value_fun,environment,agent_row_position,agent_col_position):
        
        if(agent_row_position==0 and agent_col_position == 0):
                env = np.array([environment[agent_row_position,agent_col_position],environment[agent_row_position,agent_col_position+1],environment[agent_row_position,agent_col_position],environment[agent_row_position+1,agent_col_position]])
                q_next_states = np.array([value_fun[agent_row_position,agent_col_position], value_fun[agent_row_position,agent_col_position+1], value_fun[agent_row_position,agent_col_position], value_fun[agent_row_position+1,agent_col_position]])
        
        elif(agent_row_position==len(environment)-1 and agent_col_position == 0):
                env = np.array([environment[agent_row_position,agent_col_position],environment[agent_row_position,agent_col_position+1],environment[agent_row_position-1,agent_col_position],environment[agent_row_position,agent_col_position]])
                q_next_states = np.array([value_fun[agent_row_position,agent_col_position], value_fun[agent_row_position,agent_col_position+1], value_fun[agent_row_position-1,agent_col_position], value_fun[agent_row_position,agent_col_position]])
        
        elif(agent_row_position==len(environment)-1 and agent_col_position == len(environment)-1):
                env = np.array([environment[agent_row_position,agent_col_position-1],environment[agent_row_position,agent_col_position],environment[agent_row_position-1,agent_col_position],environment[agent_row_position,agent_col_position]])
                q_next_states = np.array([value_fun[agent_row_position,agent_col_position-1], value_fun[agent_row_position,agent_col_position], value_fun[agent_row_position-1,agent_col_position], value_fun[agent_row_position,agent_col_position]])
                
        elif(agent_row_position==0 and agent_col_position == len(environment)-1):
                env = np.array([environment[agent_row_position,agent_col_position-1],environment[agent_row_position,agent_col_position],environment[agent_row_position,agent_col_position],environment[agent_row_position+1,agent_col_position]])
                q_next_states = np.array([value_fun[agent_row_position,agent_col_position-1], value_fun[agent_row_position,agent_col_position], value_fun[agent_row_position,agent_col_position], value_fun[agent_row_position+1,agent_col_position]])
                
        elif(agent_row_position==0 and (agent_col_position > 0  and agent_col_position < len(environment)-1)):
                env = np.array([environment[agent_row_position,agent_col_position-1],environment[agent_row_position,agent_col_position+1],environment[agent_row_position,agent_col_position],environment[agent_row_position+1,agent_col_position]])
                q_next_states = np.array([value_fun[agent_row_position,agent_col_position-1], value_fun[agent_row_position,agent_col_position+1], value_fun[agent_row_position,agent_col_position], value_fun[agent_row_position+1,agent_col_position]])
        
        elif(agent_row_position==len(environment)-1 and (agent_col_position > 0 and agent_col_position < len(environment)-1)):
                env = np.array([environment[agent_row_position,agent_col_position-1],environment[agent_row_position,agent_col_position+1],environment[agent_row_position-1,agent_col_position],environment[agent_row_position,agent_col_position]])
                q_next_states = np.array([value_fun[agent_row_position,agent_col_position-1], value_fun[agent_row_position,agent_col_position+1], value_fun[agent_row_position-1,agent_col_position], value_fun[agent_row_position,agent_col_position]])
                
        elif(agent_col_position==len(environment)-1 and (agent_row_position > 0 and agent_row_position < len(environment)-1)):
                env = np.array([environment[agent_row_position,agent_col_position-1],environment[agent_row_position,agent_col_position],environment[agent_row_position-1,agent_col_position],environment[agent_row_position+1,agent_col_position]])
                q_next_states = np.array([value_fun[agent_row_position,agent_col_position-1], value_fun[agent_row_position,agent_col_position], value_fun[agent_row_position-1,agent_col_position], value_fun[agent_row_position+1,agent_col_position]])
              
        
        elif(agent_col_position==0 and (agent_row_position > 0 and agent_row_position < len(environment)-1)):
                env = np.array([environment[agent_row_position,agent_col_position],environment[agent_row_position,agent_col_position+1],environment[agent_row_position-1,agent_col_position],environment[agent_row_position+1,agent_col_position]])
                q_next_states = np.array([value_fun[agent_row_position,agent_col_position], value_fun[agent_row_position,agent_col_position+1], value_fun[agent_row_position-1,agent_col_position], value_fun[agent_row_position+1,agent_col_position]])
                
        else:
                env = np.array([environment[agent_row_position,agent_col_position-1],environment[agent_row_position,agent_col_position+1],environment[agent_row_position-1,agent_col_position],environment[agent_row_position+1,agent_col_position]])
                q_next_states = np.array([value_fun[agent_row_position,agent_col_position-1], value_fun[agent_row_position,agent_col_position+1], value_fun[agent_row_position-1,agent_col_position], value_fun[agent_row_position+1,agent_col_position]])
            
              
        return env, q_next_states


    def bellman_value_func(self,value_fun,environment,agent_row_position,agent_col_position,action):
       
        
        value_fun_sum = 0
        proba_values = np.zeros(4)
        env, q_next_states = self.states(value_fun,environment,agent_row_position,agent_col_position)
        #states =[]
        if(action == "right"):
           
            proba_values = np.array([0,0.7,0,0])
            value_fun_sum = proba_values * (env + self.discount_reward * q_next_states)
            value_fun_sum = round(np.sum(value_fun_sum),2)
                
        elif(action == "left"):
            
            proba_values = np.array([0.7,0,0,0])
            value_fun_sum = proba_values * (env + self.discount_reward * q_next_states)
            value_fun_sum = round(np.sum(value_fun_sum),2)
            
        elif(action == "down"):
            proba_values = np.array([0,0,0,0.7])
            value_fun_sum = proba_values * (env + self.discount_reward * q_next_states)
            value_fun_sum = round(np.sum(value_fun_sum),2)
            
        elif(action == "up"):
            proba_values = np.array([0,0,0.7,0])
            value_fun_sum = proba_values * (env + self.discount_reward * q_next_states)
            value_fun_sum = round(np.sum(value_fun_sum),2)
                
        return value_fun_sum
    
   
    
    def check_optimality(self,Q_table1,Q_table2):
        error = np.fabs(Q_table2-Q_table1)
        sum_error =np.sum(error)
        #avg_error =sum_error/len(error)
        
        if(sum_error <= self.tolerance):
            return True,sum_error
        else:
            return False,sum_error
        
    def agent_navigate(self,agent_row_position,agent_col_position,goal_row_position,goal_col_position,goal2_row_position,goal2_col_position, number_of_states, optimal_val_func,buttons,root):
            
           
            reshape_buttons = list()
            index = 0
            for i in range(0, number_of_states):
                buttons_ = list()
                for j in range(0, number_of_states):
                   buttons_.append(buttons[index])
                   index = index + 1
                reshape_buttons.append(buttons_)
                    
            for i in range(0, number_of_states*number_of_states*5):
                try:
                    if(optimal_val_func[agent_row_position,agent_col_position] == 0):
                        if(agent_col_position > 0):
                            reshape_buttons[agent_row_position][agent_col_position-1].config(bg="#B10DC9")
                            agent_row_position,agent_col_position = agent_row_position,agent_col_position-1
                        else:
                            reshape_buttons[agent_row_position][agent_col_position].config(bg="#B10DC9")
                            agent_row_position,agent_col_position = agent_row_position,agent_col_position
                            
                    elif(optimal_val_func[agent_row_position,agent_col_position] == 1):
                        if(agent_col_position < number_of_states-1):
                            reshape_buttons[agent_row_position][agent_col_position+1].config(bg="#B10DC9")
                            agent_row_position,agent_col_position = agent_row_position,agent_col_position+1
                        else:
                            reshape_buttons[agent_row_position][agent_col_position].config(bg="#B10DC9")
                            agent_row_position,agent_col_position = agent_row_position,agent_col_position
                            
                    elif(optimal_val_func[agent_row_position,agent_col_position] == 2):
                        if(agent_row_position > 0):
                            reshape_buttons[agent_row_position-1][agent_col_position].config(bg="#B10DC9")
                            agent_row_position,agent_col_position = agent_row_position-1,agent_col_position
                        else:
                            reshape_buttons[agent_row_position][agent_col_position].config(bg="#B10DC9")
                            agent_row_position,agent_col_position = agent_row_position,agent_col_position
                            
                    elif(optimal_val_func[agent_row_position,agent_col_position] == 3):
                        if(agent_row_position < number_of_states-1):
                            reshape_buttons[agent_row_position+1][agent_col_position].config(bg="#B10DC9")
                            agent_row_position,agent_col_position = agent_row_position+1,agent_col_position
                        else:
                            reshape_buttons[agent_row_position][agent_col_position].config(bg="#B10DC9")
                            agent_row_position,agent_col_position = agent_row_position,agent_col_position
                            
                except IndexError:
                    print("Agent is confused")
                    break
                else:
                    root.update()
                    if( (agent_row_position == goal_row_position and agent_col_position == goal_col_position) or (agent_row_position == goal2_row_position and agent_col_position == goal2_col_position)):
                        break
                    
    
            
    
    def live_agent_navigate(self,agent_row_position,agent_col_position,goal_row_position,goal_col_position,goal2_row_position,goal2_col_position, number_of_states, optimal_action,buttons,root):
        reshape_buttons = list()
        index = 0
        for i in range(0, number_of_states):
            buttons_ = list()
            for j in range(0, number_of_states):
               buttons_.append(buttons[index])
               index = index + 1
            reshape_buttons.append(buttons_)
        if(optimal_action == 0):
            if(agent_col_position > 0):
                reshape_buttons[agent_row_position][agent_col_position-1].config(bg="#B10DC9")
                root.update()
                agent_row_position,agent_col_position = agent_row_position,agent_col_position-1
            else:
                reshape_buttons[agent_row_position][agent_col_position].config(bg="#B10DC9")
                root.update()
                agent_row_position,agent_col_position = agent_row_position,agent_col_position
                
        elif(optimal_action == 1):
            if(agent_col_position < number_of_states-1):
                reshape_buttons[agent_row_position][agent_col_position+1].config(bg="#B10DC9")
                root.update()
                agent_row_position,agent_col_position = agent_row_position,agent_col_position+1
            else:
                reshape_buttons[agent_row_position][agent_col_position].config(bg="#B10DC9")
                root.update()
                agent_row_position,agent_col_position = agent_row_position,agent_col_position
                
        elif(optimal_action == 2):
            if(agent_row_position > 0):
                reshape_buttons[agent_row_position-1][agent_col_position].config(bg="#B10DC9")
                root.update()
                agent_row_position,agent_col_position = agent_row_position-1,agent_col_position
            else:
                reshape_buttons[agent_row_position][agent_col_position].config(bg="#B10DC9")
                root.update()
                agent_row_position,agent_col_position = agent_row_position,agent_col_position
                
        elif(optimal_action == 3):
            if(agent_row_position < number_of_states-1):
                reshape_buttons[agent_row_position+1][agent_col_position].config(bg="#B10DC9")
                root.update()
                agent_row_position,agent_col_position = agent_row_position+1,agent_col_position
            else:
                reshape_buttons[agent_row_position][agent_col_position].config(bg="#B10DC9")
                root.update()
                agent_row_position,agent_col_position = agent_row_position,agent_col_position
        
        return agent_row_position,agent_col_position
            
              
                
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
        
               
    
    
    def explore_with_state_value_func(self,goal_index, goal2_index, start_index,wall_index, fire_index,number_of_states,buttons,root):
        number_of_states = int(math.sqrt(number_of_states))
        self.reset_colors(start_index,goal_index,goal2_index,wall_index,fire_index,buttons,root)
        environment,agent_row_position,agent_col_position,goal_row_position,goal_col_position,goal2_row_position,goal2_col_position = self.create_env(start_index,goal_index, goal2_index, wall_index,fire_index,number_of_states)
        #states_value_func_policies = np.zeros((number_of_states, 5))
        #Q_table_ = np.zeros((number_of_states,5))
        value_fun_matrix_left = np.zeros((number_of_states,number_of_states))
        value_fun_matrix_right = np.zeros((number_of_states,number_of_states))
        value_fun_matrix_up = np.zeros((number_of_states,number_of_states))
        value_fun_matrix_down = np.zeros((number_of_states,number_of_states))
        
        value_fun_matrix = np.zeros((number_of_states,number_of_states))
        compare_value_fun_matrix = np.zeros((number_of_states,number_of_states))
        #QValue_table = np.zeros((number_of_states,5))
        optimal_val_func = np.zeros((number_of_states,number_of_states))
        errors=list()
    
      
        for epoch in range(0, self.epochs):
            for i in range(0, len(environment)):
                for j in range(0, len(environment)):
                   if((i==goal_row_position and j==goal_col_position) or (i==goal2_row_position and j==goal2_col_position)):
                       continue
                   else:
                       value_fun_matrix_left[i,j] = self.bellman_value_func(value_fun_matrix,environment,i,j,"left")
                       value_fun_matrix_right[i,j] = self.bellman_value_func(value_fun_matrix,environment,i,j,"right")
                       value_fun_matrix_up[i,j] = self.bellman_value_func(value_fun_matrix,environment,i,j,"up")
                       value_fun_matrix_down[i,j] = self.bellman_value_func(value_fun_matrix,environment,i,j,"down")
                   
               
            for i in range(0, len(value_fun_matrix)):
                for j in range(0, len(value_fun_matrix)):
                    value_fun_matrix[i,j] = np.max(np.array([value_fun_matrix_left[i,j], value_fun_matrix_right[i,j], value_fun_matrix_up[i,j], value_fun_matrix_down[i,j]]))
                    
                    best_action = np.argmax(np.array([value_fun_matrix_left[i,j], value_fun_matrix_right[i,j], value_fun_matrix_up[i,j], value_fun_matrix_down[i,j]]))
                    optimal_val_func[i,j] = best_action
            
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
            
        # for i in range(0, len(value_fun_matrix)):
        #         for j in range(0, len(value_fun_matrix)):
        #             optimal_val_func[i,j] = np.argmax(np.array([value_fun_matrix[i,j], value_fun_matrix_right[i,j], value_fun_matrix_up[i,j], value_fun_matrix_down[i,j]]))
                    
         
            
        #QValue_tables = pd.DataFrame(QValue_table, columns=["state","policy1","valueL","policy2","valueR"] )
        print(errors)
        self.agent_navigate(agent_row_position,agent_col_position,goal_row_position,goal_col_position,goal2_row_position,goal2_col_position, number_of_states, optimal_val_func,buttons,root)
        print(optimal_val_func)
        print(value_fun_matrix)
        plt.figure(figsize=(5,5))
        plt.plot(range(0,len(errors)), errors)
        plt.xlabel("number of epochs")
        plt.ylabel("Q_errors")
        plt.title("Convergence graph")
        
        
    def live_explore_with_state_value_func(self,goal_index, goal2_index, start_index,wall_index, fire_index,number_of_states,buttons,root):
            number_of_states = int(math.sqrt(number_of_states))
            self.reset_colors(start_index,goal_index,goal2_index,wall_index,fire_index,buttons,root)
            environment,agent_row_position,agent_col_position,goal_row_position,goal_col_position,goal2_row_position,goal2_col_position = self.create_env(start_index,goal_index, goal2_index, wall_index,fire_index,number_of_states)
            start_row_position,start_col_position = agent_row_position,agent_col_position
            value_fun_matrix_left = np.zeros((number_of_states,number_of_states))
            value_fun_matrix_right = np.zeros((number_of_states,number_of_states))
            value_fun_matrix_up = np.zeros((number_of_states,number_of_states))
            value_fun_matrix_down = np.zeros((number_of_states,number_of_states))
            
            value_fun_matrix = np.zeros((number_of_states,number_of_states))
            compare_value_fun_matrix = np.zeros((number_of_states,number_of_states))
            optimal_val_func = np.zeros((number_of_states,number_of_states))
            errors=list()
            #exploration = 1.0
            
        
            
            for epoch in range(0, self.epochs):
                agent_row_position,agent_col_position = start_row_position,start_col_position
                self.reset_colors(start_index,goal_index,goal2_index,wall_index,fire_index,buttons,root) 
                index=0
                for i in range(0, number_of_states*2 + 1):
                       
                       value_fun_matrix_left[agent_row_position,agent_col_position] = self.bellman_value_func(value_fun_matrix,environment,agent_row_position,agent_col_position,"left")
                       value_fun_matrix_right[agent_row_position,agent_col_position] = self.bellman_value_func(value_fun_matrix,environment,agent_row_position,agent_col_position,"right")
                       value_fun_matrix_up[agent_row_position,agent_col_position] = self.bellman_value_func(value_fun_matrix,environment,agent_row_position,agent_col_position,"up")
                       value_fun_matrix_down[agent_row_position,agent_col_position] = self.bellman_value_func(value_fun_matrix,environment,agent_row_position,agent_col_position,"down")
                       
                       value_fun_matrix[agent_row_position,agent_col_position] = np.max(np.array([value_fun_matrix_left[agent_row_position,agent_col_position], value_fun_matrix_right[agent_row_position,agent_col_position], value_fun_matrix_up[agent_row_position,agent_col_position], value_fun_matrix_down[agent_row_position,agent_col_position]]))
                      
                       exploitation = round(np.random.rand(),2)
                       exploration = round(np.random.rand(),2)
                       
                       actions = np.array([value_fun_matrix_left[agent_row_position,agent_col_position], value_fun_matrix_right[agent_row_position,agent_col_position], value_fun_matrix_up[agent_row_position,agent_col_position], value_fun_matrix_down[agent_row_position,agent_col_position]])
                       
                      
                       select_action = np.argmax(actions)
                       optimal_val_func[agent_row_position,agent_col_position] = select_action
                       if(exploration > exploitation):
                           optimal_action = np.random.randint(0,4)
                       else:
                           optimal_action = select_action 
                           
                       agent_row_position,agent_col_position = self.live_agent_navigate(agent_row_position,agent_col_position,goal_row_position,goal_col_position,goal2_row_position,goal2_col_position, number_of_states, optimal_action,buttons,root)
                       print(agent_row_position,agent_col_position)
                       index = index + 1
                       if( (agent_row_position == goal_row_position and agent_col_position == goal_col_position) or (agent_row_position == goal2_row_position and agent_col_position == goal2_col_position)):
                           break
                       
                       print(f"move: {index}")
                
                
                if(epoch > 6):
                   condition,error = self.check_optimality(value_fun_matrix,compare_value_fun_matrix)
                   errors.append(error)
                   if(condition):
                       print("convergence acheived")
                       break
                   else:
                       pass
                
                compare_value_fun_matrix = value_fun_matrix
                print(f"epochs: {epoch}")
            
            agent_row_position,agent_col_position = start_row_position,start_col_position
            self.reset_colors(start_index,goal_index,goal2_index,wall_index,fire_index,buttons,root)
            # for i in range(0, len(value_fun_matrix)):
            #         for j in range(0, len(value_fun_matrix)):
            #             optimal_val_func[i,j] = np.argmax(np.array([value_fun_matrix[i,j], value_fun_matrix_right[i,j], value_fun_matrix_up[i,j], value_fun_matrix_down[i,j]]))
                        
             
            self.agent_navigate(agent_row_position,agent_col_position,goal_row_position,goal_col_position,goal2_row_position,goal2_col_position, number_of_states, optimal_val_func,buttons,root)
            print(optimal_val_func)
            print(value_fun_matrix)
           # print(compare_value_fun_matrix)
        
        
   
      
        
       
   
        
        
     
        
     
    
        
        
               
    
        
        
        
        
        
        
        
        
        