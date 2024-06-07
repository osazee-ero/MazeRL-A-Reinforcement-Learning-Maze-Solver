import tkinter as tk
import numpy as np
import math
import Qlearning_v1 as ql

class Qboard:
    def __init__(self):
        self.fire_reward = 1
        self.discount_reward = 0.5
        self.loss_reward = 0.01
        self.goal_reward = 1
        
    def playGame(self):
        q_session = ql.Qlearn(self.goal_reward, self.loss_reward, self.discount_reward, self.fire_reward, 0.01, 1000)
        self.Qtable,self.Q_matrix,self.action_matrix = q_session.explore(self.start_index, self.goal_index, self.walls_index, self.fire_index, int(math.sqrt(self.no_of_states)),1e-9)
        
        
    def drawBoard(self,gridsize,walls,fire,random_state):
       np.random.seed(random_state)
       self.root = tk.Tk()
       self.main_canvas = tk.Frame(self.root, bg="white", border=10)
       self.settings = tk.Frame(self.root, bg="white", border=2)
       self.play = tk.Frame(self.root, bg="white",border=2)
       if(gridsize >= 3 or gridsize <= 50):
           gridsize_ = gridsize
       else:
           gridsize_=3
       self.states = list()
       self.no_of_states = 0
       
       for i in range(0, gridsize_):
           for j in range(0, gridsize_):
               self.Button = tk.Button(self.main_canvas, text="MOV", justify="center")
               self.Button.grid(row=i, column=j, ipadx=5, ipady=5)
               self.states.append(self.Button)
               self.no_of_states += 1 
       
       if(walls >= 0.3 or walls <= 0):
           walls_ = round(0.3*self.no_of_states)
       else:
           walls_ = round(walls*self.no_of_states)
       self.walls_index = list()
       for i in range(0, walls_):
           rand_number = np.random.randint(0, self.no_of_states)
           self.states[rand_number].config(bg="black", text="WAL",justify="center")
           self.walls_index.append(rand_number)
           
       if(fire >= 0.2 or fire <= 0):
           fire_ = round(0.2*self.no_of_states)
       else:
           fire_ = round(fire*self.no_of_states)
       self.fire_index = list()
       for i in range(0, fire_):
           rand_number = np.random.randint(0, self.no_of_states)
           while(rand_number in self.walls_index):
               rand_number = np.random.randint(0, self.no_of_states)
           self.states[rand_number].config(bg="red", text="FIRE",justify="center")
           self.fire_index.append(rand_number)
       self.goal_index = np.random.randint(0, self.no_of_states)
       while(self.goal_index in self.walls_index or self.goal_index in self.fire_index):
            self.goal_index = np.random.randint(0, self.no_of_states)
            
       self.states[self.goal_index].config(bg="green", text="GOAL",justify="center")
       
       self.start_index = np.random.randint(0, self.no_of_states)
       while(self.start_index in self.walls_index or self.start_index in self.fire_index or self.start_index == self.goal_index):
           self.start_index = np.random.randint(0, self.no_of_states)
       self.states[self.start_index].config(bg="#B10DC9", text="START",justify="center")
      
               
               
       def get_fire_slider_values(e):
           self.fire_reward = float(e)
           #print(self.fire_reward)
           
       def get_discount_slider_values(e):
           self.discount_reward = float(e)
           #print(self.discount_reward)
           
       def get_loss_slider_values(e):
           self.loss_reward = float(e)
           #print(self.loss_reward)
           
       def get_goal_slider_values(e):
           self.goal_reward = float(e)
           #print(self.goal_reward)
       print(self.walls_index)
       print(self.fire_index)
       print(self.goal_index)
       print(self.start_index)
           
           
       self.fire_reward_slider = tk.Scale(self.settings, label="fire_reward", from_=0, to=10, orient='horizontal', digits = 3,  resolution=0.01, command=get_fire_slider_values)
       self.fire_reward_slider.set(self.fire_reward)
       self.fire_reward_slider.grid(row=0, column=1, padx=5, pady=5)
       
       self.loss_reward_slider = tk.Scale(self.settings, label="loss_reward", from_=0, to=1, orient='horizontal', digits = 3, resolution=0.01, command=get_loss_slider_values)
       self.loss_reward_slider.set(self.loss_reward)
       self.loss_reward_slider.grid(row=0, column=2, padx=5, pady=5)
       
       self.discount_slider = tk.Scale(self.settings, label="discount", from_=0, to=1, orient='horizontal', digits = 3, resolution=0.01, command=get_discount_slider_values)
       self.discount_slider.set(self.discount_reward)
       self.discount_slider.grid(row=0, column=3, padx=5, pady=5)
       
       self.goal_slider = tk.Scale(self.settings, label="goal_reward", from_=1, to=10, orient='horizontal', digits = 3, resolution=0.01, command=get_goal_slider_values)
       self.goal_slider.set(self.goal_reward)
       self.goal_slider.grid(row=0, column=4, padx=5, pady=5)
       
       self.playButton = tk.Button(self.play, text="START", font=('', 14), justify='center', pady=10, bg="#7FDBFF", command=self.playGame)
       self.playButton.grid(row=0, column=1, padx=10,pady=10)
       
       self.settings.pack()
       self.main_canvas.pack()
       self.play.pack()
       self.root.title("Qlearning")
       self.root.config(bg="black")
       self.root.mainloop()
       
       return self.Qtable, self.Q_matrix,self.action_matrix
       
       
       
board = Qboard()
Qtable,Qmatrix,action_matrix = board.drawBoard(10,0.1,0.1,444)
