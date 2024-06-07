import tkinter as tk
import numpy as np
import math
import ValueLearning_2 as ql

class Qboard:
    def __init__(self):
        self.fire_reward = 1
        self.discount_reward = 0.3
        self.loss_reward = 0.01
        self.goal_reward = 1
        self.goal_reward2 = 1
        self.num_of_states = 5
        self.history = 2
        
    def playGame(self):
        
        qsession = ql.Qlearn(self.goal_reward, self.goal_reward2, self.discount_reward, self.loss_reward, self.fire_reward,30,100,0.0001,0.2)
        qsession.explore_with_val_func(self.goal_index, self.goal_index2, self.start_index, self.walls_index, self.fire_index, self.no_of_states,self.states,self.root)
        #qsession.live_explore_with_state_value_func(self.goal_index, self.goal_index2, self.start_index, self.walls_index, self.fire_index, self.no_of_states,self.states,self.root)
    
    
    def newGame(self):
        rand_gen = np.random.randint(0, 105350)
        self.root.destroy()
        #board = Qboard()
        self.drawBoard(self.num_of_states,0.2,0.1,rand_gen)
        
        
        
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
       
       self.goal_index2 = np.random.randint(0, self.no_of_states)
       while(self.goal_index2 in self.walls_index or self.goal_index2 in self.fire_index or self.goal_index2 == self.goal_index):
            self.goal_index2 = np.random.randint(0, self.no_of_states)
            
       self.states[self.goal_index2].config(bg="green", text="GOAL2",justify="center")
       
       self.start_index = np.random.randint(0, self.no_of_states)
       while(self.start_index in self.walls_index or self.start_index in self.fire_index or self.start_index == self.goal_index or self.start_index == self.goal_index2):
           self.start_index = np.random.randint(0, self.no_of_states)
       self.states[self.start_index].config(bg="#B10DC9", text="START",justify="center")
      
       
       def get_num_states(e):
           self.num_of_states = int(e)
           
       def get_history(e):
           self.history = int(e)        
               
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
           
       def get_goal_slider_values2(e):
          self.goal_reward2 = float(e)
           #print(self.goal_reward)
       
           
           
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
       
       self.goal_slider1 = tk.Scale(self.settings, label="goal_reward2", from_=0, to=10, orient='horizontal', digits = 3, resolution=0.01, command=get_goal_slider_values2)
       self.goal_slider1.set(self.goal_reward2)
       self.goal_slider1.grid(row=0, column=5, padx=5, pady=5)
       
       self.playButton = tk.Button(self.play, text="START", font=('', 14), justify='center', pady=10, bg="#7FDBFF", command=self.playGame)
       self.playButton.grid(row=0, column=1, padx=10,pady=10)
       
       self.newButton = tk.Button(self.play, text="NEW GAME", font=('', 14), justify='center', pady=7, bg="orange", command=self.newGame)
       self.newButton.grid(row=0, column=2, padx=7,pady=7)
       
       self.num_states = tk.Scale(self.play, label="num_states", from_=4, to=50, orient='horizontal', command=get_num_states)
       self.num_states.set(self.num_of_states)
       self.num_states.grid(row=0, column=3, padx=7, pady=7)
       
       self.hist = tk.Scale(self.play, label="history", from_=1, to=self.num_of_states, orient='horizontal', command=get_history)
       self.hist.set(self.history)
       self.hist.grid(row=0, column=4, padx=7, pady=7)
       
       self.settings.pack()
       self.main_canvas.pack()
       self.play.pack()
       self.root.title("Qlearning")
       self.root.config(bg="black")
       self.root.mainloop()
       
      
       
      
       
       
       
board = Qboard()
states = board.drawBoard(5,0.1,0.1,34326)
