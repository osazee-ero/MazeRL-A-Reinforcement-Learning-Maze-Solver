import tkinter as tk
import numpy as np
import math
import Qlearning_2Actions as ql

class Qboard:
    def __init__(self):
        self.discount_reward = 0.5
        self.goal_reward = 1
        self.goal_reward2 = 1
        self.loss_reward = 0.01
        self.num_of_states = 5
        self.history = 2
        
    def playGame(self):
        qsession = ql.Qlearn(self.goal_reward, self.goal_reward2, self.discount_reward, self.loss_reward, 1e-48, 100, self.history)
        qsession.explore_with_state_value_func(self.goal_index, self.goal_index2, self.start_index, self.no_of_states,self.states,self.root)
        
    def newGame(self):
        
        rand_gen = np.random.randint(0, 100000)
        self.root.destroy()
        #board = Qboard()
        board.drawBoard(self.num_of_states,rand_gen)
        
        
    def drawBoard(self,gridsize,random_state):
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
       
       for i in range(0, 1):
           for j in range(0, gridsize_):
               self.Button = tk.Button(self.main_canvas, text="MOV", justify="center")
               self.Button.grid(row=i, column=j, ipadx=5, ipady=5)
               self.states.append(self.Button)
               self.no_of_states += 1 
       
      
       self.goal_index = np.random.randint(0, self.no_of_states)
            
       self.states[self.goal_index].config(bg="green", text="GOAL",justify="center")
       
       self.goal_index2 = np.random.randint(0, self.no_of_states)
       while(self.goal_index2 == self.goal_index):
           self.goal_index2 = np.random.randint(0, self.no_of_states)
       self.states[self.goal_index2].config(bg="green", text="GOAL2",justify="center")
           
       self.start_index = np.random.randint(0, self.no_of_states)
       while(self.start_index == self.goal_index or self.start_index == self.goal_index2):
           self.start_index = np.random.randint(0, self.no_of_states)
       self.states[self.start_index].config(bg="#B10DC9", text="START",justify="center")
      
               
       def get_num_states(e):
           self.num_of_states = int(e)
           
       def get_history(e):
           self.history = int(e)
           
           
       def get_discount_slider_values(e):
           self.discount_reward = float(e)
           
           
       def get_goal_slider_values(e):
           self.goal_reward = float(e)
    
           
       def get_loss_slider_values(e):
           self.loss_reward = float(e)
          
           
       def get_goal_slider_values2(e):
          self.goal_reward2 = float(e)
       

       
       self.discount_slider = tk.Scale(self.settings, label="discount", from_=0, to=1, orient='horizontal', digits = 3, resolution=0.01, command=get_discount_slider_values)
       self.discount_slider.set(self.discount_reward)
       self.discount_slider.grid(row=0, column=1, padx=5, pady=5)
       
       self.goal_slider = tk.Scale(self.settings, label="goal_reward", from_=1, to=10, orient='horizontal', digits = 3, resolution=0.01, command=get_goal_slider_values)
       self.goal_slider.set(self.goal_reward)
       self.goal_slider.grid(row=0, column=2, padx=5, pady=5)
       
       self.goal_slider1 = tk.Scale(self.settings, label="goal_reward2", from_=1, to=10, orient='horizontal', digits = 3, resolution=0.01, command=get_goal_slider_values2)
       self.goal_slider1.set(self.goal_reward2)
       self.goal_slider1.grid(row=0, column=3, padx=5, pady=5)
       
       self.loss_slider = tk.Scale(self.settings, label="loss_reward", from_=0, to=1, orient='horizontal', digits = 3, resolution=0.01, command=get_loss_slider_values)
       self.loss_slider.set(self.loss_reward)
       self.loss_slider.grid(row=0, column=4, padx=5, pady=5)
       
       self.playButton = tk.Button(self.play, text="START", font=('', 14), justify='center', pady=7, bg="#7FDBFF", command=self.playGame)
       self.playButton.grid(row=0, column=1, padx=7,pady=7)
       
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
board.drawBoard(5,0)
