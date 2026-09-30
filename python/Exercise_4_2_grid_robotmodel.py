import numpy as np

class Grid:
    def __init__(self, x_min, x_max, y_min, y_max, grid_cell_size):
        self.x_limits = (x_min, x_max)
        self.y_limits = (y_min, y_max)
        self.grid_cell_size = grid_cell_size
        
        self.number_of_celles= [int((x_max-x_min)//grid_cell_size), int((y_max-y_min)//grid_cell_size)] #Beregner hvor mange celler vi skal have for at ramme m og cellestørrelse
        
        self.grid_matrix = np.zeros((self.number_of_celles[0], self.number_of_celles[1]), dtype=bool) #lav matrix der svarer til grid størrelse med korrekt cellestørrlese, men kun med 0.
        
    
    def obstacles(self, landmarks_list, robot_radius):
        """
        obstacles are equal to ``1`` or ``True`` in the grid, otherwise ``0`` or ``False``.
        
        centrum for celle (i,j).
        Calculation i row dimension is x_min + grid størrelse * rækkenummer+ 0.5 (så vi lander i midten)
        Eksempel:
        Hvis x_min =0 og y_min=2 og cellestørrelse er 0.3, så ligger celle (1,1)'s centrum:
        (0+0.3*(1+0,5),2+0.3*(1+0,5))
        
        Vi tjekker for hvert celle i grid, om der ligger landmark i cellen.
        Hvis landmark er i cellen sætter vi 0 til 1 i vores grid.
        
        """
         
        for i in range(self.number_of_celles[0]):
            for j in range(self.number_of_celles[1]):
                centroid = np.array([self.x_limits[0] + self.grid_cell_size * (i+0.5), 
                                     self.y_limits[0] + self.grid_cell_size * (j+0.5)])
    
        
                for id, centrum, radius in landmarks_list:
                    X, Y,_ =centrum
                    center=np.array([X,Y])
                    buffer_radius=0.1
                    added_radius = radius + robot_radius + buffer_radius #det er landmark radius og robotradius, så vi ikke måler fra centrum men fra siden af de to.
                                                                
                    
                    if np.linalg.norm(centroid - center) <= added_radius:
                        self.grid_matrix[i, j] = 1 

"""
Some simple first-order robot dynamics models

This provides an extendable interface to make the simulation of robot motion independent of planning algorithms
The most basic RRT implementation just needs PointMassModel
"""
import numpy as np

class RobotModel:

    def __init__(self, ctrl_range) -> None:
        #record the range of action that can be used to integrate the state
        self.ctrl_range = ctrl_range
        return
    
    def forward_dyn(self, x, u, T):
        #need to return intergated path of X with u along the horizon of T
        return NotImplementedError

    def inverse_dyn(self, x, x_goal, T):
        #return dynamically feasible path to move to the x_goal as close as possible
        return NotImplementedError

class PointMassModel(RobotModel):
    #Note Arlo is differential driven and may be simpler to avoid Dubins car model by rotating in-place to direct and executing piecewise straight path  
    #this is the "noise-free" motion model: shift x according to a command of u (velocity*dt).
    #it supports u of T steps and return the full trajectory. Just use T=1 if only one step is considered.
    def forward_dyn(self, x, u, T):
        path = [x]
        #note u must have T ctrl to apply
        for i in range(T):
            x_new = path[-1] + u[i] #u is velocity command here
            path.append(x_new)    
        
        return path[1:]

    def inverse_dyn(self, x, x_goal, T):
        #this is the inverse of motion model: what u of T steps can realize x_goal or move as close as possible given the current state x
        #return realized trajectory. this would be used to check if the trajectory would be incollision or not.
        #for point mass, the path is just a straight line by taking full ctrl_range at each step
        dist = np.linalg.norm(x_goal-x)
        dir = (x_goal-x)/dist
        #we cannot move more than the maximum control range at each step; clip it to make move as close as possible
        u = np.array([dir*min( [self.ctrl_range[1], dist/float(T)] ) for _ in range(T)])

        return self.forward_dyn(x, u, T)