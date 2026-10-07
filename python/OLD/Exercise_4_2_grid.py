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
        
        Vi tjekker for hvert celle i grid, om der ligger landmark+robot+buffer i cellen.
        Hvis landmark er i cellen sætter vi 0 til 1 i vores grid.
        
        """
         
        for i in range(self.number_of_celles[0]):
            for j in range(self.number_of_celles[1]):
                centroid = np.array([self.x_limits[0] + self.grid_cell_size * (i+0.5), 
                                     self.y_limits[0] + self.grid_cell_size * (j+0.5)])
    
        
                for id, centrum, radius in landmarks_list:
                    X, Y,_ =centrum
                    center=np.array([X,Y])
                    added_radius = radius + robot_radius + 0.1 #det er landmark radius og robotradius, så vi ikke måler fra centrum men fra siden af de to.
                                                                
                    
                    if np.linalg.norm(centroid - center) <= added_radius:
                        self.grid_matrix[i, j] = 1 

