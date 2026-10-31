from __future__ import annotations

from typing import Sequence, Union

import numpy as np

Landmark = tuple[int, Union[Sequence[float], np.ndarray], float]

class Grid:
    x_limits: tuple[float, float]
    y_limits: tuple[float, float]
    grid_cell_size: float #cm
    number_of_celles: list[int]
    grid_matrix: np.ndarray
    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float, grid_cell_size: float) -> None:
        self.x_limits = (x_min, x_max)
        self.y_limits = (y_min, y_max)
        self.grid_cell_size = grid_cell_size
        
        self.number_of_celles= [int((x_max-x_min)//grid_cell_size), int((y_max-y_min)//grid_cell_size)] #Beregner hvor mange celler vi skal have for at ramme cm og cellestørrelse
        
        self.grid_matrix = np.zeros((self.number_of_celles[0], self.number_of_celles[1]), dtype=bool) #lav matrix der svarer til grid størrelse med korrekt cellestørrlese, men kun med 0 values.
    
    def obstacles(self, landmarks_list: Sequence[Landmark], robot_radius: float) -> None:
        """
        Funktion der opdaterer matrix.
        Der tjekkes om hver celle er på en obstacle position, hvis ja ændres valure til 1.
        obstacles are equal to ``1`` or ``True`` in the grid, otherwise ``0`` or ``False``.    
        """     
        for i in range(self.number_of_celles[0]):
            for j in range(self.number_of_celles[1]):
                centroid = np.array([self.x_limits[0] + self.grid_cell_size * (i+0.5), 
                                     self.y_limits[0] + self.grid_cell_size * (j+0.5)])
    
                for id, centrum, radius in landmarks_list:
                    X, Y =centrum
                    center=np.array([X,Y])
                    added_radius = radius + robot_radius + 10.0 #landmark radius og robotradius + buffer, så vi ikke måler fra centrum men fra siden af de to.
                                                                
                    if np.linalg.norm(centroid - center) <= added_radius:
                        self.grid_matrix[i, j] = 1 #Vi opdatere matrix til 1, hvis celler har obstacle position
