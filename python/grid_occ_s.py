import numpy as np
import matplotlib.pyplot as plt

class Grid:
    
    def __init__(self, x_min, x_max, y_min, y_max, grid_cell_size):
        self.x_limits=[x_min, x_max]
        self.y_limits=[y_min, y_max]
        self.grid_cell_size=grid_cell_size
        
        self.grid_size = [x_max-x_min, y_max-y_min] 
        
        self.number_of_celles= [int(x/grid_cell_size) for x in self.grid_size] #Beregner hvor mange celler vi skal have for at ramme m og cellestørrelse
        
        self.grid_matrix= np.zeros((self.number_of_celles[0], self.number_of_celles[1])) #lav matrix der svarer til grid størrelse med korrekt cellestørrlese, men kun med 0.
        
    
    def obstacles(self, landmarks_list, robot_radius):
        """
        obstacles are =1 in the grid
        
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
                    added_radius = radius + robot_radius #det er landmark radius og robotradius, så vi ikke måler fra centrum men fra siden af de to.
                                                                
                    
                    if np.linalg.norm(centroid - center) <= added_radius:
                        self.grid_matrix[i, j] = 1 
            
class Node:
    
    def __init__(self, position):
        self.position=position #x, y 
        self.distance=0 #betegner distancen vi vælger, da vi mødte denne node
        self.node_path=[]
        self.parent=None
        
          

    #euclidean distance from node to target node
    def eucl_dist(self,target_node):
        return np.linalg.norm(np.array(target_node.position[:2]) - np.array(self.position[:2]))        
            
                        
class RRT:
    
    
    def __init__(self,
                 start, #start coords
                 goal,  #goal coords
                 robot_model, 
                 map,
                 moving_dist_limit=1, #Hvad vi max bevæger os ved ny node
                 grid_cell_size=0.05,
                 goal_sample_rate=5,    #Procentdel på hvor tit en random node skal være goal node
                 max_iter=500,          #Max forsøg på at finde path
                 ):

        self.start = Node(start) #create start node
        self.end = Node(goal) #create goal node
        self.robot = robot_model
        self.map = map
        
        self.min_rand = (map.x_limits[0],map.y_limits[0])
        self.max_rand = (map.x_limits[1],map.y_limits[1])

        self.moving_dist_limit = moving_dist_limit
        self.grid_cell_size=grid_cell_size
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter #Vi bruger max_iter ligesom dem, for at sætte grænse på hvor længe algoritmen må forsøge at finde path

        self.node_list = []
    
    def path_planning(self):

        self.node_list = [self.start]

        for i in range(self.max_iter):
            print(f"iteration{i}")
            random_node = self.random_node()
            nearest_node = self.nearest_node(self.node_list, random_node)

            new_node = self.steering(nearest_node, random_node, self.moving_dist_limit)
            
            #Hvis der ikke er obstacles på vejen til noden, tilføjer vi til node_list
            if self.check_collision(nearest_node,new_node):
                self.node_list.append(new_node)

            #Denne blok kode tager vi fra dem
            #try to steer towards the goal if we are already close enough
            if self.node_list[-1].eucl_dist(self.end) <= self.moving_dist_limit:
                final_node = self.steering(self.node_list[-1], self.end,
                                        self.moving_dist_limit)
                if self.check_collision(self.node_list[-1],final_node):
                    return self.generate_final_course(len(self.node_list) - 1)

        return None
    

    
    #Vi bruger den givne random node funktion, hvor målnode vælges en lille procentdel af tidel
    # def random_node(self):
    #     if np.random.randint(0, 100) > self.goal_sample_rate:
    #         rnd = Node(
    #             np.random.uniform((self.map.x_limits[0],self.map.y_limits[0]), (self.map.x_limits[1],self.map.y_limits[1]))
    #             )
    #     else:  #goal point sampling
    #         rnd = Node(self.end.position)
    #     return rnd
    
    def random_node(self):
        rnd = Node(
                np.random.uniform((self.map.x_limits[0],self.map.y_limits[0]), (self.map.x_limits[1],self.map.y_limits[1]))
                )
    
        return rnd
    
    
    def nearest_node(self,node_list, random_node):
        distance_list=[]
        for node in node_list:
            node_dist=node.eucl_dist(random_node) #For hver node i listen, beregner vi afstand til random node
            distance_list.append(node_dist)
            
        min_index = np.argmin(distance_list)

        return node_list[min_index]
    
    
    def steering(self, old_node, new_node, moving_dist_limit): #moving_dist_limit er hvad vi maksimal bevæger os når vi finder ny node
        print("steering")
        x1, y1 =old_node.position
        x2, y2 =new_node.position
        
        diff_x= x2-x1
        diff_y=y2-y1
        
        distance= old_node.eucl_dist(new_node)
        
        
        path_distance=min(distance, moving_dist_limit) #hvis distancen er mindre end vores limit, vælger vi distancen til noden.
        
        new_x = old_node.position[0] + path_distance * diff_x / distance
        new_y = old_node.position[1] + path_distance * diff_y / distance


        new_node.position = np.array([new_x,new_y])
        new_node.distance =path_distance

        new_node.node_path.append([new_x, new_y]) #tilføjer ny node til path
        new_node.parent =old_node
        
        return new_node
        
        
#tjekker om der er obstacles på vejen fra old_node til new_node
    def check_collision(self, old_node, new_node): 
        print("collision")
        x1, y1=old_node.position
        x2, y2=new_node.position
        distance=new_node.distance
    
        #antal celler på distancen
        number_cells = int(distance / self.grid_cell_size) + 1
    
        #for hver celle, laver vi et punkt på vejen som vi tjekker for obstacles.
        xs = np.linspace(x1, x2, number_cells) 
        ys = np.linspace(y1, y2, number_cells)
    
        coords=zip(xs,ys)
    
        #Nu skal vi så tjekke collision for hvert punkt(celle).
        for x,y in coords:
            x_cell=int((x-self.map.x_limits[0]) //self.grid_cell_size) #x-x.min//cell_size, dette er x_cell for punktet
            y_cell=int((y-self.map.y_limits[0]) //self.grid_cell_size) #y-y.min//cell_size, dette er y_cell for punktet

            if self.map.grid_matrix[x_cell][y_cell]==1:
                return False    #Collision
            
        return True
            
    #Denne kode har vi pt. også taget fra dem
    def generate_final_course(self, goal_ind):
        path = [self.end.position]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append(node.position)
            node = node.parent
        path.append(node.position)

        return path
    
    def simpler_path(self, path):
        """Vi laver denne funktion, så vi kan springe noder over i pathen, hvis der alligevel ikke er obstacles.
        På den måde undgår vi unødvendige rotationer."""

        simpler_path = [path[0]] #path skal starte ved start
        num_nodes =len(path)-1
        current_node_index = 0

        while current_node_index < num_nodes:

            furthest = current_node_index + 1 #Indtil vi har tjekket node tættere på, er dette den næste node 

            for node in range(num_nodes, current_node_index, -1): #Vi looper baglængs gennem path punkter og tjekker om det er collision fri path fra start til noden

                node1 = Node(path[current_node_index])
                node2 = Node(path[node])

                dx = node2.position[0] - node1.position[0]
                dy = node2.position[1] - node1.position[1]

                distance = np.sqrt(dx**2+ dy**2)

                node2.distance = distance 

                if self.check_collision(node1, node2): #Hvis vi møder node hvor vi ikke har collision, gør vi dette til næste node i path (så vi skipper måske nogle noder)
                    furthest = node
                    break

            simpler_path.append(path[furthest])
            current_node_index = furthest

        return simpler_path
    
#import Exercise_1 as ex1
    
def robot_path(arlo,path):
    """Funktion til at køre den simpler path"""
    executed_path = [path[0]]
    
    path_length=len(path)
    
    for i in range(path_length - 1):

        x1, y1 = path[i] #from node
        x2, y2 = path[i + 1]# til node

        dx = x2 - x1 
        dy = y2 - y1

        distance = np.sqrt(dx**2 + dy**2)

        desired_angle = np.degrees(np.arctan2(dy, dx))
        
       # ex1.rotate_inplace(arlo,degrees=desired_angle)

        # Kør frem
        #ex1.straight_ahead(arlo,meters=distance)
        
        executed_path.append([x2, y2])

    return executed_path

        
    
import robot_models
import Exercise_4_1_plot as ex4_1
import json
#import robot

arlo_radius = 0.225 #robot.ARLO_RADIUS

#Hent info fra json
with open('./python/coordinates.json') as file:
    data = json.load(file)
    landmarks=data
    landmarks_list=[]
    for landmark in landmarks:
        id=landmark["id"]
        center=landmark["center"]
        radius=landmark["radius"]
        landmarks_list.append((id,center,radius))
    print(landmarks_list)


def main():
    
    START_POINT=[0, 0]
    goal=[0,6]
    GRID_CELL_SIZE= 0.1
    
    
    map = Grid(x_min=-1,x_max=10,y_min=0, y_max=10, grid_cell_size=GRID_CELL_SIZE)
    map.obstacles(landmarks_list, robot_radius=arlo_radius) #generer vores landmarks som obstacles
    print()

    robot = robot_models.PointMassModel(ctrl_range=[-GRID_CELL_SIZE, GRID_CELL_SIZE])   #

    rrt = RRT(start=START_POINT
              ,goal=[0,6],
        robot_model=robot,
        map=map,
        moving_dist_limit=1,
        grid_cell_size=GRID_CELL_SIZE,
        )
    
    
    path = rrt.path_planning()
    
    
    if path is None:
            print("Cannot find path")
    else:
        print("found path!!")
        simpler_path=rrt.simpler_path(path)
        execute_path=robot_path(robot,simpler_path)
        
        
        fig, ax =plt.subplots()
        ex4_1.draw_landmarks(ax, landmarks_list, arlo_radius)
        
        
        executed_x = [p[0] for p in execute_path]
        executed_y = [p[1] for p in execute_path]

        ax.plot(executed_x,executed_y,'-',linewidth=2,label="Robot path")
        plt.grid(True)
        plt.pause(0.01)
        #ax.set_xlim(-10,10)
        #ax.set_ylim(-10,10)
        ax.scatter(goal[0],goal[1], color='green')
    
        plt.show()



if __name__ == '__main__':
    main()
    