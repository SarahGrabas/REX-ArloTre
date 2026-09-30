import numpy as np

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
    """Jo højere max_iter vi har, jo lavere moving distance limit, kan vi have hvor den stadig finder path.
    Dette gør måske path mere præcis??"""
    
    def __init__(self,
                 start, #start coords
                 goal,  #goal coords 
                 map,
                 moving_dist_limit=0.5, #Hvad vi max bevæger os ved ny node
                 grid_cell_size=0.05,
                 goal_sample_rate=5,    #Procentdel på hvor tit en random node skal være goal node
                 max_iter=500,          #Max forsøg på at finde path
                 ):

        self.start = Node(start) #create start node
        self.end = Node(goal) #create goal node
        self.map = map
        
        self.moving_dist_limit = moving_dist_limit
        self.grid_cell_size=grid_cell_size
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter #Vi bruger max_iter ligesom dem, for at sætte grænse på hvor længe algoritmen må forsøge at finde path

        self.node_list = []
    
    def path_planning(self):

        self.node_list = [self.start] #til start har vi kun startnode

        for i in range(self.max_iter):
            print(f"iteration{i}")
            random_node = self.random_node()
            nearest_node = self.nearest_node(self.node_list, random_node)

            new_node = self.steering(nearest_node, random_node, self.moving_dist_limit)
            
            #Hvis der ikke er obstacles på vejen til noden, tilføjer vi til node_list
            if self.check_collision(nearest_node,new_node):
                self.node_list.append(new_node)

            #Denne blok kode har vi taget vi fra dem
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
        rnd = Node(np.random.uniform((self.map.x_limits[0],self.map.y_limits[0]), 
                                     (self.map.x_limits[1],self.map.y_limits[1])))
        
        
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
        """Funktion til at checke collision.
        Vi opretter punkter i hvert gridcell på distance mellem to noder.
        Vi tjekker obstacles på hvert af disse punkter"""
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
        
        path.reverse()

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

                from_node = Node(path[current_node_index])
                to_node = Node(path[node])
                
                x1,y1 = from_node.position
                x2,y2 =to_node.position

                dx = x2-x1 
                dy = y2-y1

                distance = np.sqrt(dx**2+ dy**2)

                to_node.distance = distance 

                if self.check_collision(from_node, to_node): #Hvis vi møder node hvor vi ikke har collision, gør vi dette til næste node i path (så vi skipper måske nogle noder)
                    furthest = node
                    break

            simpler_path.append(path[furthest])
            current_node_index = furthest

        return simpler_path