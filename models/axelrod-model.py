'''
Copyright 2018 University of Minnesota
Bryan C. Runck
Department of Geography, Environment and Society

Overview: This python file contains the object class for a modified version of the Axelrod 1997 model.
'''
import numpy as np
from matplotlib import pylab as plt

class axelrod_model(object):
    
    def __init__(self, agent_class, culture_size, world_size_rows, world_size_columns, simulation_duration, random_number_seed):
        self.agent_class = agent_class #the object defining agent decision-making process
        self.culture_size = culture_size #number of words starting in agents
        self.world_size_rows = world_size_rows # rows in world
        self.world_size_columns = world_size_columns # columns in world
        self.simulation_duration = simulation_duration # number of time steps in model
        self.set_of_world_positions = [] # used to randomize order for each iteration
        self.random_number_seed = random_number_seed
        self.model_world = [] # row array that will be populated with column arrays of agents

        self.model_similarity = []
        
        
    def initialize(self, word_embeddings_object, text):
        '''
        Input: natural language text
        
        Model initializes size of the model world and populates it with agents built from provided text.
        '''

        for i in range(0, self.world_size_rows):
            self.model_world.append([])
            for j in range(0, self.world_size_columns):
                split_text=text.split()
                np.random.shuffle(split_text)
                string_of_words = ' '.join(split_text[0:self.culture_size])
                self.set_of_world_positions.append([i,j])
                self.model_world[i].append(self.agent_class(word_embeddings_object, string_of_words))

                
    def run(self):
        '''
        0. For the duration specified at the beginning of the model:
        1. Create agent list in random order to iterate through
        2. For each agent, choose a neighbor at random.
        3. Agents interact with a probability proportional to their self_vector similarities
        4. Update agents' average similarity to the population
        '''
        np.random.seed(self.random_number_seed)
        
        self.model_similarity = []
        for iteration in range(0, self.simulation_duration):

            np.random.shuffle(self.set_of_world_positions) # reshuffle the order of target agents
            
            for target_agent_position in self.set_of_world_positions:
                target_agent = self.model_world[target_agent_position[0]][target_agent_position[1]]
                
                ''' Find neighbors ''' 
                neighbors = self.find_neighbors(target_agent_position)
                
                ''' Select a random neighbor '''
                np.random.shuffle(neighbors)
                random_neighbor_loc = neighbors[0]
                random_neighbor = self.model_world[random_neighbor_loc[0]][random_neighbor_loc[1]]
                
                '''Perceive and Interact with Random Neighbor'''
                target_agent.perceive_and_evaluate(random_neighbor)
                target_agent.interact_and_learn(random_neighbor)
                
                
            ''' Calculate Agent Similarity'''
            if iteration % 5==0:
                print("Iteration: ", iteration)
                self.calculate_global_similarity()
                self.calculate_local_similarity()
                self.report_local_similarities()
   

    def find_neighbors(self, target_agent_position):
        '''
        Input: target agent
        Returns: A list of row, column duples for each neighbor
        '''
        rows=[]
        columns=[]
        for i in range(-1, 2):
            for j in range(-1,2):
                if (
                    (
                        (target_agent_position[0]+i != target_agent_position[0]) | 
                        (target_agent_position[1]+j != target_agent_position[1])
                    ) &
                    (
                        (target_agent_position[0]+i >= 0) & (target_agent_position[1]+j >= 0)
                    ) &
                    (
                        (target_agent_position[0]+i < self.world_size_rows) & 
                        (target_agent_position[1]+j < self.world_size_columns)) 
                    ):
                    rows.append(target_agent_position[0]+i)
                    columns.append(target_agent_position[1]+j)

        return list(zip(rows,columns))
    
                
    def calculate_global_similarity(self):
        '''
        Input: target agent; 
        loops through all other agents in world and calculates average similarity
        '''
        for target_agent_position in self.set_of_world_positions:
            target_agent = self.model_world[target_agent_position[0]][target_agent_position[1]]

            for each_row in self.model_world:
                for each_column in each_row:
                    target_agent.calculate_average_similarity(each_column)
            
            self.model_similarity.append(target_agent.average_similarity)
                        
    def calculate_local_similarity(self):
        '''
        Input: target agent; 
        loops through all other agents in world and calculates average similarity
        for the neighborhoods of interaction
        '''
        for target_agent_position in self.set_of_world_positions:
            target_agent = self.model_world[target_agent_position[0]][target_agent_position[1]]
            neighbor_list = self.find_neighbors(target_agent_position)
            
            for neighbor_location in neighbor_list:
                target_agent.calculate_local_similarity(self.model_world[neighbor_location[0]][neighbor_location[1]])

    def report_local_similarities(self):
        
        local_sims=[]
        for i in range(0, len(self.model_world)):
            local_sims.append([])
            for j in range(0, len(self.model_world[0])):
                local_sims[i].append(self.model_world[i][j].local_similarity)
        plt.imshow(local_sims, vmin=0, vmax=1)
        plt.colorbar()
        plt.show()
        
        
                
    
                