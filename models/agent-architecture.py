'''
Copyright 2018 University of Minnesota
Bryan C. Runck
Department of Geography, Environment and Society


**Overview**
This class is the agent architecture for using word embeddings
as the basis for agent cognition. It consists of two objects:

1) agent object
2) word embedding object

The agent object relies on the word embeddings object to query
word vectors. It is designed this way to reduce memory use in
the full agent-based model.

'''

import numpy as np
import string

class agent_experiment_1(object):

  ''' State Variables / Data '''
  def __init__(self, word_embeddings_object, bag_of_words=None):
    self.word_embeddings = word_embeddings_object
    self.prompt_vector = []
    self.DO_options_vectors = []
    self.evaluations = []

  ''' Decision Methods '''
  def perceive(self, prompt, options):
    ''' 
      Inputs: 
      prompt - string
      options - python list of strings, each is an option 
      that the agent could choose
      If new information is perceived,
      agent forgets old options and perceptions.
      
      Returns: --
    '''
    self.evaluations = []
    self.prompt_vector = []
    self.DO_options_vectors = []

    self.prompt_vector = clean(prompt, self.word_embeddings)
    
    for DO in options:
        self.DO_options_vectors.append(clean(DO, self.word_embeddings))

    
  def evaluate_perceptions(self):
    '''
        Inputs: --
        Returns: array of similarity perceptions
    '''
    
    for DO in self.DO_options_vectors:
        self.evaluations.append(sum(self.prompt_vector * DO) / 
              (np.sqrt(sum(np.square(self.prompt_vector))) * 
               np.sqrt(sum(np.square(DO)))))
        
    return self.evaluations



'''
AXELROD Agent
'''

class agent_axelrod_model(object):
    
    def __init__(self, word_embeddings_object, string_of_words):
        self.word_embeddings = word_embeddings_object
        self.bag_of_words = string_of_words
        self.neighbor_sim_perception = None # Compute at each perceive_and_evaluate method call
        self.self_vector = None # Computed as the average vector of the complete bag_of_words
        self.average_similarity = 0 # average similarity to agent population
        self.local_similarity = 0
        self.create_self_vector() # construct vector from bag-of-words
    
    def perceive_and_evaluate(self, random_neighbor):
        self.neighbor_sim_perception = 0
        '''Calculate the cosine similarity between the agents' vectors '''
        self.neighbor_sim_perception = (sum(self.self_vector * random_neighbor.self_vector) / 
                (np.sqrt(sum(np.square(self.self_vector))) * 
                 np.sqrt(sum(np.square(random_neighbor.self_vector)))))
        
    def interact_and_learn(self, random_neighbor):
        try:
            
            if np.random.binomial(1, np.absolute(self.neighbor_sim_perception)+0.000001) == 1:
                ''' 
                    Target agent: look into neighbor agent and at random 
                    copy a word that is not in self.bag_of_words 

                1. create unique bags of words from between two agents 
                2. shuffle order
                3. choose first word in list -- append where brother is vv
                '''

                new_words=[]
                for word in random_neighbor.bag_of_words.split():
                    if word not in self.bag_of_words.split():
                        new_words.append(word)

                word_to_add=''
                try:
                    word_to_add = new_words[np.random.randint(0, len(new_words)-1)]
                except:
                    print("No new words")

                self.bag_of_words = self.bag_of_words + ' ' + word_to_add

                ## update the self vector
                self.create_self_vector()
        except:
            print("Too dissimilar")
    
    def create_self_vector(self):
        self.self_vector = clean(self.bag_of_words, self.word_embeddings)
    
    def calculate_average_similarity(self, other_agent):
        self.average_similarity = (self.average_similarity + (sum(self.self_vector * other_agent.self_vector) / 
                (np.sqrt(sum(np.square(self.self_vector))) * 
                 np.sqrt(sum(np.square(other_agent.self_vector)))))) / 2
    
    def calculate_local_similarity(self, other_agent):
        self.local_similarity = (self.local_similarity + (sum(self.self_vector * other_agent.self_vector) / 
                (np.sqrt(sum(np.square(self.self_vector))) * 
                 np.sqrt(sum(np.square(other_agent.self_vector)))))) / 2


'''
####### HELPER FUNCTIONS FOR NLP and Vector Operations 
'''


def clean(string_of_words, word_embeddings):
    '''
      Input: string of words
      Return: averaged vector representing string
      
      Process flow: 
      string of words > remove punctuation > remove numbers > 
      lowercase > remove stopwords > tokenize >
      intersect word embeddings with each word > calculate average embedding
    '''
    # Remove Punctuation
    string_of_words = remove_punctuation(string_of_words)

    # Remove Numbers
    string_of_words = remove_numbers(string_of_words)

    # Lowercase
    string_of_words = string_of_words.lower()

    # Remove Stopwords
    string_of_words = remove_stopwords(string_of_words)

    # Tokenize
    tokens = string_of_words.split()

   
    # Intersect word embeddings and Calculate average embedding
    average_word_vector = calculate_phrase_vector(tokens, word_embeddings)
    
    return average_word_vector


def remove_punctuation(s):
    '''
    Input: string
    Returns: string without punctuation
    '''
    return ''.join((char for char in s if char not in string.punctuation))


def remove_numbers(s):
    '''
    Input: string
    Returns: string without numbers
    '''
    return ''.join([i for i in s if not i.isdigit()])


def remove_stopwords(s, other_stoplist=None):
    '''
    Input: a string of words
    Returns: list of tokens with stopwords removed
    '''    
    if other_stoplist is not None:
        stoplist = set(other_stoplist.split())
    stoplist = set.union(set('for a of the and to is in ie'.split()))
    return ' '.join([word for word in s.split() if word not in stoplist])


def calculate_phrase_vector(word_set, embeddings):
    '''
    Input: list of words
    Output: average vector
    '''
    phrase_vector = np.zeros(embeddings.dimensions)
    
    for word in word_set:
        # goes through each word, finds the vector in the precomputed vector file, 
        # multiplies it by the frequency of that word, and then adds it to the phrase vector
        try:
            phrase_vector = np.add(phrase_vector, embeddings.get_embedding(word))
        except:
            print("Skipped", word, "in phrase vector")
    try:
        phrase_vector = np.divide(phrase_vector, len(word_set)) # averages the phrase vector by total number of words in phrase
    except:
        print("Phrase Vector 0")
        phrase_vector = np.zeros(embeddings.dimensions)
    
    return phrase_vector






'''
The word embeddings class serves as a way to hold the 
large embeddings file in memory
so multiple agents can access the 
same word representations.

To create a word embedding object, the object needs 
a file path pointed to GloVe word embeddings from 
https://nlp.stanford.edu/projects/glove/ as described 
in Pennington et al. 2014.

The word embeddings object has two methods:
1. load_embeddings( ) - this is used to load 
all of the embeddings into a dictionary. 
The file size is roughly 2 GB depending on 
the specific set of embeddings.

2. get_embedding( ) - this method takes a word string and returns an embedding.


'''


class word_embeddings(object):
    
    def __init__(self, file):
        self.file = file
        self.word_embeddings = None
        self.dimensions = None
    
    def load_embeddings(self):
        print("Loading Embeddings:", self.file)
        print("Go grab a beverage. This may take some time...")
        print("-----------------------------------------------")
        word_list = []
        vector_list = []
        with open(self.file) as word_vectors:
            i = 0
            for row in word_vectors:
                vector = np.array(row.rstrip("\n").rsplit(" ")[1:]).astype(float)
                word = row.rstrip("\n").rsplit(" ")[0:1]
                word_list.append(word[0])
                vector_list.append(vector)
                i=i+1
                if i%10000==0:
                    print("At word ", i, " : ", word[0])
                    #if i%200000 == 0:
                        #break
        self.dimensions = len(vector)        
        self.word_embeddings = dict(zip(word_list, vector_list))

    def get_embedding(self, word):
        try:
            return self.word_embeddings[word]
        except:
            print("Could not get ", word)
