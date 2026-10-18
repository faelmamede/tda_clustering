import gudhi
import time

import my_constants

class TDACalculator:
    def __init__(self, points, MAX_EDGE_LENGTH=my_constants.DEFAULT_MAX_EDGE_LENGTH, SPARSE=my_constants.DEFAULT_SPARSE, MAX_DIMENSION=my_constants.MAX_DIMENSION, HOMOLOGY_COEFF_FIELD=my_constants.HOMOLOGY_COEFF_FIELD):
        self.points = points 

        self.MAX_EDGE_LENGTH = float(MAX_EDGE_LENGTH)
        self.SPARSE = float(SPARSE)
        self.MAX_DIMENSION = MAX_DIMENSION
        self.HOMOLOGY_COEFF_FIELD = HOMOLOGY_COEFF_FIELD

    def create_rips_complex(self):
        start_time = time.time()

        rips = gudhi.RipsComplex(points=self.points, max_edge_length=self.MAX_EDGE_LENGTH, sparse=self.SPARSE)
        end_time = time.time()

        print(f"Create RipsComplex - Execution Time: {end_time-start_time:.4f} sec")
        return rips
    
    def create_simplex_tree(self, rips):
        start_time = time.time()
        simplex_tree = rips.create_simplex_tree(max_dimension=my_constants.MAX_DIMENSION)
        end_time = time.time()

        print(f"Create simplex tree - Execution Time: {end_time-start_time:.4f} sec")
        return simplex_tree
    
    def create_persistence(self, simplex_tree):
        start_time = time.time()
        persistence = simplex_tree.persistence(homology_coeff_field=my_constants.HOMOLOGY_COEFF_FIELD, min_persistence=0)
        end_time = time.time()

        print(f"Create persistence - Execution Time: {end_time-start_time:.4f} sec")
        return persistence
    
    def calculate_flag_persistence_generators(self, simplex_tree):
        start_time = time.time()
        persistence_generators = simplex_tree.flag_persistence_generators()
        end_time = time.time()

        print(f"Calculated flag persistence generators - Execution Time: {end_time-start_time:.4f} sec")
        return persistence_generators
    
    def calculate_intervals_of_persistence(self, simplex_tree):
        start_time = time.time()
        birth_death_intervals = simplex_tree.persistence_intervals_in_dimension(my_constants.DIMENSION)
        end_time = time.time()

        print(f"Calculated birth and death intervals of persistences - Execution Time: {end_time-start_time:.4f} sec\n")
        return birth_death_intervals

    def apply(self):
        print("\nAPLYING TDA...\n")

        start_time = time.time()
        rips = self.create_rips_complex()
        simplex_tree = self.create_simplex_tree(rips)
        persistence = self.create_persistence(simplex_tree)
        persistence_generators = self.calculate_flag_persistence_generators(simplex_tree)
        birth_death_intervals = self.calculate_intervals_of_persistence(simplex_tree)
        end_time = time.time()

        print(f"Total - Execution Time: {end_time-start_time:.4f} sec")

        return persistence, persistence_generators, birth_death_intervals