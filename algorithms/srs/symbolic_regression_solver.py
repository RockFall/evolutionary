from datetime import datetime
import numpy as np

from srs.result import Result
from srs.evolutionary import EvolutionaryAlg
from srs.util import _protected_division


class SymbolicRegressionSolver:
    def __init__(self, X_size=2, pop_size=100, max_generations=30, max_tree_depth=7, min_tree_depth=2, crossover_rate=0.9, mutation_rate=0.05, elitism_size=10, tournament_size=2):
        self.evolutionary_alg = EvolutionaryAlg(X_size, 
                                                pop_size, 
                                                max_generations, 
                                                max_tree_depth, 
                                                min_tree_depth, 
                                                crossover_rate, 
                                                mutation_rate, 
                                                elitism_size, 
                                                tournament_size)
        
        self.results = []

    def set_params(self, X_size=2, pop_size=100, n_generations=30, max_tree_depth=7, min_tree_depth=2, crossover_rate=0.9, mutation_rate=0.05, elitism_size=10, tournament_size=2, **kwargs):
        self.evolutionary_alg = EvolutionaryAlg(X_size, 
                                                pop_size, 
                                                n_generations, 
                                                max_tree_depth, 
                                                min_tree_depth, 
                                                crossover_rate, 
                                                mutation_rate, 
                                                elitism_size, 
                                                tournament_size)

    def test_change(self):
        print("All ok 3")

    def fit(self, X, y, name=str(datetime.now())):
        # Evoluir até o número máximo de gerações a ser alcançado
        data = self.evolutionary_alg.Evolve(X, y)

        self.results.append((name, data))

    def predict(self, X, name=None, custom_phenotype=None):
        # Retornar um vetor com as predições para cada linha de X
        y_hat = np.zeros(X.shape[0])

        if name  != None:
            for result in self.results:
                if result[0] == name:
                    best_phenotype = result[1][-1]["best_all"]["phenotype"]
                    break
        else:
            best_phenotype = self.results[-1][1][-1]["best_all"]["phenotype"]

        if custom_phenotype != None:
            best_phenotype = custom_phenotype

        for i in range(X.shape[0]):
            try:
                y_hat[i] = eval(best_phenotype, globals(), {"x": X[i], "protec_div": _protected_division})
            except (OverflowError, ValueError, ZeroDivisionError) as e:
                return self._invalid_fitness_value
            
        return y_hat

    def score(self, y, y_hat):
        # Calcular a porcentagem de acerto
        return np.sum(y == y_hat) / y.shape[0]

    def get_predicted_expression(self):
        # Retornar a expressão do melhor indivíduo
        return self.results[-1][1][-1]["best_all"]["phenotype"]
    
    def get_best_param_per_generation(self, param_name, name = None):
        per_generation = []

        if name == None:
            name = self.results[-1][0]

        for result in self.results:
            if result[0] == name:
                for generation in result[1]:
                    per_generation.append(generation[param_name])
                break
        
        return per_generation
    
    def get_best_experiment(self, name):
        best = None
        best_name = None
        for result in self.results:
            if name in result[0]:
                if best == None or result[1][-1]["best_all_fitness"] < best[1][-1]["best_all_fitness"]:
                    best = result
                    best_name = result[0]
        return best_name
    
