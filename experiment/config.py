class ExperimentConfigGroup:
    def __init__(
        self,
        n_iterations_per_config=1,
        pop_size=None,
        n_generations=None,
        mutation_rate=None,
        crossover_rate=None,
        selection_type=None,
        mutation_method=None,
        crossover_method=None,
        initialization_strategy=None,
        elitism_size=None,
        elitism_rate=None,
        tournament_size=None,
        min_tree_depth=None,
        max_tree_depth=None,
        n_features=None
        # Adicione outros hiperparâmetros possíveis aqui
        # Cada um com valor padrão None ou um valor específico
    ):
        """
        Inicializa um objeto ExperimentConfig com os hiperparâmetros necessários para a execução de um experimento.
        """
        # Hiperparâmetros
        self.n_iterations_per_config = n_iterations_per_config
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.selection_type = selection_type
        self.mutation_method = mutation_method
        self.crossover_method = crossover_method
        self.initialization_strategy = initialization_strategy
        self.elitism_size = elitism_size
        self.elitism_rate = elitism_rate
        self.tournament_size = tournament_size
        self.min_tree_depth = min_tree_depth
        self.max_tree_depth = max_tree_depth
        self.n_features = n_features
        # Adicione outros hiperparâmetros aqui

        # Validar os parâmetros
        self.validate_parameters()
        self.configurations = self.expand_configurations()

    def validate_parameters(self):
        # Verifica se pelo menos um valor foi definido para cada hiperparâmetro necessário
        required_params = ['pop_size', 'n_generations', 'mutation_rate', 'crossover_rate', 'selection_type']
        for param in required_params:
            if getattr(self, param) is None:
                raise ValueError(f"O hiperparâmetro '{param}' é obrigatório e não foi definido.")

    def get_hyperparameters(self):
        # Retorna um dicionário com os hiperparâmetros
        # Faça o codigo sem hard code
        return {p : getattr(self, p) for p in dir(self) if not p.startswith('_') and not callable(getattr(self, p))}
    
    def expand_configurations(self):
        # Gerar todas as combinações de hiperparâmetros
        from itertools import product

        # Filtrar hiperparâmetros que são listas para combinar
        hyperparams = self.get_hyperparameters()
        param_lists = {k: v if isinstance(v, list) else [v] for k, v in hyperparams.items() if v is not None}
        keys = list(param_lists.keys())
        values = list(param_lists.values())

        # but add n_iterations_per_config at beggining of each
        combinations = [dict({'n_iterations_per_config': self.n_iterations_per_config}, **dict(zip(keys, combination))) for combination in product(*values)]
        return combinations
