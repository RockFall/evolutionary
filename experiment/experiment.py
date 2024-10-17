from .config import ExperimentConfigGroup
import copy

class Experiment:
    def __init__(self):
        self.solver = None
        self.dataset = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.configurations = []  # Lista de ExperimentConfigGroup
        self.results = []
        self.current_configuration_group_idx = 0  # Para controle de interrupções
        self.current_configuration_idx = 0  # Para controle de interrupções
        self.current_absolute_idx = 0  # Para debug
        self.backup_file = 'experiment_backup.pkl'  # Caminho do arquivo de backup

    def setup(self, solver, dataset, X_train, y_train, X_test, y_test):
        self.solver = solver
        self.dataset = dataset
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def add_configuration(self, experiment_config):
        #if not isinstance(experiment_config, ExperimentConfigGroup):
        #    raise TypeError("O parâmetro deve ser uma instância de ExperimentConfigGroup.")
        self.configurations.append(experiment_config)

    def run_all(self):
        # Carrega backup se existir
        self._load_backup()

        total_config_groups = len(self.configurations)
        self.total_configurations = sum([len(config.configurations) for config in self.configurations])
        self.current_absolute_idx = 0

        for group_idx in range(self.current_configuration_idx, total_config_groups):
            self.current_configuration_group_idx = group_idx

            print(f"# Executando grupo de configurações {group_idx+1}/{total_config_groups}")
            for config_idx in range(self.current_configuration_idx, len(self.configurations[group_idx].configurations)):
                self.current_configuration_idx = config_idx

                print(f"## Executando configuração {config_idx+1}/{len(self.configurations[group_idx].configurations)} do grupo {group_idx+1}")
                self._run_configuration(self.configurations[group_idx].configurations[config_idx])
                self._save_backup()

                self.current_absolute_idx += 1

    def _run_configuration(self, config):
        print(config)
        for iteration in range(config['n_iterations_per_config']):
            print(f"  ### Iteração {iteration+1}/{config['n_iterations_per_config']} ({self.current_absolute_idx}/{self.total_configurations} configurações")
            
            # New instance per iteration
            solver = copy.deepcopy(self.solver)
            solver.set_params(**config)

            # Fit e predict
            solver.fit(self.X_train, self.y_train)
            prediction = solver.predict(self.X_test)

            # Metrics
            from sklearn.metrics import root_mean_squared_error
            rmse = root_mean_squared_error(self.y_test, prediction)
            metrics = solver.get_metrics() if hasattr(solver, 'get_metrics') else {}
            # Armazena o resultado
            result = {
                'config': config.copy(),
                'iteration': iteration + 1,
                'rmse': rmse,
                'metrics': metrics,
                # Adicione outras métricas ou informações conforme necessário
            }
            self.results.append(result)

    def get_results(self):
        return self.results

    def _save_backup(self):
        import pickle
        backup_data = {
            'current_configuration_idx': self.current_configuration_idx,
            'results': self.results
        }
        with open(self.backup_file, 'wb') as f:
            pickle.dump(backup_data, f)

    def _load_backup(self):
        import os
        if os.path.exists(self.backup_file):
            import pickle
            with open(self.backup_file, 'rb') as f:
                backup_data = pickle.load(f)
            self.current_configuration_idx = backup_data['current_configuration_idx']
            self.results = backup_data['results']
            print(f"Retomando do backup na configuração {self.current_configuration_idx}")
        else:
            print("Nenhum backup encontrado, iniciando do início")
            self.current_configuration_idx = 0
            self.results = []

    def print_all_configurations(self):
        from itertools import product

        for idx, experiment_config in enumerate(self.configurations):
            print(f"Grupo de configurações: {idx+1}/{len(self.configurations)}")
            hyperparams = experiment_config.get_hyperparameters()
            # Filtrar hiperparâmetros que são listas para combinar
            param_lists = {k: v if isinstance(v, list) else [v] for k, v in hyperparams.items() if v is not None}
            keys = list(param_lists.keys())
            values = list(param_lists.values())

            combinations = [dict(zip(keys, combination)) for combination in product(*values)]

            total_combinations = len(combinations)
            for config_idx, cnfiguration in enumerate(combinations):
                print(f"  Configuração {config_idx+1}/{total_combinations}: {cnfiguration}")

