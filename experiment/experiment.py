from .config import ExperimentConfigGroup
from .result import ExperimentResult
from tqdm.auto import tqdm
import copy

class Experiment:
    def __init__(self, problem_type="regression", namespace="experiment", backup_folder=None, save_backup=False):
        self.namespace = namespace
        self.problem_type = problem_type
        self.solver = None
        #self.dataset = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self._default_config_indices = []
        self.configurations = []  # ExperimentConfigGroup list
        self.results = ExperimentResult(problem_type=problem_type)

        self._current_configuration_group_idx = 0  # Para controle de interrupções
        self._current_configuration_idx = 0  # Para controle de interrupções
        self._current_absolute_idx = 0  # Para debug

        self.save_backup = save_backup
        self.backup_folder = backup_folder
        self.backup_file = f"{namespace}_backup.pkl"
        #self.overwrite_backup = overwrite_backup

    def setup(self, solver, X_train, y_train, X_test, y_test, problem_type=None, namespace=None, backup_folder=None, save_backup=False, custom_params=None):
        self.solver = solver
        #self.dataset = dataset
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self._current_configuration_group_idx = 0
        self._current_configuration_idx = 0
        self._current_absolute_idx = 0

        self.namespace = namespace if namespace else self.namespace
        self.problem_type = problem_type if problem_type else self.problem_type

        self.results = ExperimentResult(problem_type=problem_type)

        self.save_backup = save_backup if save_backup else self.save_backup
        self.backup_folder = backup_folder if backup_folder else self.backup_folder
        self.backup_file = f"{namespace}_backup.pkl"

        self.custom_params = custom_params

    def add_configuration(self, experiment_config):
        #if not isinstance(experiment_config, ExperimentConfigGroup):
        #    raise TypeError("O parâmetro deve ser uma instância de ExperimentConfigGroup.")
        self.configurations.append(experiment_config)

    def add_default_configuration(self, experiment_config):
        idx = len(self.configurations)
        self._default_config_indices.append(idx)
        self.configurations.append(experiment_config)

    def reset_configurations_to_default(self):
        self.configurations = [self.configurations[i] for i in self._default_config_indices]


    def run_all(self):
        # Carrega backup se existir
        backup_existed = self._load_backup()
            
        self.total_configurations = sum([len(config.configurations) for config in self.configurations])

        if not backup_existed:
            self._current_configuration_group_idx = 0
            self._current_configuration_idx = 0
            self._current_absolute_idx = 0

        with tqdm(total=self.total_configurations, desc="Configurations", position=0, leave=True) as config_pbar:
            for group_idx in range(self._current_configuration_group_idx, len(self.configurations)):
                self._current_configuration_group_idx = group_idx

                config_group = self.configurations[group_idx]
                total_configs_in_group = len(config_group.configurations)

                for config_idx in range(self._current_configuration_idx, total_configs_in_group):
                    self._current_configuration_idx = config_idx
                    config = config_group.configurations[config_idx]

                    # Inserting custom parameters
                    if self.custom_params:
                        for key, value in self.custom_params.items():
                            config[key] = value

                    # Atualiza a barra de Configurações
                    config_pbar.update(1)
                    config_pbar.set_postfix(config=f"{self._current_absolute_idx + 1}/{self.total_configurations}")

                    # Executa a configuração com barra de Iterações
                    self._run_configuration(config)

                    self._current_absolute_idx += 1
                    if self.save_backup:
                        self._save_backup()

                # Reseta o índice de configuração após cada grupo
                self._current_configuration_idx = 0

    def _run_configuration(self, config):
        n_iterations = config['n_iterations_per_config']
        # Barra de progresso para Iterações

        id_config = self.results.start_configuration(config)

        with tqdm(total=n_iterations, desc="Iterations", position=1, leave=False) as iter_pbar:
            for iteration in range(n_iterations):
                iter_pbar.update(1)
                iter_pbar.set_postfix(iteration=f"{iteration + 1}/{n_iterations}")

                # New instance per iteration
                solver = copy.deepcopy(self.solver)
                solver.set_params(**config)

                # Fit and predict
                solver.fit(self.X_train, self.y_train)
                prediction = solver.predict(self.X_test)

                y_scores = None
                if self.problem_type == 'classification':
                    # Try to get predicted probabilities
                    if hasattr(solver, 'predict_proba'):
                        y_scores = solver.predict_proba(self.X_test)
                    # If predict_proba is not available, try decision_function
                    elif hasattr(solver, 'decision_function'):
                        y_scores = solver.decision_function(self.X_test)
                    else:
                        # If neither method is available, y_scores remains None
                        #print("Warning: Solver does not provide predict_proba or decision_function.") # TODO: ADD LOGGER
                        pass

                # Metrics
                self.results.add_iteration(id_config, solver, prediction, self.y_test, y_scores, self.X_test)

        self.results.end_configuration(id_config)
                

    def get_results(self):
        return self.results.get_results()

    def _save_backup(self):
        import pickle
        backup_data = {
            '_current_configuration_group_idx': self._current_configuration_group_idx,
            '_current_configuration_idx': self._current_configuration_idx,
            '_current_absolute_idx': self._current_absolute_idx,
            'configurations': self.configurations,
            'results': self.results
        }
        backup_path = f"{self.backup_folder}/{self.backup_file}" if self.backup_folder else self.backup_file
        with open(backup_path, 'wb') as f:
            pickle.dump(backup_data, f)

    def _delete_backup(self):
        import os
        backup_path = f"{self.backup_folder}/{self.backup_file}" if self.backup_folder else self.backup_file
        if os.path.exists(backup_path):
            os.remove(backup_path)

    def _load_backup(self):
        import os
        backup_path = f"{self.backup_folder}/{self.backup_file}" if self.backup_folder else self.backup_file
        if os.path.exists(backup_path):
            import pickle
            with open(self.backup_path, 'rb') as f:
                backup_data = pickle.load(f)
            self._current_configuration_group_idx = backup_data['_current_configuration_group_idx']
            self._current_configuration_idx = backup_data['_current_configuration_idx']
            self._current_absolute_idx = backup_data['_current_absolute_idx']
            self.configurations = backup_data['configurations']
            self.results = backup_data['results']
            #print(f"Retomando do backup na configuração {self._current_configuration_idx}") # TODO: ADD LOGGER
            return True
        else:
            #print("Nenhum backup encontrado, iniciando do início") # TODO: ADD LOGGER
            return False

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


"""
%%html
<style>
.cell-output-ipywidget-background {
    background-color: transparent !important;
}
:root {
    --jp-widgets-color: var(--vscode-editor-foreground);
    --jp-widgets-font-size: var(--vscode-editor-font-size);
}  
</style>
"""