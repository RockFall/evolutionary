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
        self.results = ExperimentResult()

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

        self.results = []
        self._current_configuration_group_idx = 0
        self._current_configuration_idx = 0
        self._current_absolute_idx = 0

        self.namespace = namespace if namespace else self.namespace
        self.problem_type = problem_type if problem_type else self.problem_type

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
        self._load_backup()

        total_config_groups = len(self.configurations)
        self.total_configurations = sum([len(config.configurations) for config in self.configurations])
        self._current_absolute_idx = 0

        with tqdm(total=self.total_configurations, desc="Configurations", position=0, leave=True) as config_pbar:
            for group_idx in range(self._current_configuration_group_idx, total_config_groups):
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

                    if self.save_backup:
                        self._save_backup()
                    self._current_absolute_idx += 1

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

                # Metrics
                self.results.add_iteration(id_config, solver, prediction, self.y_test)

        self.results.end_configuration(id_config)
                

    def get_results(self):
        return self.results

    def _save_backup(self):
        import pickle
        backup_data = {
            '_current_configuration_idx': self._current_configuration_idx,
            'results': self.results
        }
        backup_path = f"{self.backup_folder}/{self.backup_file}" if self.backup_folder else self.backup_file
        with open(backup_path, 'wb') as f:
            pickle.dump(backup_data, f)

    def _load_backup(self):
        import os
        backup_path = f"{self.backup_folder}/{self.backup_file}" if self.backup_folder else self.backup_file
        if os.path.exists(backup_path):
            import pickle
            with open(self.backup_file, 'rb') as f:
                backup_data = pickle.load(f)
            self._current_configuration_idx = backup_data['_current_configuration_idx']
            self.results = backup_data['results']
            print(f"Retomando do backup na configuração {self._current_configuration_idx}")
        else:
            print("Nenhum backup encontrado, iniciando do início")
            self._current_configuration_idx = 0
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