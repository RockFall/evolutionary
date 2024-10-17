class ExperimentRunner:
    def __init__(self, experiment):
        """
        Inicializa o gerenciador de experimentos.
        - experiment: Instância de Experiment.
        """
        self.experiment = experiment
        self.results = []

    def run_experiments(self):
        """
        Executa experimentos com diferentes combinações de parâmetros.
        """
        parameter_combinations = self._generate_parameter_combinations(self.experiment.config.parameters)
        
        for params in parameter_combinations:
            Logger.log(f'Running experiment with parameters: {params}')
            
            # Atualiza a configuração do experimento com os novos parâmetros
            self.experiment.config.parameters = params
            
            # Executa o experimento
            result = self.experiment.run()
            self.results.append({'params': params, 'result': result})

        return self.results

    def _generate_parameter_combinations(self, param_dict):
        """
        Gera todas as combinações possíveis de parâmetros.
        """
        keys = param_dict.keys()
        values = (param_dict[key] for key in keys)
        return [dict(zip(keys, combination)) for combination in itertools.product(*values)]
