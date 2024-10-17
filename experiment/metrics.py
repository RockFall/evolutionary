class Metrics:
    def __init__(self):
        self.metrics = {}

    def add_metric(self, name, value):
        """
        Adiciona uma métrica aos resultados.
        """
        self.metrics[name] = value

    def get_metric(self, name):
        """
        Retorna o valor de uma métrica específica.
        """
        return self.metrics.get(name, None)
