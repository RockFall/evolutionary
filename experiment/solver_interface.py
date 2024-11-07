from abc import ABC, abstractmethod

class SolverInterface(ABC):
    @abstractmethod
    def fit(self, X, y):
        """Treina o solver nos dados fornecidos.
        
        Args:
            X (array-like): Dados de entrada.
            y (array-like): Saídas desejadas.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """Realiza previsões para os dados de entrada.
        
        Args:
            X (array-like): Dados de entrada para prever.
            
        Returns:
            array-like: Previsões do solver.
        """
        pass

    @abstractmethod
    def set_params(self, **params):
        """Configura os parâmetros do solver.
        
        Args:
            **params: Parâmetros a serem configurados.
        """
        pass

    @abstractmethod
    def get_metrics(self):
        """Obtém as métricas de avaliação do solver após o treinamento.
        
        Returns:
            dict: Métricas relevantes, como acurácia, erro, etc.
        """
        pass
