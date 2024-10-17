import matplotlib.pyplot as plt

class Visualizer:
    @staticmethod
    def plot_fitness_progress(results):
        """
        Plota o progresso da fitness ao longo das gerações.
        """
        generations = range(len(results))
        fitness_values = [result['metric_value'] for result in results]

        plt.figure(figsize=(10, 6))
        plt.plot(generations, fitness_values, marker='o')
        plt.title('Fitness Progress over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_diversity_progress(results):
        """
        Plota a diversidade da população ao longo das gerações.
        """
        generations = range(len(results))
        diversity_values = [result['diversity'] for result in results]

        plt.figure(figsize=(10, 6))
        plt.plot(generations, diversity_values, marker='o')
        plt.title('Diversity Progress over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Diversity')
        plt.grid(True)
        plt.show()
