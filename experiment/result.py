import pandas as pd
from sklearn.metrics import (
    mean_squared_error, root_mean_squared_error,mean_absolute_error, r2_score,
    explained_variance_score, mean_absolute_percentage_error
)
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, log_loss, cohen_kappa_score,
    matthews_corrcoef, hamming_loss, jaccard_score, top_k_accuracy_score
)
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)
from sklearn.utils.multiclass import type_of_target
import numpy as np
import copy


class ExperimentResult:
    # Stores the results of an experiment
    #   The results can be viewed by the mean of the iterations for each configuration
    #       or by the individual iteration results
    #   The global best configuration can also be retrieved
    #   The metrics collected may vary according to the solver and the problem:
    #       - 'mse': Mean Squared Error (for regression problems)
    #       - 'rmse': Root Mean Squared Error (for regression problems)
    #       - 'mae': Mean Absolute Error (for regression problems)
    #       - 'r2': R^2 score (for regression problems)
    #       - 'accuracy': Accuracy score (for classification problems)
    #       - 'f1': F1 score (for classification problems)
    #       - 'precision': Precision score (for classification problems)
    #       - 'recall': Recall score (for classification problems)
    #       - 'specificity': Specificity score (for classification problems)
    #       - 'sensitivity': Sensitivity score (for classification problems)
    #       - 'fpr': False Positive Rate (for classification problems)
    #       - 'fnr': False Negative Rate (for classification problems)
    #       - 'tpr': True Positive Rate (for classification problems)
    #       - 'tnr': True Negative Rate (for classification problems)
    #       - 'confusion_matrix': Confusion Matrix (for classification problems)
    #       - 'classification_report': Classification Report (for classification problems)
    #       - 'roc_curve': ROC Curve (for classification problems)
    #       - 'pr_curve': Precision-Recall Curve (for classification problems)
    #       - 'log_loss': Log Loss (for classification problems)
    #       - 'explained_variance': Explained Variance (for regression problems)
    #       - 'adjusted_r2': Adjusted R^2 score (for regression problems)
    #       - 'auc-roc': Area Under the ROC Curve (for classification problems)
    #       - 'mbd': Mean Bias Deviation (for regression problems)
    #       - 'cohens_kappa': Cohen's Kappa (for classification problems)
    #       - 'matthews_corrcoef': Matthews Correlation Coefficient (for classification problems)
    #       - 'hamming_loss': Hamming Loss (for multilabel classification problems)
    #       - 'jaccard_score': Jaccard Score (for classification problems)
    #       - 'top_k_accuracy': Top-K Accuracy (for classification problems)
    #       - 'gini': Gini Coefficient (for classification problems)
    #       - 'mape': Mean Absolute Percentage Error (for regression problems)
    #       - 'silhouette_score': Silhouette Score (for clustering problems)
    #       - 'calinski_harabasz_score': Calinski-Harabasz Score (for clustering problems)
    #       - 'davies_bouldin_score': Davies-Bouldin Score (for clustering problems)
    def __init__(self, problem_type="regression"):
        """
        Initializes the ExperimentResult object.

        Parameters:
            problem_type (str): Type of problem ('regression', 'classification').
        """
        self.problem_type = problem_type
        self.results_df = pd.DataFrame()
        self.configurations = {}
        self.current_empty_config_id = 0

    def start_configuration(self, config):
        """
        Starts a new configuration and returns a unique configuration ID.

        Parameters:
            config (dict): Configuration parameters.

        Returns:
            int: Unique configuration ID.
        """
        config_id = self.current_empty_config_id
        self.current_empty_config_id += 1
        self.configurations[config_id] = copy.deepcopy(config)
        return config_id
    
    def add_iteration(self, config_id, solver, prediction, y_true, y_scores=None, X=None):
        """
        Adds the results of an iteration to the results DataFrame.

        Parameters:
            config_id (int): Configuration ID.
            solver: Solver instance after training.
            prediction: Predictions made by the solver.
            y_true: True target values.
        """

        # Calculate metrics
        solver_metrics = {}
        if hasattr(solver, 'get_metrics'):
            solver_metrics = solver.get_metrics()

        metrics = self._compute_metrics(y_true, prediction, y_scores=y_scores, X=X)

        # Build the result entry
        config_params = self.configurations[config_id]
        iteration_num = len(self.results_df[self.results_df['config_id'] == config_id]) + 1

        result_entry = {
            'config_id': config_id,
            'iteration': iteration_num,
            'solver': solver.__class__.__name__,
            'prediction': prediction,
            'y_true': y_true,
            **solver_metrics,
            **metrics,
            **config_params
        }

        # Append the result entry to the results DataFrame
        self.results_df = pd.concat([self.results_df, pd.DataFrame([result_entry])], ignore_index=True)

    def end_configuration(self, config_id):
        """
        Ends a configuration by computing mean metrics over all iterations for the configuration
        and adding a new entry to the results DataFrame with iteration = -1.

        Parameters:
            config_id (int): Configuration ID.
        """

        config_iterations = self.results_df[
            (self.results_df['config_id'] == config_id) & (self.results_df['iteration'] != -1)
        ]

        if config_iterations.empty:
            #print(f"No iterations found for configuration {config_id}") TODO: LOG
            return
        
        non_metric_columns = ['config_id', 'iteration'] + list(self.configurations[config_id].keys())
        metric_columns = [col for col in config_iterations.columns if col not in non_metric_columns]

        mean_metrics = config_iterations[metric_columns].mean()
        best = config_iterations.loc[config_iterations['mse'].idxmin()]

        result_entry = {
            'config_id': config_id,
            'iteration': -1,
            # same solver name as the last iteration
            'solver': config_iterations['solver'].iloc[-1],
            # prediction of the one with least mse
            'prediction': best['prediction'],
            'y_true': best['y_true'],
            **mean_metrics,
            **self.configurations[config_id]
        }

        self.results_df = pd.concat([self.results_df, pd.DataFrame([result_entry])], ignore_index=True)

    def _compute_default_metrics(self, y_true, y_pred, y_scores=None, X=None):
        """
        Computes default metrics based on the problem type.

        Parameters:
            y_true: True target values.
            y_pred: Predicted values.
            y_scores: Predicted scores or probabilities (for some metrics like roc_auc).
            X: Input features (for clustering metrics that require data).

        Returns:
            dict: Computed metrics.
        """
        metrics = {}
        
        if self.problem_type == 'regression':
            # Regression metrics
            metrics = {
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': root_mean_squared_error(y_true, y_pred),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred),
                'explained_variance': explained_variance_score(y_true, y_pred),
                'mape': mean_absolute_percentage_error(y_true, y_pred),
                'mbd': np.mean(y_pred - y_true),
            }
            # Compute adjusted R² if possible
            if hasattr(self, 'n_features') and self.n_features is not None:
                n = len(y_true)
                p = self.n_features
                r2 = metrics['r2']
                adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
                metrics['adjusted_r2'] = adjusted_r2

        elif self.problem_type == 'classification':
            # Determine classification type
            y_type = type_of_target(y_true)
            n_classes = len(np.unique(y_true))
            average_method = 'binary' if n_classes == 2 else 'weighted'

            # Common metrics
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['f1'] = f1_score(y_true, y_pred, average=average_method)
            metrics['precision'] = precision_score(y_true, y_pred, average=average_method)
            metrics['recall'] = recall_score(y_true, y_pred, average=average_method)
            metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
            metrics['classification_report'] = classification_report(y_true, y_pred, output_dict=True)
            
            # Binary classification specific metrics
            if n_classes == 2 and y_type == 'binary':
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
                metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
                metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
                metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0
                metrics['tpr'] = metrics['sensitivity']
                metrics['tnr'] = metrics['specificity']
                metrics['cohens_kappa'] = cohen_kappa_score(y_true, y_pred)
                metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
                if y_scores is not None:
                    metrics['log_loss'] = log_loss(y_true, y_scores)
                    metrics['auc-roc'] = roc_auc_score(y_true, y_scores)
                    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
                    metrics['roc_curve'] = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
                    precision_vals, recall_vals, thresholds_pr = precision_recall_curve(y_true, y_scores)
                    metrics['pr_curve'] = {'precision': precision_vals, 'recall': recall_vals, 'thresholds': thresholds_pr}
                    # Gini Coefficient: Gini = 2 * AUC - 1
                    metrics['gini'] = 2 * metrics['auc-roc'] - 1
            else:
                # Multiclass or Multilabel metrics
                if y_type in ['multilabel-indicator', 'multiclass-multioutput']:
                    # Multilabel specific metrics
                    metrics['hamming_loss'] = hamming_loss(y_true, y_pred)
                    metrics['jaccard_score'] = jaccard_score(y_true, y_pred, average='samples')
                else:
                    # Multiclass metrics
                    metrics['cohens_kappa'] = cohen_kappa_score(y_true, y_pred)
                    metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
                    if y_scores is not None:
                        metrics['log_loss'] = log_loss(y_true, y_scores)
                        try:
                            metrics['auc-roc'] = roc_auc_score(y_true, y_scores, multi_class='ovr', average='weighted')
                        except ValueError:
                            pass
                    # Top-K Accuracy (assuming k=5)
                    if y_scores is not None:
                        metrics['top_k_accuracy'] = top_k_accuracy_score(y_true, y_scores, k=5, labels=np.unique(y_true))
                # Cross-entropy (same as log_loss)
                if y_scores is not None:
                    metrics['cross_entropy'] = log_loss(y_true, y_scores)
        
        elif self.problem_type == 'clustering':
            # Clustering metrics
            if X is None or y_pred is None:
                raise ValueError("X and y_pred are required for clustering metrics.")
            metrics = {
                'silhouette_score': silhouette_score(X, y_pred),
                'calinski_harabasz_score': calinski_harabasz_score(X, y_pred),
                'davies_bouldin_score': davies_bouldin_score(X, y_pred)
            }
        else:
            raise ValueError(f"Unsupported problem type: {self.problem_type}")
        return metrics
