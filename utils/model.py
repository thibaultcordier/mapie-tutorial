import numpy as np


class MapieFromTransformers():
    """
    A wrapper to use a transformers model with mapie.
    
    Parameters
    ----------
    estimator: transformers.pipelines.text_classification.TextClassificationPipeline
        The transformers model to wrap.

    lab2idx: dict
        A dictionary that maps the labels to their indices.
    """

    def __init__(self, estimator, lab2idx, **kwargs) -> None:
        self.estimator = estimator
        self.lab2idx_ = lab2idx
        self.classes_ = list(self.lab2idx_.values())
    
    def fit(self, X, y):
        """
        Do nothing in a prefit setting.
        """
        return self

    def predict_proba(self, X, **kwargs):
        """
        Return the prediction of the estimator (probabilities).
        
        Parameters
        ----------
        X: list
            The input data (row = text sample).

        Returns
        -------
        y_proba: np.ndarray
            The prediction of the estimator (probabilities).
        """
        # Adapt the shape of X with respect to the estimator API
        new_X = [x[0] for x in X]
    
        # Call the predict method of the estimator
        y_pred_raw = self.estimator(new_X, top_k=None)

        # Adapt the shape of y_pred_raw with respect to the mapie API
        key_fct = lambda elt: self.lab2idx_[elt['label']]
        sort_fct = lambda x: [elt['score'] for elt in sorted(x, key=key_fct)]
        y_proba = np.array(list(map(sort_fct, y_pred_raw)))

        # Return the prediction
        return y_proba

    def predict(self, X):
        """
        Return the prediction of the estimator (argmax).
        
        Parameters
        ----------
        X: list
            The input data (row = text sample).

        Returns
        -------
        y_pred: np.ndarray
            The prediction of the estimator (argmax).
        """
        y_pred = np.argmax(self.predict_proba(X), axis=1)
        return y_pred
