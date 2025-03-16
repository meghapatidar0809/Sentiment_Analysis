import numpy as np
from scipy import sparse as sp



class MultinomialNaiveBayes:
    """
    A Multinomial Naive Bayes model
    """ 
    
    def __init__(self, alpha=0.01) -> None:
        """
        Initialize the model
        :param alpha: float
            The Laplace smoothing factor (used to handle 0 probs)
            Hint: add this factor to the numerator and denominator
        """
        self.alpha = alpha
        self.priors = None
        self.means = None
        self.i = 0                   # to keep track of the number of examples seen
        self.feature_count = None    # Store raw feature counts
        self.class_count = None      # Store class-wise counts
        self.total_features = None   # Total number of features




    def fit(self, X: sp.csr_matrix, y: np.ndarray, update=False) -> None:
        """
        Fit the model on the training data
        :param X: sp.csr_matrix
            The training data
        :param y: np.ndarray
            The training labels
        :param update: bool
            Whether to the model is being updated with new data
            or trained from scratch
        :return: None
        """
        
        # Get unique class labels and their counts
        classes = np.unique(y, return_counts=False)

        if not update:
            # Initialize model parameters from scratch, if not updating
            self.total_features = X.shape[1]                                    # Total number of features in the dataset
            self.class_count = np.zeros(len(classes))                           # Array to store counts for each class
            self.feature_count = np.zeros((len(classes), self.total_features))  # Feature occurrence per class
        else:
            # Ensure the number of classes matches
            assert len(classes) == len(self.class_count), "Number of classes must match during update."

        # Iterate over each class and update class counts and feature counts
        for i, c in enumerate(classes):
            class_mask = (y == c)                               # Boolean mask for selecting samples of the current class
            class_count = np.sum(class_mask)                    # Total occurrences of the current class in y
            feature_sum = np.ravel(X[class_mask].sum(axis=0))

            if update:
                # Update class counts and feature counts incrementally
                self.class_count[i] += class_count
                self.feature_count[i, :] += feature_sum

                # Update priors incrementally
                total_samples = np.sum(self.class_count)
                self.priors[i] = np.log(self.class_count[i] / total_samples)

                # Update sum_means incrementally
                self.means[i, :] = (self.feature_count[i, :] + self.alpha) / \
                                    (self.feature_count[i, :].sum() + self.alpha * self.total_features)
                self.sum_means[i, :] = np.log(self.means[i, :])
            else:
                # Initialize class and feature counts from scratch
                self.class_count[i] = class_count
                self.feature_count[i, :] = feature_sum

        if not update:
            # Compute log priors and log probabilities only once if training from scratch
            self.priors = np.log(self.class_count / np.sum(self.class_count))
            
            # Compute and store smoothed conditional probabilities (Laplace smoothing applied)
            self.means = (self.feature_count + self.alpha) / \
                         (self.feature_count.sum(axis=1, keepdims=True) + self.alpha * self.total_features)
            self.sum_means = np.log(self.means)





    def predict(self, X: sp.csr_matrix) -> np.ndarray:
        """
        Predict the labels for the input data
        :param X: sp.csr_matrix
            The input data
        :return: np.ndarray
            The predicted labels
        """
        assert self.priors.shape[0] == self.means.shape[0], "Mismatch between priors and means dimensions."
    
        '''preds = []
        for i in range(X.shape[0]):
            log_likelihoods = X[i] @ self.sum_means.T + self.priors  # Compute log probabilities for the i-th sample
            preds.append(np.argmax(log_likelihoods))                 # Append the class with the highest log probability
        return np.array(preds)'''
        log_probs = X @ self.sum_means.T + self.priors
        return np.argmax(log_probs, axis=1)



    
    def predict_proba(self, X: sp.csr_matrix) -> np.ndarray:
        """
        Compute probability estimates for each class, by exponentiating log probabilities and normalizing them.
        This helps in uncertainty estimation and decision-making in multi-class classification.
        :param X: sp.csr_matrix
            The input data
        :return: np.ndarray
            Probability estimates for each class
        """
        
        log_probs = X @ self.sum_means.T + self.priors                    # Compute log probabilities
        log_probs -= log_probs.max(axis=1, keepdims=True)                 # Normalize
        probs = np.exp(log_probs)                                       
        return probs / probs.sum(axis=1, keepdims=True)                   # Convert to probabilities