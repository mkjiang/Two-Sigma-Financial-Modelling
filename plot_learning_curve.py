"""
========================
Plotting Learning Curves
========================

"""
#print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
#from sklearn import cross_validation
#from sklearn.naive_bayes import GaussianNB
#from sklearn.svm import SVC
#from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.tree import export_graphviz
import subprocess


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, scoring='f1', train_sizes=np.linspace(.1, 1.0, 5), verbose=10):
    """
    Generate a simple plot of the test and traning learning curve.

    Determines cross-validated training and test scores for different training 
    set sizes.
    
    A cross-validation generator splits the whole dataset k times in training 
    and test data. Subsets of the training set with varying sizes will be used 
    to train the estimator and a score for each training subset size and the 
    test set will be computed. Afterwards, the scores will be averaged over all
    k runs for each training subset size.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    train_sizes: array-like, shape (n_ticks,), dtype float or int
    Relative or absolute numbers of training examples that will be used to
    generate the learning curve. If the dtype is float, it is regarded as 
    a fraction of the maximum size of the training set (that is determined
    by the selected validation method), i.e. it has to be within (0, 1]. 
    Otherwise it is interpreted as absolute sizes of the training sets. 
    Note that for classification the number of samples usually have to be 
    big enough to contain at least one sample from each class. (default: 
    np.linspace(0.1, 1.0, 5))
        
    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=scoring, verbose=verbose)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.grid(True)
    return plt

#
#digits = load_digits()
#X, y = digits.data, digits.target


#title = "Learning Curves (Naive Bayes)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
#cv = cross_validation.ShuffleSplit(digits.data.shape[0], n_iter=100,
#                                   test_size=0.2, random_state=0)

#estimator = GaussianNB()
#plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

#title = "Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
# SVC is more expensive so we do a lower number of CV iterations:
#cv = cross_validation.ShuffleSplit(digits.data.shape[0], n_iter=10,
#                                   test_size=0.2, random_state=0)
#estimator = SVC(gamma=0.001)
#plot_learning_curve(estimator, title, X, y, (0.7, 1.01), cv=cv, n_jobs=4)

#plt.show()