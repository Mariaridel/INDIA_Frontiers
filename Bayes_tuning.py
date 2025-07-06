#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 08:01:46 2024

@author: mariaridel
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedShuffleSplit

import matplotlib.pyplot as plt


def analyze_class_priors_with_reports(
    X_train, y_train, X_test, y_test, positive_label, priors_range
):
    """
    Analyzes the effect of changing class priors in GaussianNB and plots confusion matrix components,
    recall, and precision for both classes.

    Parameters:
    X_train, y_train : Training data and labels.
    X_test, y_test   : Test data and labels.
    positive_label   : The label of the positive class (e.g., "ISQUEMICO").
    priors_range     : List of tuples representing priors (e.g., [(0.05, 0.95), (0.1, 0.9), ...]).

    Returns:
    results          : A dictionary containing confusion matrix components and classification reports.
    """
    results = {}

    # Initialize lists for precision and recall tracking
    precision_class0 = []
    precision_class1 = []
    recall_class0 = []
    recall_class1 = []

    for priors in priors_range:
        # Initialize and fit the GaussianNB classifier with the current priors
        clf = GaussianNB(priors=priors)
        clf.fit(X_train, y_train)

        # Predict and compute confusion matrix
        pred = clf.predict(X_test)
        conf_matrix = confusion_matrix(y_test, pred)
        TN, FP, FN, TP = conf_matrix.ravel()



        # Generate classification report
        report = pd.DataFrame(
            classification_report(y_test, pred, output_dict=True)
        ).transpose()
        
        # Store confusion matrix components
        results[priors] = {
            "confusion_matrix": conf_matrix,
            "TN": TN,
            "FP": FP,
            "FN": FN,
            "TP": TP,
            "precision_H": report.loc["HEMORRAGICO", "precision"],
            "precision_I": report.loc["ISQUEMICO", "precision"],
            "recall_H": report.loc["HEMORRAGICO", "recall"],
            "recall_I": report.loc["ISQUEMICO", "recall"]
        }
        
        # Access metrics using class labels
        precision_class0.append(report.loc["HEMORRAGICO", "precision"])
        precision_class1.append(report.loc["ISQUEMICO", "precision"])
        recall_class0.append(report.loc["HEMORRAGICO", "recall"])
        recall_class1.append(report.loc["ISQUEMICO", "recall"])

    # Plot confusion matrix components
    plt.figure(figsize=(12, 8))
    for component in ["TN", "FP", "FN", "TP"]:
        plt.plot(
            [prior[1] for prior in priors_range],
            [results[prior][component] for prior in priors_range],
            label=component,
        )
    plt.xlabel("Class Prior for Positive Class")
    plt.ylabel("Count")
    plt.title("Confusion Matrix Components vs. Class Prior")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot precision and recall for both classes
    plt.figure(figsize=(12, 8))
    plt.plot([prior[1] for prior in priors_range], precision_class0, label="Precision HEMORRAGICO")
    plt.plot([prior[1] for prior in priors_range], recall_class0, label="Recall HEMORRAGICO")
    plt.plot([prior[1] for prior in priors_range], precision_class1, label="Precision ISQUEMICO")
    plt.plot([prior[1] for prior in priors_range], recall_class1, label="Recall ISQUEMICO")
    plt.xlabel("Class Prior for Positive Class")
    plt.ylabel("Score")
    plt.title("Precision and Recall vs. Class Prior")
    plt.legend()
    plt.grid()
    plt.show()

    return results


# Example usage:
priors_range = [(i, 1 - i) for i in np.arange(0.05, 1.0, 0.01)]
results_conf = analyze_class_priors_with_reports(
    X_train, y_train, X_test, y_test, positive_label="ISQUEMICO", priors_range=priors_range
)

