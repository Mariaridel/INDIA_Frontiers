#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 15:33:03 2024

@author: mariaridel
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import os
import warnings
import seaborn as sns
from itertools import chain
import pickle

import openpyxl
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.utils import get_column_letter
from openpyxl.styles import NamedStyle
from openpyxl.worksheet.views import SheetView, Selection
from openpyxl.styles import Border, Side, Alignment, PatternFill, Font

from scipy.stats import chi2_contingency, fisher_exact, ttest_ind, mannwhitneyu, shapiro, spearmanr
import statsmodels.formula.api as smf


from sklearn import linear_model
from sklearn.svm import l1_min_c
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.utils import resample,shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_curve

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier     
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTENC, SMOTE, BorderlineSMOTE
import shap

import pywinEA2
from pywinEA2.report import *


from Logger import Logger
from time_logging_decorator import log_execution_time

class FeatureAnalysis():
    '''
    
    Class to develope model and feature selection
    
    '''
    
    def __init__(self,  data: pd.DataFrame(), 
                        name: list, 
                        columns_analyze: list, 
                        columns_to_drop: list = [], 
                        ID: str = "CIPA", 
                        label: str = 'TIPO_ICTUS', 
                        cweight: str = None, 
                        flag_sampling: int = 0, 
                        flag_ischemic: bool = False,
                        drop_MD: bool = True, 
                        drop_MD_vars: bool = False, 
                        seed: int = 1234, 
                        metric: str = "roc_auc_ovo",
                        flag_genetic: bool = False,
                        change_positive_class: bool = False):
        '''
        

        Parameters
        ----------
        data : pd.DataFrame()
            Dataframe to run experiement. Should include:
                - CIPA
                - TIPO_ICTUS
                - OGV
                - EDAD
                - ICTUS_PUNTUACION
        name : str
            Name of the modelo to be developed. Used to save the results.
        columns_analyze : list
            List with column names to be analyzed.
        columns_to_drop : list, optional
            List with column names to be dropped. Usefull to reuse preveious parameter. 
            The default is [].
        ID : str, optional
            ID of the dataframe. The default is "CIPA".
        label : str, optional
            Label to be analyzed. It should be:
                - TIPO_ICTUS
                - OGV
            The default is 'TIPO_ICTUS'.
        cweight : str, optional
            Parameter to implement class_weights to train model with sklearn.
            Can be dict, “balanced” or None
            If “balanced”, class weights will be given by n_samples / (n_classes * np.bincount(y)). 
            If a dictionary is given, keys are classes and values are corresponding class weights. 
            If None is given, the class weights will be uniform.. The default is None.
        flag_sampling : int, optional
            Flag to define sampleing strategy:
                - 0: no sampling strategy implemented
                - 1: upsampling implemented
                - 2: SMOTEC implemented
                - 3: SMOTE Borderline implemented
            The default is 0.
        flag_ischemic : bool, optional
            Flag to use only ischemic episodes to develop models. Usefull when
            OGV is the label to be predicted.
            The default is False.
        drop_MD : bool, optional
            Flag to run same experiment but including MD scale. 
            To include it, set it to False. 
            The default is True.
        drop_MD_vars : bool, optional
            Flag to run same experiment but removing MD variables. The default is False.
        seed : int, optional
            Seed to make experiments reproducible. The default is 1234.
        metric : str, optional
            Metric to be optimized during model training. The default is "roc_auc_ovo".
        flag_genetic : bool, optional
            Flag to run genetic experiments. The default is False.
        change_positive_class: bool, optimal
            Flag to change the default positive class to be optimized. 
            The default is false

        Returns
        -------
        None.

        '''
        self.logger = Logger('model_logger').logger
        
        self.data_original = data
        self.name_general, self.name = name
        self.columns_analyze = columns_analyze
        self.columns_to_drop = columns_to_drop
        self.ID = ID
        self.label = label
        self.cweight = cweight

        self.flag_sampling = flag_sampling
        self.flag_ischemic = flag_ischemic
        self.drop_MD = drop_MD
        self.drop_MD_vars = drop_MD_vars
        self.seed = seed
        self.metric = metric
        
        self.flag_genetic = flag_genetic
        self.flag_plot = flag_plot
        self.flag_test = flag_test
        self.change_positive_class = change_positive_class
        
        self.general_path = 'Results/' + self.name_general
        self.excel_path = self.general_path + '/analysis_gen_' + self.name_general + '.xlsx'
        self.plot_path = self.general_path + '/Plot/' + self.name
        self.test_path = self.general_path + '/StatisticalTest/' + self.name
        self.model_path = self.general_path + '/Model/' + self.name
        self.genetic_path = self.general_path + '/Genetic/' + self.name
        self.MD_path = self.general_path + '/MD'
        
        self.test_size = 0.3
        
        # Define characteristics for each label
        if self.label == "TIPO_ICTUS":
            self.label_aux = "OGV"
            self.row_summary = 3
            if self.change_positive_class == False:
                self.positive_label = "ISQUEMICO"
                self.negative_label = "HEMORRAGICO"
                # Mapping for SMOTE
                self.cat_mapping={'ISQUEMICO':1,'HEMORRAGICO':2}
                # Mappint for genetic algorithm
                self.cat_mapping_genetic={'ISQUEMICO':1,'HEMORRAGICO':0}
            elif self.change_positive_class == True:
                self.positive_label = "HEMORRAGICO"
                self.negative_label = "ISQUEMICO"
                # Mapping for SMOTE
                self.cat_mapping={'HEMORRAGICO':1,'ISQUEMICO':2}
                # Mappint for genetic algorithm
                self.cat_mapping_genetic={'HEMORRAGICO':1,'ISQUEMICO':0}

            
        elif self.label == "OGV":
            self.positive_label = "SI"
            self.negative_label = "NO"
            # Mapping for SMOTE
            self.cat_mapping = {'SI':1,'NO':2}
            # Mappint for genetic algorithm
            self.cat_mapping_genetic={'SI':1,'NO':0}
            self.label_aux = "TIPO_ICTUS"
            self.row_summary = 23
            
        self.MD_var = "ICTUS_PUNTUACION"
        self.age = "EDAD"
        
        if self.drop_MD_vars == True:
            MD_vars = [col for col in self.data_original if col.startswith('ICTUS_')]
            self.MD_vars = MD_vars + [self.age] + ["CTES_TAS"]
        else: 
            self.MD_vars = []

        # Create scorer depending on metric
        if self.metric == "f1":
            self.scorer = make_scorer(f1_score, pos_label=self.positive_label)
        elif metric == "recall":
            self.scorer =  make_scorer(recall_score, pos_label=self.positive_label)
        elif metric == "precision":
            self.scorer =  make_scorer(precision_score, pos_label=self.positive_label)
        else:
            self.scorer = metric
        
        # Genetic algorithm only accepts precision, recall, accuracy, auc and f1
        # for classification
        if self.metric in ["precision", "recall", "accuracy", "f1"]:
            self.genetic_metric = self.metric
        else:
            self.genetic_metric = "auc"
        
        # Hyperparameter tuning 
        self.CV = 3
        self.tuning = "Random",
        self.random_iter = 100 #200
        self.random_iter_gen = 10 #20
        self.population_size = 100
        self.max_generations = 100
        
        #ROC CURVE TOTAL
        self.cmap = get_cmap("plasma")
        self.combined_figure = plt.figure(figsize=(25,15))
        self.combined_ax = self.combined_figure.add_subplot(1, 1, 1)
        self.combined_ax.set_title("ROC curves", fontsize=16)
        self.call_count = 0
    
    
            
    def is_ordinal(self, data, variable):
        '''
        Function to define variables that could be ordinal, to implement specific 
        statistic test.

        Parameters
        ----------
        data : pd.DataFrame()
            DESCRIPTION.
        variable : str
            name of variable to be studied.

        Returns
        -------
        bool
            True if variable is ordinal.

        '''
        
        if pd.api.types.is_integer_dtype(data[variable]) or pd.api.types.is_categorical_dtype(data[variable]):
            unique_vals = data[variable].nunique()
            if unique_vals < 10:  # Arbitrary threshold: less than 10 unique values likely indicates ordinal
                return True
        return False
            
    
    @log_execution_time
    def drop_variables(self, data: pd.DataFrame()):
        '''   
        Function to drop variables
        
        Parameters
        ----------
        data : pd.Dataframe()
            Dataframe to remove variables from.

        Returns
        -------
        data : pd.Dataframe()
            Dataframe with variables removed.

        '''
        
        try:
            data = data.drop(self.columns_to_drop, axis = 1)
            return data
        
        except Exception as e:
            self.logger.error(f"Error in drop_variables: {e}")
            return pd.Dataframe()

    
    @log_execution_time 
    def data_preprocessing(self):
        '''
        Data preprocessing:
            - Keep only relevant variables.
            - Remove constant variables.
            - Substitute infinite values by NaN.
            - Remove columns with all NaN values.
            - Remove variables with all information NaN. Threshold defined at 7
            as it is the number of variables of the label dataframe.
            - Remove patient with any missing.

        Returns
        -------
        None.

        '''
        
        try:
            numeric_columns = ["EDAD",
                               "CTES_TAS",                     
                               "CTES_TAD",                
                               "CTES_FC",                       
                               "CTES_FR",                       
                               "CTES_SATO2",                    
                               "CTES_GLUCEMIA",                
                               "CTES_TEMPERATURA",
                               "ICTUS_PUNTUACION"]
    
            self.data_original[numeric_columns] = self.data_original[numeric_columns].apply(pd.to_numeric, errors='coerce')
    
            # Select initial variables
            data = self.data_original[self.columns_analyze].copy()
            
            # Remove features to simplify class calling
            data =  self.drop_variables(data)
            
            #Remove constant variables
            data = data.loc[:, (data != data.iloc[0]).any()]
            
            #Remove infinite values
            data.replace([np.inf, -np.inf], np.nan, inplace = True)
            
            #Remove variables with all NaN valus
            data = data.dropna(axis=1, how='all')
            
            # Remove patients with all delfos info missing. 
            # CIPA, FECHA, TIPO_ICTUS, OGV, TROMBECTOMIA, TROMBOLISIS
            data.dropna(thresh=7, inplace = True)    
            
            # Remove columns with missing values
            # print("Dataset with all columns informed")
            # data = data.dropna(axis = 1, how = "any" )
    
            # Remove patients with missing values except in second label (for statistics)
            print("Dataset with all rows informed")
            subset_columns = [col for col in data.columns if col != self.label_aux]
            self.data = data.dropna(axis = 0, how = "any", subset=subset_columns)
        
        except Exception as e:
            self.logger.error(f"Error in data_preprocessing: {e}")
            return pd.DataFrame()

    

    @log_execution_time 
    def calculate_MD_mod(self):
        '''
        Obtain Madrid Direct performance in test sample with two strategies:
            - Extracting ICTUS_PUNTUACION and comparing the result to OGV
            - Extracting ICTUS_PUNTUACION, remove age factor to compare it more 
            easily to the OGV label. 
            
        For this two strategies, confussion matrix and sklearn report are calculated.

        '''
        try:

            data = pd.concat([self.data.copy(deep=True), 
                              self.CIPA.copy(deep=True), 
                              self.MD_df.copy(deep=True) ], axis = 1)
            
            if self.MD_var not in self.data.columns:
                data = pd.concat([data, self.MD.copy(deep=True)], axis=1)
                
            MD = pd.DataFrame()
            
            if self.label == "OGV":
                MD = data[[self.ID, self.label, self.MD_var, self.age]]
                MD.loc[:,"OGV_MD"] = MD[self.MD_var].map(lambda x: 'SI' if 1 < x else 'NO')
                MD.loc[:,"EDAD_resta"] = MD["EDAD"]-85
                MD.loc[:,'EDAD_resta'] = MD['EDAD_resta'].where(MD['EDAD_resta'] >= 0, 0)
                MD.loc[:,"ICTUS_PUNTUACION_mod"]= MD[[self.MD_var, 'EDAD_resta']].sum(axis=1)
                MD.loc[:,"OGV_MD_mod"] = MD['ICTUS_PUNTUACION_mod'].map(lambda x: 'SI' if 1 < x else 'NO')
            
            
            if not MD.empty:
                
                if not os.path.isdir(self.MD_path):
                    os.makedirs(self.MD_path, exist_ok=True)
                
                MD_train, self.MD_test = MD.iloc[self.train_index, :], MD.iloc[self.test_index, :] 
                
                conf_matrix_MD = pd.DataFrame(confusion_matrix(self.MD_test[self.label], self.MD_test["OGV_MD"]))
                report_MD = pd.DataFrame(classification_report(self.MD_test[self.label], self.MD_test["OGV_MD"], output_dict=True)).transpose()[0:2].T
                
                
                conf_matrix_MD_mod = pd.DataFrame(confusion_matrix(self.MD_test[self.label], self.MD_test["OGV_MD_mod"]))
                report_MD_mod = pd.DataFrame(classification_report(self.MD_test[self.label], self.MD_test["OGV_MD_mod"], output_dict=True)).transpose()[0:2].T
                  
                
                conf_matrix_MD_total = pd.DataFrame(confusion_matrix(MD[self.label], MD["OGV_MD_mod"]))
                report_MD_total = pd.DataFrame(classification_report(MD[self.label], MD["OGV_MD_mod"], output_dict=True)).transpose()[0:2].T
                
                return conf_matrix_MD, conf_matrix_MD_mod, report_MD, report_MD_mod, conf_matrix_MD_total, report_MD_total
            
            else:
                return None, None, None, None, None, None
        
        except Exception as e:
            self.logger.error(f"Error in calculate_MD_mod: {e}")
            return None, None, None, None, None, None
    

    
    @log_execution_time
    def prepare_data(self):
        '''
        Function to prepare data for model training:
            - OTROS is removed from sample
            - if flag_ischemic is True, keep only ischemic population
            - if drop_MD is True, remove Madrid Direct scale. Keep information 
            for MD calculation.
            - Remove all MD variables. If drop_MD_vars is false, this list is empty,
            and all variables remain the same. Otherwhie, all variables
            related to MD are dropped.
            - Remove CIPA.
            - Split data in training data (X) and label (y)

        '''
        
        try:
            # Remove OTROS
            self.data = self.data[self.data['TIPO_ICTUS'] != "OTROS"]
    
            # Remove strokes that are not ischemic
            if self.label == 'OGV' and self.flag_ischemic == True:
                self.data=self.data[self.data['TIPO_ICTUS']=="ISQUEMICO"]
            
            # Remove MD scale
            if self.MD_var in self.data.columns: 
                self.MD = self.data[self.MD_var]
                
                if self.drop_MD == True:
                    self.data = self.data.drop(columns=[self.MD_var])
            else:
                self.MD = pd.DataFrame()
            
            # Remove all MD variables
            self.MD_df = self.data[[var for var in self.MD_vars if var in self.data.columns]]
            self.data = self.data.drop(columns=[var for var in self.MD_vars if var in self.data.columns])
                        
            # Remove other label
            self.data = self.data.drop(columns=[self.label_aux])
                        
            # Eliminamos ID
            self.CIPA = self.data[self.ID]
            self.data = self.data.drop(columns=[self.ID])
            
            self.ncols = len(self.data.columns)
            self.cols = (self.data.columns)
            
            
            # Split X,y
            self.y = self.data[self.label]
            self.X = self.data.drop(columns=self.label)
        
        except Exception as e:
            self.logger.error(f"Error in prepare_data: {e}")
        
    
    @log_execution_time
    def great_low_label(self):
        '''
        Calculate which is the label with more population and the lowest,
        to implement upsampling techniques.
        '''
        
        try:
            self.great_label = self.y_size.idxmax()
            self.low_label = self.y_size.idxmin()
            
        except Exception as e:
            self.logger.error(f"Error in great_low_label: {e}")


    @log_execution_time  
    def data_sampling(self):
        '''
        Function to obtain final train and test sample to train models.
        
        When flag_sampling is 1 or 2, upsampling or SMOTE are implemented.
        
        If flag_sampling is 3 it is not implemented in this class, as only numerical variables are accepted.
        Will apply only for genetic models


        '''
        
        try:
                
            split_data = StratifiedShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.seed)
            # split_data = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
    
            split_data.get_n_splits(self.X, self.y)
            self.train_index, self.test_index = next(split_data.split(self.X, self.y))
            
            self.X_train, self.X_test = self.X.iloc[self.train_index, :], self.X.iloc[self.test_index, :] 
            self.y_train, self.y_test = self.y.iloc[self.train_index], self.y.iloc[self.test_index]
            
            # UPSAMPLING
            if self.flag_sampling == 1:
                self.logger.info("Apply Upsampling")
                self.great_low_label()
                
                aux = pd.concat([self.X_train, self.y_train], axis=1)
                
                X_train_great=aux[aux[self.label]==self.great_label]
                X_train_low=aux[aux[self.label]==self.low_label]
                
                X_train_low_up = resample(X_train_low,
                                           random_state=self.seed,
                                           n_samples=len(X_train_great),
                                           replace=True)
                
                X_train = pd.concat([X_train_low_up,X_train_great])
                X_train = X_train.sample(frac = 1)
                
                self.y_train = X_train[self.label]
                self.X_train = X_train.drop(columns=self.label)
    
            # SMOTE
            elif self.flag_sampling == 2:
                self.logger.info("Apply SMOTE Categorical")
                
                y_train=self.y_train.map(self.cat_mapping)
                
                categorical_feature_mask = self.X.dtypes == object
                categorical_feature_mask = categorical_feature_mask.to_list()
                
                #apply smote
                oversample = SMOTENC(categorical_features = categorical_feature_mask,  random_state=self.seed)
                #oversample = SMOTE()
                self.X_train, y_train = oversample.fit_resample(self.X_train, y_train)
                self.y_train=y_train.map( {v: k for k, v in self.cat_mapping.items()})
                
        
        except Exception as e:
            self.logger.error(f"Error in data_sampling: {e}")
            
    
    @log_execution_time
    def model_preprocess(self):
        '''
        Function to define pre-processing for model development:
            - Numeric variables: scale
            - Categorical variables: one hot encodding
            
        Implemented in sklearn pipeline

        '''
        
        try:
            numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
            numeric_features = self.X.select_dtypes(include=np.number).columns
    
            numeric_transformer = Pipeline(
                steps=[("scaler", StandardScaler())]
            )
            
            categorical_features = self.X.select_dtypes(exclude=numerics).columns
            categorical_transformer = Pipeline(
                steps=[
                    ("encoder", OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
                ]
            )
            
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, numeric_features),
                    ("cat", categorical_transformer, categorical_features),
                ]
            )
        
        except Exception as e:
                self.logger.error(f"Error in model_preprocess: {e}")
    
    @log_execution_time
    def model_pipeline(self, model_name, clf_model, grid_model):
        '''
        Model development as a base line to genetic tuning.

        Parameters
        ----------
        model_name : str
            Name of the model to be developed.
        clf_model : sklearn model
            class of the model.
        grid_model : dict
            grid to define hyoerparameter tuning.

        Returns
        -------
        clf_random : TYPE
            DESCRIPTION.
        column_names_preprocesor : list
            list of features used by the model.
        model_info : dict
            dict containing relevant information for the model.

        '''
        
        try:
            filename = self.model_path + '/' + model_name +'.pkl'
            file_random = self.model_path + '/' + model_name +'_clf_random.pkl'
            file_preprocessor = self.model_path + '/' + model_name +'_preprocessor.pkl'
            
            color = self.cmap(self.call_count / 6)
            
            if not os.path.isfile(filename):
            
                self.logger.info("Model development")
                self.logger.info("Random Search CV")
                
                
                if not os.path.isdir(self.model_path):
                    os.makedirs(self.model_path, exist_ok=True)
                
                clf_random = RandomizedSearchCV(estimator = clf_model, 
                                                param_distributions = grid_model, 
                                                n_iter = self.random_iter, 
                                                cv = self.CV, 
                                                verbose = 2, 
                                                random_state = self.seed, 
                                                n_jobs = -1,
                                                scoring = self.scorer,
                                                return_train_score = True
                                                )
             
                clf = Pipeline(        
                    steps=[("preprocessor", self.preprocessor), ("classifier", clf_random)])
        
                clf.fit(self.X_train, self.y_train) 
                
                
                with open(filename, "wb") as file: 
                    pickle.dump(clf, file)
                
                with open(file_random, "wb") as file: 
                    pickle.dump(clf_random, file)
                    
                with open(file_preprocessor, "wb") as file: 
                    pickle.dump(self.preprocessor, file)
                    
            else:
                
                self.logger.info("Loading model")
                
                with open(filename, 'rb') as file:
                    clf = pickle.load(file)
                    
                with open(file_random, 'rb') as file:
                    clf_random = pickle.load(file)
                
                with open(file_preprocessor, 'rb') as file:
                    self.preprocessor = pickle.load(file)
                
                
            pred = clf.predict(self.X_test)
            proba = clf.predict_proba(self.X_test )[:,1]
            
            proba_train = clf.predict_proba(self.X_train)[:,1]
            
            
            basic = pd.DataFrame({
                        "Modelo" : model_name, 
                        "Optimization metric": self.metric,
                        "Params" : str(grid_model),
                        "Best Params": str(clf_random.best_params_),
                        "seed": self.seed}, index=[0]).T
            
    
            metrics = pd.DataFrame({
                            "Train Best metric": clf_random.best_score_,
                            "Train AUC": roc_auc_score(self.y_train, proba_train),
                            "Test AUC": roc_auc_score(self.y_test, proba),
                            "Test F1": f1_score(self.y_test, pred, pos_label=self.positive_label),
                            "Test Accuracy": accuracy_score(self.y_test, pred)}, index=[0]).T
    
    
            conf_matrix = pd.DataFrame(confusion_matrix(self.y_test, pred))
            
            report = pd.DataFrame(classification_report(self.y_test, pred, output_dict=True)).transpose()[0:2].T
            
            
            model_info = {"BASIC": basic,
                          "METRICS": metrics,
                          "CONF_MATRIX": conf_matrix,
                          "REPORT": report}
            
    
            column_names_preprocesor = self.preprocessor.get_feature_names_out()    
            
            
            individual_fig = plt.figure(figsize=(20, 12))
            fpr, tpr, _ = roc_curve(self.y_test,  proba, pos_label=self.positive_label)
            auc = roc_auc_score(self.y_test, proba)
            plt.plot(fpr,tpr,label="Test auc="+str(round(auc,2)), color = color, linewidth=2.5)
            plt.legend(loc=4)

            fpr_train, tpr_train, _ = roc_curve(self.y_train,  proba_train, pos_label=self.positive_label)
            auc_train = roc_auc_score(self.y_train, proba_train)
            plt.plot(fpr_train,tpr_train,label="Train auc="+str(round(auc_train,2)), color = "gainsboro", linewidth=2.5)
            plt.legend(loc=4, fontsize=14)
            plt.title(f"{model_name} ROC curve", fontsize=16)
            
            #plt.savefig(f"{self.model_path}/{model_name}_ROC.png", dpi=175, bbox_inches='tight')
            individual_fig.savefig(f"{self.model_path}/{model_name}_ROC.png", dpi=175, bbox_inches='tight')
            plt.close(individual_fig)
            
            self.combined_ax.plot(fpr,tpr, label=f"{model_name} AUC " + str(round(auc,2)), color=color, linewidth=2.5)
            
            # Increment the call count
            self.call_count += 1
            
            return clf_random, column_names_preprocesor, model_info

        
        except Exception as e:
            self.logger.error(f"Error in model_pipeline: {e}")
            return None, [], {}
                
    
    @log_execution_time
    def genetic_preprocess(self):
        '''
        Function to adapt preprocessing to genetic requirements. Only numerical variables are accepted

        Returns
        -------
        None.

        '''
        
        try:
            y_train_gen=self.y_train.map(self.cat_mapping_genetic)
            y_test_gen=self.y_test.map(self.cat_mapping_genetic)
            
            # pywinEA requires dataframe with numerical variables and label included
            # preprocessor is applied previously as it is not accepted in pywinEA
            
            X_train_gen = self.preprocessor.transform(self.X_train)
            X_test_gen = self.preprocessor.transform(self.X_test)
    
            column_names= self.preprocessor.get_feature_names_out()  
            
            self.data_train_gen = pd.concat([pd.DataFrame(X_train_gen, columns=column_names), 
                                  y_train_gen.reset_index(drop=True)], axis =1)
            
            self.data_test_gen = pd.concat([pd.DataFrame(X_test_gen, columns=column_names), 
                                  y_test_gen.reset_index(drop=True)], axis = 1)
            

            if self.flag_sampling == 3:
                
                self.logger.info("Apply SMOTE Borderline")
                oversample = BorderlineSMOTE(random_state=self.seed)
                
                X_train_gen = self.data_train_gen.drop(self.label, axis=1)
                X_train_gen, y_train_gen = oversample.fit_resample(X_train_gen, 
                                                                  self.data_train_gen[self.label])
                
                self.data_train_gen = pd.concat([X_train_gen, y_train_gen], axis =1)
            
            
        except Exception as e:
            self.logger.error(f"Error in genetic_preprocess: {e}")
    
    
    @log_execution_time
    def MOG_genetic_pipeline(self, model_name, clf_model, grid_model):
         
         try:
             self.logger.info("Genetic algorithm")
             
             # Create folder to save genetic results
             if not os.path.isdir(self.genetic_path):
                 os.makedirs(self.genetic_path, exist_ok=True)
                 
             # to look for files starting with model_name
             # filename_preffix = self.genetic_path + '/' + model_name    
             # files = os.listdir(filename_preffix)
             # prefixed_files = [f for f in files if f.startswith(model_name)]
             
             # look specifically for muga report
             filename_report = self.genetic_path + '/muga_report_' + model_name + '.pkl'
             
             target_features = self.data_train_gen.columns.tolist()
             target_features.remove(self.label)
             
             if not os.path.isfile(filename_report):
                 
                 self.logger.info("Algorithm development")
                 
                 if model_name in ["Decission Tree", "KNN", "Logistic Regression"]:
                     
                     self.logger.info("Genetic Grid search")
                     
                     # Define GridSearch as model for genetic algorithm
                     clf_grid = GridSearchCV(
                         clf_model,
                         param_grid = (grid_model),
                         refit=True,
                         n_jobs=-1,
                         cv = self.CV,
                         return_train_score = True
                         )
                 
                 elif model_name in ["Gradient Boosting", "Random Forest", "SVC"]:
                     
                     self.logger.info("Genetic Random search")
                     
                     clf_grid = RandomizedSearchCV(estimator = clf_model, 
                                                     param_distributions = grid_model, 
                                                     n_iter = self.random_iter_gen, 
                                                     cv = self.CV, 
                                                     verbose = 1, 
                                                     random_state = self.seed, 
                                                     refit=True,
                                                     n_jobs = -1,
                                                     # scoring = self.scorer,
                                                     return_train_score = True
                                                     )
            
                  
                 # MULTI OBJECTIVE GENETIC ALGORITHM definition
                 muga = pywinEA2.MultiObjFeatureSelectionNSGA2(
                                 # data-related parameters
                                 data=self.data_train_gen,
                                 model=clf_grid,
                                 score=self.genetic_metric,            # also 'f1' provided as string could work
                                 y=[self.label],                       # target variable (must be present in the data)
                                 population_size=self.population_size,
                                 max_generations=self.max_generations,
                                 optim='max',
                                 # ... subject all variables except the 'target' variable to the feture selection
                                 target_feats=target_features,  
                                 # cross-validation parameters used for evaluate the inner model
                                 cv=self.CV,
                                 cv_reps=1,
                                 stratified=True,
                             )
                 
             
                 with warnings.catch_warnings(record=True) as w:
                     warnings.simplefilter("always")
                     muga_report = pywinEA2.run(muga, type='nsga2', verbose=True)
                 
                
                 
                 with open(filename_report, "wb") as file: 
                     pickle.dump(muga_report, file)
                     
                 # Plots       
                 displayMultiObjectiveConvergence(
                        muga_report,
                        title='Convergence', 
                        objective_names=[self.genetic_metric, 'Features'],
                        figsize=(15, 12),
                        title_size=15,
                        legend_size=12,
                        save_plot = self.genetic_path + "/results_MUG_conv" + "_" + self.genetic_metric + "_" + model_name + ".png"
                    )   
                
                 displayParetoFront(
                    muga_report,
                    objective_names=['Features', self.genetic_metric],
                    figsize=(15, 12),
                    title_size=15,
                    save_plot = self.genetic_path + "/results_MUG_features" + "_" + self.genetic_metric + "_" + model_name  + ".png"
                 )
                        
                 # pareto_front = muga_report.pareto_front
                 # best_solution = np.array([ind.fitness.values for ind in pareto_front])
                 
             else:
                 self.logger.info("Algorithm loading")
                 
                 with open(filename_report, 'rb') as file:
                     muga_report = pickle.load(file)
                 
             model_gen_total = []
            
             model_number = 0
             for ind in muga_report.pareto_front:
                selected_features = np.array(target_features)[np.array(ind, dtype=bool)]
                
                gen_model = muga_report._algorithm._model.fit(self.data_train_gen[selected_features], 
                                                              self.data_train_gen[self.label])
                
                filename = self.genetic_path + '/' + model_name + str(model_number) + '.pkl'
                
                if not os.path.isfile(filename):
                    with open(filename, "wb") as file: 
                        pickle.dump(gen_model, file)
                    
                
                pred = gen_model.predict(self.data_test_gen[selected_features])
                
                proba = gen_model.predict_proba(self.data_test_gen[selected_features])[:,1]
                proba_train = gen_model.predict_proba(self.data_train_gen[selected_features])[:,1]
                
                model_number +=1
                
                basic = pd.DataFrame({
                            "Modelo" : model_name + " GENETIC NSGA2", 
                            "Otimization metric": self.genetic_metric,
                            #"Genetic Params": str(muga_report.algorithm.getParams()), #Quitar info
                            "Clasificator Params" : str(grid_model),
                            "Clasificator Best Params": str(gen_model.best_params_),
                            "Selected Features": str(selected_features),
                            "Number of features selected": len(selected_features),
                            "% of features selected": len(selected_features) / (self.data_train_gen.shape[1] - 1),
                            "seed": self.seed}, index=[0]).T
                

                metrics = pd.DataFrame({
                                "Train Best metric": gen_model.best_score_,
                                "Train AUC": roc_auc_score(self.data_train_gen[self.label], proba_train),
                                "Test AUC": roc_auc_score(self.data_test_gen[self.label], proba),
                                "Test F1": f1_score(self.data_test_gen[self.label], pred),
                                "Test Accuracy": accuracy_score(self.data_test_gen[self.label], pred)}, index=[0]).T


                conf_matrix = pd.DataFrame(confusion_matrix(self.data_test_gen[self.label], pred))
                
                report = pd.DataFrame(classification_report(self.data_test_gen[self.label], pred, output_dict=True)).transpose()[0:2].T
                
                model_gen = {"BASIC": basic,
                             "METRICS": metrics,
                             "CONF_MATRIX": conf_matrix,
                             "REPORT": report}
            
                try:
                    # TODO smth not working here!!!!
                    if model_name in ["Gradient Boosting", "Random Forest"]:
                        model_best = gen_model.best_estimator_
                        importances = pd.DataFrame(model_best.feature_importances_, index = selected_features)
                        model_gen["FEATURE_IMPORTANCE"] =  importances
                        
                    elif model_name in ["Logistic Regression"]:
                        model_best = gen_model.best_estimator_
                        model_coef = pd.DataFrame(model_best.coef_, columns = selected_features).T
                        model_gen["FEATURE_COEFFICIENT"] =  model_coef
                        
                except Exception as e:
                    self.logger.error(f"Error in Feature Coefficiente: {e}")
                    model_gen["FEATURE_COEFFICIENT"] =  []
                    
                
                model_gen_total.append(model_gen)
                model_number +=1
         
           
                return model_gen_total

         except Exception as e:
             self.logger.error(f"Error in MOG_genetic_pipeline: {e}")
             return []
            
    
    @log_execution_time
    def model_definition(self):
        '''
        Model pipeline for Decission Tree, Gradient Boosting, Random Forest, KNN, SVM and Logistic Regression
        '''
        
        try:
        
            self.model_info = []
            self.model_info_genetic = []
                
            
            combined_figure = plt.figure()
            combined_ax = combined_figure.add_subplot(1, 1, 1)
            combined_ax.set_title("ROC curves")
            
            #Decission tree
            print("Decission Tree")
            self.logger.info("Decission Tree")
            clf_dt = DecisionTreeClassifier(random_state=self.seed, 
                                            class_weight = self.cweight)
           
            dt_grid = {'criterion': ['gini', 'entropy'],
                       'min_samples_leaf': [5, 10, 15, 25],
                       'max_depth': [int(x) for x in np.linspace(2, 20, 10)]}
            
            self.dt, column_names, dt_info = self.model_pipeline(model_name = "Decission Tree", 
                                                                clf_model = clf_dt, 
                                                                grid_model = dt_grid)
            if self.dt is not None:
                self.dt.best_estimator_
                # tree.plot_tree(self.dt)
                
            self.model_info.append(dt_info)
            
            self.genetic_preprocess()    
        
            if self.flag_genetic == True:
                
                # self.genetic_preprocess()
                
                dt_info_genetic = self.MOG_genetic_pipeline(model_name = "Decission Tree", 
                                                                clf_model = clf_dt, 
                                                                grid_model = dt_grid)
                self.model_info_genetic.append(dt_info_genetic)
                
                
            
            #XGBoost
            print("Gradient Boosting")
            self.logger.info("Gradient Boosting")
            clf_gb = GradientBoostingClassifier(random_state=self.seed)
            
            gb_grid = {'n_estimators': [int(x) for x in np.linspace(200, 2000, 10)],
                       'learning_rate': [0.001, 0.01, 0.1],
                       'max_depth': [int(x) for x in np.linspace(2, 20, 10)]}
            
            self.xgb, column_names, xgb_info = self.model_pipeline(model_name = "Gradient Boosting", 
                                                                  clf_model = clf_gb, 
                                                                  grid_model = gb_grid)
            if self.xgb is not None:
                xgb_best = self.xgb.best_estimator_
                importances = pd.DataFrame(xgb_best.feature_importances_, index = column_names)
                xgb_info["FEATURE_IMPORTANCE"] =  importances
            
            self.model_info.append(xgb_info)
            
            
            if self.flag_genetic == True:
                
                xgb_info_genetic = self.MOG_genetic_pipeline(model_name = "Gradient Boosting", 
                                                             clf_model = clf_gb, 
                                                             grid_model = gb_grid)
                
                self.model_info_genetic.append(xgb_info_genetic)
            
    
            #Random Forest
            print("Random Forest")
            self.logger.info("Random Forest")
            clf_rf = RandomForestClassifier(random_state=self.seed, class_weight = self.cweight)
            
            rf_grid = {'criterion': ['gini', 'entropy'],
                       'min_samples_leaf': [5, 10, 15, 25],
                       'max_depth': [int(x) for x in np.linspace(2, 20, 10)],
                       'n_estimators': [int(x) for x in np.linspace(200, 2000, 10)],}
            
            
            self.rf, column_names, rf_info = self.model_pipeline(model_name = "Random Forest", 
                                                                 clf_model = clf_rf, 
                                                                 grid_model = rf_grid)
            if self.rf is not None:
                #importances = rf.feature_importances_
                rf_best = self.rf.best_estimator_
                importances = pd.DataFrame(rf_best.feature_importances_, index = column_names)
                rf_info["FEATURE_IMPORTANCE"] =  importances
            
            self.model_info.append(rf_info)
    
            if self.flag_genetic == True:
                rf_info_genetic = self.MOG_genetic_pipeline(model_name = "Random Forest", 
                                                            clf_model = clf_rf, 
                                                            grid_model = rf_grid)
                
                self.model_info_genetic.append(rf_info_genetic)
            
            #KNN
            print("KNN")
            self.logger.info("KNN")
            clf_knn = KNeighborsClassifier()
            knn_grid = {'n_neighbors': [int(x) for x in np.linspace(2, 50, 25)]}
            
            self.knn, column_names, knn_info = self.model_pipeline(model_name = "KNN", 
                                                                  clf_model = clf_knn, 
                                                                  grid_model = knn_grid)
            
            self.model_info.append(knn_info)
            
            if self.flag_genetic == True:
                knn_info_genetic = self.MOG_genetic_pipeline(model_name = "KNN", 
                                                                      clf_model = clf_knn, 
                                                                      grid_model = knn_grid)
                self.model_info_genetic.append(knn_info_genetic)
            
            #SVC
            print("SVC")
            self.logger.info("SVC")
            clf_svc = SVC(probability= True, class_weight = self.cweight, max_iter=10000)
            svc_grid = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                        'gamma': ['scale', 'auto'],
                        'degree': [2, 3, 4],
                        'C': [0.1, 1, 10, 100]}
            
            self.svc, column_names, svc_info = self.model_pipeline(model_name = "SVC", 
                                                                  clf_model = clf_svc, 
                                                                  grid_model = svc_grid)
            
            self.model_info.append(svc_info)
            
            if self.flag_genetic == True:
                svc_info_genetic = self.MOG_genetic_pipeline(model_name = "SVC", 
                                                            clf_model = clf_svc, 
                                                            grid_model = svc_grid)
                self.model_info_genetic.append(svc_info_genetic)
            
            
            #Logistic Regression
            print("LR")
            self.logger.info("LR")
            clf_lr= LogisticRegression(class_weight = self.cweight)
            lr_grid = {'penalty': ['l1', 'l2', 'elasticnet', None],
                       'C': [0.1, 1, 10, 100]
                        }
            
            self.lr, column_names, lr_info = self.model_pipeline(model_name = "Logistic Regression", 
                                                                clf_model = clf_lr, 
                                                                grid_model = lr_grid)
            if self.lr is not None:
                lr_best = self.lr.best_estimator_
                lr_coef = pd.DataFrame(lr_best.coef_, columns = column_names).T
                
                lr_info["FEATURE_COEFFICIENT"] =  lr_coef
            
            self.model_info.append(lr_info)
            
            if self.flag_genetic == True:
                lr_info_genetic = self.MOG_genetic_pipeline(model_name = "Logistic Regression", 
                                                            clf_model = clf_lr, 
                                                            grid_model = lr_grid)
                self.model_info_genetic.append(lr_info_genetic)
                
            if not os.path.isdir(self.model_path):
                os.makedirs(self.model_path, exist_ok=True)
                
            
            filename = self.model_path + '/model_info.pkl'
            with open(filename, "wb") as file: 
                pickle.dump(self.model_info, file)
                
            
            if self.model_info_genetic != []:
                if not os.path.isdir(self.genetic_path):
                    os.makedirs(self.genetic_path, exist_ok=True)
                
                filename = self.genetic_path + '/model_info.pkl'
                with open(filename, "wb") as file: 
                    pickle.dump(self.model_info_genetic, file)
                    
            self.combined_ax.legend(fontsize=14)
            self.combined_figure.savefig(f"{self.model_path}/TOTAL_ROC.png", dpi=250, bbox_inches='tight')
            plt.close(self.combined_figure)
            

        except Exception as e:
            self.logger.error(f"Error in model_definition: {e}")
    
    
    @log_execution_time 
    def complete_flow(self):
        
        
        # PROCESS TATA TO REMOVE MISSINGS AND WRONG VALUES
        print("Data process: eliminate missing, variables, infinite values...")
        self.logger.info("Data process: eliminate missing, variables, infinite values...")
        self.data_preprocessing()
        
        # Ensure there is enough data
        if not self.data.empty:
        
           # PREAPRE LABEL DEPENDING ON LABEL AND FLAGS
           print("Prepare data sample for model")
           self.logger.info("Prepare data sample for model")
           self.prepare_data()
               
           # Ensure minimum y_size to execute model development
           self.y_size = self.y.groupby(self.y).size()
           
           # Ensure that at least there are 2 classes in the sample
           if len(self.y_size) > 1:
               # MODEL CALCULATION
               if self.y_size.gt(3).all():
                   print("Model execution")
                   self.logger.info("Model execution")
                   
                   print("Data sampling")
                   self.logger.info("Data sampling")
                   self.data_sampling()
                   
                   print("Preprocessor")
                   self.logger.info("Preprocessor")
                   self.model_preprocess()
                   
                   print("Model calculation")
                   self.logger.info("Model calculation")
                   flag_model = 0
                   while flag_model == 0:
                       try:
                           self.model_definition()
                           flag_model = 1
                       except Exception as e:
                            print(f"Error: {e}")
               else:
                   print("Not enought targets in each category to develop model")
                   self.logger.info("Not enought targets in each category to develop model")
                   
           else:
               self.logger.info("Reduced number of classes")
        
        return
    
    

    # Example usage:

columns_analyze = [
    "CIPA",
    "TIPO_ICTUS",
    "OGV",
    "SEXO",
    "EDAD",
    "CTES_TAS",
    "CTES_TAD",
    "CTES_FC",
    "ICTUS_HEMIPARESIA",
    "ICTUS_AFASIAMOTORA",
    "ICTUS_PUNTUACION",
    "FARMACO",
    "FA",
    "ICTUS_PREVIO",
    "ANTICOAGULANTE"]

call = FeatureAnalysis(data = data,
                        name = [["IH_V1", "94_Vars"]],
                        columns_analyze = columns_analyze,
                        label = "TIPO_ICTUS",
                        cweight = True,
                        flag_sampling = 2,
                        seed = 42,
                        metric = "recall", 
                        flag_genetic = True,
                        flag_ischemic = False,
                        drop_MD = False)
                    
call.complete_flow()