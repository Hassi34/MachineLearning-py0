import os
import opendatasets as od
import pandas as pd
import numpy as np
from zipfile import ZipFile
import joblib
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

# machine learning algorithms
from mlclasses import FinalProject
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from xgboost import XGBClassifier 
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier

#model validation
from sklearn.metrics import accuracy_score,classification_report, log_loss,roc_auc_score,precision_score,f1_score,recall_score, confusion_matrix, make_scorer,roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV,StratifiedKFold

# data visualization
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

class FinalProject:
    '''
    This class has number of methods for making the overall process 
    more effiecit by reusing the code.
    
    '''
    def __init__(self):
        pass
    def setup_data_repo (self,URL):
        '''
        Provide the URL to any kaggle competition and let this method setup the
        data repository for you.

        '''
        od.download(URL, force=True)
        p_name = URL.split('/')[-1]
        my_data_dir = os.listdir("./"+p_name)
        for file in my_data_dir:
            if file.endswith(".zip"):
                with ZipFile (p_name+"/"+file) as zip:
                    zip.extractall(path=p_name)
        for file in my_data_dir :
            if file.endswith(".zip"):
                complete_path = os.path.join(dir_name, file)
                os.remove(complete_path)
        return print('\033[1m'+f'Folder "{p_name}" now contains these files : {os.listdir(p_name)}'+'\033[0m')

    def impute_num(self, df_inputs, imputer = None):
        '''
        This method will take your dataframe as the input and will return the 3 variable as output:
        1) - imputer_vals : Impute values for all numerical features in the dataframe
        2) - df_output = Dataframe after the imputation
        3) - imputer = Imputer that could be used to perform the any numeric data 
        Note : if you already have an imputer, make sure to input as key value pair along with the data set 
        '''
        if imputer:
            imputer_vals = pd.DataFrame(imputer.statistics_, index=input_cols, columns=["Impute value"])
            df_output = imputer.transform(df_inputs[input_cols])
            return imputer_vals, df_output, imputer
        else:
            imputer = SimpleImputer(strategy='mean')
            imputer.fit(df_inputs[input_cols])
            imputer_vals = pd.DataFrame(imputer.statistics_, index=input_cols, columns=["Impute value"])
            df_output = imputer.transform(df_inputs[input_cols])
        return imputer_vals, df_output, imputer
    def num_scaler( self, df, scaler=None ):
        '''
        This method will take the numerical features and scale them using MinMax Scaler.
        It will return the scaled dataframe with a new scaler incase you don't bring your own.
        
        '''
        if scaler: 
            df = scaler.transform(df)
            return df, scaler        
        else:
            scaler = MinMaxScaler()
            scaler.fit(df)
            df = scaler.transform(df)
            return df, scaler
        
    def compare_models(self, models, X_train, y_train, X_val , y_val ):
        '''
        This method takes the dictionary of pre-defined dictionary of models along 
        with the respective dataframes to evaluate the model
        
        '''
        results = {}
        for i in models:
            acc_train = []
            acc_val = []
            precision = []
            recall = []
            roc_train = []
            roc_val = []
            f1 = []
            model = models[i].fit(X_train, y_train)
            y_predicted_train =  model.predict(X_train)
            y_predicted =  model.predict(X_val)
            
            y_predicted_prob_train = model.predict_proba(X_train)
            y_predicted_prob_score_train = y_predicted_prob_train[:,1]
            
            y_predicted_prob_val = model.predict_proba(X_val)
            y_predicted_prob_score_val = y_predicted_prob_val[:,1]

            acc_train.append(accuracy_score(y_train, y_predicted_train))
            acc_val.append(accuracy_score(y_val,y_predicted))
            precision.append(precision_score(y_val, y_predicted))
            recall.append(recall_score(y_val, y_predicted))
            roc_train.append(roc_auc_score(y_train, y_predicted_prob_score_train))
            roc_val.append(roc_auc_score(y_val, y_predicted_prob_score_val))
            f1.append(f1_score(y_val, y_predicted))
            results[i] = [acc_train, acc_val, precision, recall,roc_train, roc_val, f1]
        return pd.DataFrame(results, index=['Training Accuracy','Val Accuracy', 'Precision', 'Recall','Training Roc', 'Val ROC' , 'F1 Score' ])
    def plot_coef(self, classifier,X_train, y_train):
        '''
        This method takes a single classifier along with the X_train and y_train and will returns the plot of model coefficients
        
        '''
        classifier.fit(X_train, y_train)
        weights = classifier.coef_
        weights = np.array(weights).flatten()
        weights_df = pd.DataFrame({
            'columns': X_train.columns,
            'weight': weights
        }).sort_values('weight', ascending=False)
        weights_df["abs_value"] = weights_df["weight"].apply(lambda x: abs(x))
        weights_df["colors"] = weights_df["weight"].apply(lambda x: "green" if x > 0 else "red")
        weights_df = weights_df.sort_values("abs_value", ascending=False)

        fig, ax = plt.subplots(1, 1, figsize=(15, 5))
        sns.barplot(x="columns",
                    y="weight",
                    data=weights_df.head(30),
                   palette=weights_df.head(30)["colors"])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=15)
        ax.set_ylabel("Coef", fontsize=20)
        ax.set_xlabel("Feature Name", fontsize=20);
    
    def plot_cm(self,y_val, y_predicted):
        '''
        This method takes true y values and predicted y values to draw a Confusion Matrix
        
        '''
        conf_matrix = confusion_matrix(y_val,y_predicted)
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

        plt.xlabel('Predictions', fontsize=18)
        plt.ylabel('Actuals', fontsize=18)
        plt.title('Confusion Matrix', fontsize=18)
        plt.show()
        
    def best_weights(self, model ,X_train,y_train ):
        '''
        This method takes model name , features and target variables to perform the
        GridSearchCV on class weights and getting the best weights to maximize the F1 Score
        
        '''
        weights = np.linspace(0.05, 0.95, 20)
        gs = GridSearchCV(
        estimator= model,
        param_grid={'class_weight': [{0: x, 1: 1.0-x} for x in weights]},
                    scoring='roc_auc',
                    cv=5, verbose=2
        )
        grid_result = gs.fit(X_train, y_train)

        print(f"Best parameters : {grid_result.best_params_}")
        print(f"Best Score : {grid_result.best_score_}")
        df = pd.DataFrame({ 'score': grid_result.cv_results_['mean_test_score'],
                           'weight': weights })
        df.plot(x='weight');
    def plot_auc_curve(self,classifiers, X_train, y_train, X_val, y_val):
        '''
        This method will takes a list of classifiers, X_train, y_train, X_val and y_val. 
        It will return the AUC Curves for each the model provided in the input list.
        
        '''
        result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])

        for cls in classifiers:
            model = cls.fit(X_train, y_train)
            yproba = model.predict_proba(X_val)[::,1]

            fpr, tpr, _ = roc_curve(y_val,  yproba)
            auc = roc_auc_score(y_val, yproba)

            result_table = result_table.append({'classifiers':cls.__class__.__name__,
                                                'fpr':fpr, 
                                                'tpr':tpr, 
                                                'auc':auc}, ignore_index=True)

        result_table.set_index('classifiers', inplace=True)

        fig = plt.figure(figsize=(8,6))

        for i in result_table.index:
            plt.plot(result_table.loc[i]['fpr'], 
                     result_table.loc[i]['tpr'], 
                     label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))

        plt.plot([0,1], [0,1], color='orange', linestyle='--')

        plt.xticks(np.arange(0.0, 1.1, step=0.1))
        plt.xlabel("Flase Positive Rate", fontsize=15)
        plt.yticks(np.arange(0.0, 1.1, step=0.1))
        plt.ylabel("True Positive Rate", fontsize=15)
        plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
        plt.legend(prop={'size':13}, loc='lower right')
        plt.show()