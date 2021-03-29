# -*- coding: utf-8 -*-

####################
#* Projet PI2 - ESILV - A4
#* Equipe 46
# ####################

#region Librairies
#* py -m pip install mysql-connector
#* py -m pip install mysql-connector-python
import mysql.connector
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

#Pour l'ADF
import statsmodels.api as sm 
from statsmodels.tsa.stattools import adfuller

#Pour linear models
#* py -m pip install sklearn
#* py -m pip install regressors
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error #metrics pr MSE, R², MAE
from sklearn.model_selection import train_test_split
#from regressors import stats

import seaborn as sb #Pour la correlation
#endregion

#region Recuperation des datas sur MySQL
def Requete_Test_SelectAll(dbConn):
    """
    Methode de test pour faire une requete sur la BDD MySQL. A faire pour vérifier que le connexion avec MySQL fonctionne
    Inputs:
    * dbConn : Nom du stream avec MySQL

    Return:
        Ne retourne rien   
    """
    frame = pd.read_sql("SELECT * FROM alldatas.composition;", dbConn)

    pd.set_option('display.expand_frame_repr', False)
    print(frame)


#Methode retournant la composition de l'indice pour une date donnee
def Composition_Indice_Date_t(dbConn, myDate):
    """ Methode retournant la composition de l'indice pour une date donnee

    Inputs:
    * dbConn = Nom du stream avec MySQL
    * myDate = Date a laquelle on cherche la composition de l'indice
    
    Return:
    * frame = dataFrame de la compostion de l'indice a la date donnee
        Dataframe pandas
    """

    #! myDate DOIT être un STRING de la forme YYYY-MM-DD
    ref = tuple((myDate, myDate))

    frame = pd.read_sql("SELECT DISTINCT(num_Stock) FROM composition WHERE start_Date < %s AND end_Date > %s AND composition.num_Stock IN (SELECT DISTINCT(num_Stock) FROM Datas);", dbConn, params=ref)
    #Dans cette requete, on verifie que les sotcks composant l'indice ont bien des datas dasn la table Datas
    #En effet pour les stocks 23 et 757, ils sont dans la data Composition MAIS PAS dans la data Datas
    #Cela engendrera donc une erreur si on les utilise dans la compiosition mais pas dans Datas
    pd.set_option('display.expand_frame_repr', False)
    return frame


def Recreation_Indice(dbConn):
    """ Méthode permettant de recréer l'indice

    Inputs:
    * dbConn = Nom du stream avec MySQL
    
    Return:
    Ne retourne rien
    """
    frame = pd.read_sql("SELECT trade_Date, log(AVG(close_Value)) FROM datas GROUP BY datas.trade_Date ;", dbConn)
    pd.set_option('display.expand_frame_repr', False)
    return frame


def Benchmark_Btw_2Dates(df, end_Date, window_Days):
    """ Methode d'extraction des valeurs du benchmark entre 2 dates

    Inputs:
    * df = Dataframe dont il faut extraire une partie
        Dataframe Pandas
    * end_Date = Date de fin (format STRING YYYY-MM-DD)
        string
    * window_Days = Taille de la fenetre
        int
    
    Return:
    * benchmark_d1_to_d2 = Benchmark extrait
        Dataframe pandas
    """
    start_Date = datetime.datetime.strptime(end_Date, '%Y-%m-%d')
    start_Date = start_Date - datetime.timedelta(days = window_Days)
    start_Date = start_Date.date()

    end_Date = datetime.datetime.strptime(end_Date, '%Y-%m-%d').date()
    #start_Date = start_Date.strftime('%Y-%m-%d')

    benchmark_d1_to_d2 = df[(df['trade_Date'] >= start_Date) & (df['trade_Date']<= end_Date)]
    benchmark_d1_to_d2.reset_index(drop = True, inplace=True)
    return benchmark_d1_to_d2


# Methode retournant le nombre de jours où il y a eu des trades sur la fenetre donnee (ne compte pas les weekends et jours feriés)
def Nb_Jours_De_Trade(dbConn, end_Date, window_Days):
    """ Methode retournant le nombre de jours où il y a eu des trades sur la fenetre donnee (ne compte pas les weekends et jours feriés)

    Inputs:
    * df = Dataframe dont il faut extraire une partie
        Dataframe Pandas
    * end_Date = Date de fin (format STRING YYYY-MM-DD)
        string
    * window_Days = Taille de la fenetre
        int
    
    Return:
    nb_Jours = nombre de jours où il y a eu des trades sur la fenetre donnee
        int
    """

    #Creation de start_Date qui est end_Date - Window_Days
    start_Date = datetime.datetime.strptime(end_Date, '%Y-%m-%d')
    start_Date = start_Date - datetime.timedelta(days = window_Days)
    start_Date = start_Date.strftime('%Y-%m-%d')

    ref = tuple((start_Date, end_Date, end_Date, end_Date))
    frame = pd.read_sql("SELECT count(trade_Date) FROM datas WHERE (trade_Date BETWEEN %s AND %s) AND datas.num_Stock IN (SELECT num_Stock FROM composition WHERE start_Date < %s AND end_Date > %s) GROUP BY num_Stock LIMIT 1;", dbConn, params=ref)
    
    #On extrait la valeur de la dataframe qui est le nombre de jours de trade sur l'intervalle de temps [start_Date ; end_Date]
    nb_Jours = frame.iloc[0,0] 
    return nb_Jours


#Methode pour extraire les log(close_Price) des stocks contenus dans l'indice en date end_Date et pour une fenêtre de x jours
def Extract_LogClosePrice_Stocks_Btw_2Dates(dbConn, end_Date, window_days, nb_TradeDays):
    """ Methode pour extraire les log(close_Price) des stocks contenus dans l'indice en date end_Date et pour une fenêtre de x jours

    Inputs:
    * dbConn = Nom du stream avec MySQL
    * end_Date = Date de fin (format STRING YYYY-MM-DD)
        string
    * window_Days = Taille de la fenetre
        int
    * nb_TradeDays = Nombre de jours de trade effectifs
        int
    
    Return:
    * frameList = log(close_Price) des stocks contenus dans l'indice
        Dataframe pandas
    """
    
    #Creation de start_Date qui est end_Date - Window_Days
    start_Date = datetime.datetime.strptime(end_Date, '%Y-%m-%d')
    start_Date = start_Date - datetime.timedelta(days = window_days)
    start_Date = start_Date.strftime('%Y-%m-%d')

    ref = tuple((start_Date, end_Date, end_Date, end_Date))
    frame = pd.read_sql("SELECT log(close_Value) FROM datas WHERE (trade_Date BETWEEN %s AND %s) AND datas.num_Stock IN (SELECT DISTINCT(num_Stock) FROM composition WHERE start_Date < %s AND end_Date > %s AND composition.num_Stock IN (SELECT DISTINCT(num_Stock) FROM Datas));", dbConn, params=ref)
    #frame = pd.read_sql("SELECT num_stock, log(close_Value) FROM datas WHERE (trade_Date BETWEEN %s AND %s) AND datas.num_Stock IN (SELECT num_Stock FROM composition WHERE start_Date < %s AND end_Date > %s);", dbConn, params=ref)

    nb_Of_Stocks = int(len(frame)/nb_TradeDays) #len(frame) retourne le nombre de ligne dans la dataframe
    #/On divise par le nombre de jours de trade pour obtenir le nombre de stocks utilises
    frameList = np.array_split(frame, nb_Of_Stocks)

    return frameList


#Methode permettant de creer la matrice des Close prices en gardant uniquement les colonnes n'ayant pas de datas manquantes
# Ajoute en nom de colonne les numéros des stocks utilisés
def Create_Df_ClosePrice(all_Close_Price, compo_Indice_Date_t, nb_J):
    """ Methode permettant de creer la matrice des Close prices en gardant uniquement les colonnes n'ayant pas de datas manquantes

    Inputs:
    * all_Close_Price = liste contenant tous les prix de clotures 
        list[float]
    * compo_Indice_Date_t = composition de l'indice a une date t
        list[int]
    * nb_TradeDays = Nombre de jours de trade effectifs
        int
    
    Return:
    * final_Df_All_ClosePrice = matrice des Close prices avec en indice les numeros des stocks
        Dataframe pandas
    """
    
    #Dataframe devant a la fin contenir tous les closes prices ordonnés par colonne avec chaque colonne 1 stock
    #Le nombre de lignes sera la taille de la fenêtre des trade_Date
    matrix_All_ClosePrice = np.asarray(all_Close_Price[0])

    #On split la df en sous-dataframe dans une list
    #Chaque sous-dataframe correspond à 1 colonne (donc 1 stock) sur la période de la fenêtre
    for i in range(len(compo_Indice_Date_t)-1):

        if len(all_Close_Price[i+1]) == nb_J :
            #Si toute la colonne a des valeurs (le stock était dans la composition sur toute la durée de la fenêtre)
            matrix_All_ClosePrice= np.append(matrix_All_ClosePrice,(np.asarray(all_Close_Price[i+1])),axis=1)
            #Rmq : Si ce n'est pas respecté alors le stock n'est pas ajouté à la matrice car il manque des valeurs
        
        else:
            #Il manque des prix car le stock n'etait pas tous les jours dans la composition
            compo_Indice_Date_t.drop([i+1], inplace = True)
            #On supprime les indices dans compo_Indice_Date_t des stock que l'on a retiré de la matrice matrix_All_ClosePrice
    
    #On est sorti du for et on reset les index de la df compo_Indice_Date_t
    compo_Indice_Date_t.reset_index(drop = True, inplace=True)
    #print(compo_Indice_Date_t)

    #print(matrix_All_ClosePrice)
    column_Names = compo_Indice_Date_t.iloc[:,0]
    column_Names = list(column_Names)
    #print(column_Names)

    #On renomme les colonne de notre matrice avec uniquement les stocks ayant toutes les datas
    final_Df_All_ClosePrice = pd.DataFrame(matrix_All_ClosePrice, columns=column_Names)
    
    return final_Df_All_ClosePrice

#endregion

#region Regression lineaire & ADF

#Methode pour diviser un dataset en 2 sous-dataset avec des pourcentages précis
def Split_Df_Train_Test(df_x, df_y, percent_Test):
    """ Methode pour diviser un dataset en 2 sous-dataset avec des pourcentages précis

    Inputs:
    * df_x = dataframes à diviser 
        Dataframe pandas
    * df_y = dataframes à diviser
        Dataframe pandas
    * percent_Test = pourcentage (0.20 pour 20%) de séparation des Train et Test datas
        float
    
    Return:
    * train_x = Training set des inputs
        Dataframe pandas
    * test_x = Test set des inputs
        Dataframe pandas
    * train_y = Training set des outputs
        Dataframe pandas
    * test_y = Test set des outputs
        Dataframe pandas
    """

    train_x, test_x, train_y, test_y = train_test_split(df_x, df_y, test_size = percent_Test, random_state = 3, shuffle = True)
    train_x.reset_index(drop = True, inplace=True)
    test_x.reset_index(drop = True, inplace=True)
    train_y.reset_index(drop = True, inplace=True)
    test_y.reset_index(drop = True, inplace=True)

    return train_x, test_x, train_y, test_y


def Fit_Model(X_train, Y_train):
    """ Méthode pour fiter un modele lineaire 

    Inputs:
    * X_train = Training set des inputs
        Dataframe pandas
    * Y_train = Training set des outputs
        Dataframe pandas
    
    Return:
    * linearModel = Modèle linéaire 
    """
    model = LinearRegression(fit_intercept=True)
    linearModel = model.fit(X_train, Y_train['log(AVG(close_Value))'])

    return linearModel


def Print_Results(model, X_test, Y_test, Pred):
    """ Méthode pour fiter un modele lineaire 

    Inputs:
    * model = Modele lineaire fité
    * X_test = Test set des inputs
        Dataframe pandas
    * Y_test = Test set des outputs
        Dataframe pandas
    * Pred = Predictions (True outputs)
        list[float]
    
    Return:
    Ne retourne rien
    """

    print("\n** INPUTS > \n",X_test)
    print("\n** Our PREDICTIONS > \n",Pred)
    print("\n** OUTPUTS WANTED > \n", Y_test)

    #Coefficients
    coeff_Beta_model = model.coef_
    print("\n* Coefficients du model > \n", coeff_Beta_model, "\n\n")

    #Calcul du MSE pr les predictions et true_Outputs
    MSE_model = mean_squared_error(Y_test['log(AVG(close_Value))'], Pred)
    print("\n$ MSE = ", MSE_model)
    R2_model = r2_score(Y_test['log(AVG(close_Value))'], Pred)
    print("$ R2 = ", R2_model)
    MAE_model = mean_absolute_error(Y_test['log(AVG(close_Value))'], Pred)
    print("$ MAE = ", MAE_model)

    #list_p_value = stats.coef_pval(model, X_train, Y_train['log(AVG(close_Value))'])
    #print("\n* p_values > \n", list_p_value)
    #print("\n\n",list_p_value[0])


def ADF(Y_test, Pred):
    """ Méthode pour fiter un modele lineaire 

    Inputs:
    * Y_test = Test set des outputs
        Dataframe pandas
    * Pred = Predictions (True outputs)
        list[float]
    
    Return:
    * True = Serie cointegree
    * False = Serie non-cointegree
    """
    residuals = Y_test['log(AVG(close_Value))'] - Pred
    result = adfuller(residuals)
    print('ADF Statistic: %f' % result[0])
    #print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
        
    if result[0] < result[4]["5%"]:
        print ("Reject Ho - Time Series is Stationary => Portfolio is Stationary and the stocks are cointegrated with more than 95% certainty")
        return True
    else:
        print ("Failed to Reject Ho - Time Series is Non-Stationary\n\n")
        return False

#endregion


def Select_Rendement(dbConn, end_Date, window_days, nb_TradeDays, compo_Indice_Date_t):
    """ Methode pour extraire les rendement des stocks contenus dans l'indice en date end_Date et pour une fenêtre de x jours

    Inputs:
    * dbConn = Nom du stream avec MySQL
    * end_Date = Date de fin (format STRING YYYY-MM-DD)
        string
    * window_Days = Taille de la fenetre
        int
    * nb_TradeDays = Nombre de jours de trade effectifs
        int
    * compo_Indice_Date_t = Composition de l'indice a une date donnee
        list[int]
    
    Return:
    * mean_df_All_Yield = Moyenne des rendements du jour
        float
    """
    
    #Creation de start_Date qui est end_Date - Window_Days
    start_Date = datetime.datetime.strptime(end_Date, '%Y-%m-%d')
    start_Date = start_Date - datetime.timedelta(days = window_days)
    start_Date = start_Date.strftime('%Y-%m-%d')

    ref = tuple((start_Date, end_Date, end_Date, end_Date))
    frame = pd.read_sql("SELECT rendement FROM datas WHERE (trade_Date BETWEEN %s AND %s) AND datas.num_Stock IN (SELECT DISTINCT(num_Stock) FROM composition WHERE start_Date < %s AND end_Date > %s AND composition.num_Stock IN (SELECT DISTINCT(num_Stock) FROM Datas));", dbConn, params=ref)
    #frame = pd.read_sql("SELECT num_stock, log(close_Value) FROM datas WHERE (trade_Date BETWEEN %s AND %s) AND datas.num_Stock IN (SELECT num_Stock FROM composition WHERE start_Date < %s AND end_Date > %s);", dbConn, params=ref)

    nb_Of_Stocks = int(len(frame)/nb_TradeDays) #len(frame) retourne le nombre de ligne dans la dataframe
    #/On divise par le nombre de jours de trade pour obtenir le nombre de stocks utilises
    frameListYield = np.array_split(frame, nb_Of_Stocks)

    #Dataframe devant a la fin contenir tous les closes prices ordonnés par colonne avec chaque colonne 1 stock
    #Le nombre de lignes sera la taille de la fenêtre des trade_Date
    matrix_All_Yields = np.asarray(frameListYield[0])

    #On split la df en sous-dataframe dans une list
    #Chaque sous-dataframe correspond à 1 colonne (donc 1 stock) sur la période de la fenêtre
    for i in range(len(compo_Indice_Date_t)-1):

        if len(frameListYield[i+1]) == nb_TradeDays :
            #Si toute la colonne a des valeurs (le stock était dans la composition sur toute la durée de la fenêtre)
            matrix_All_Yields= np.append(matrix_All_Yields,(np.asarray(frameListYield[i+1])),axis=1)
            #Rmq : Si ce n'est pas respecté alors le stock n'est pas ajouté à la matrice car il manque des valeurs
        
        else:
            #Il manque des prix car le stock n'etait pas tous les jours dans la composition
            compo_Indice_Date_t.drop([i+1], inplace = True)
            #On supprime les indices dans compo_Indice_Date_t des stock que l'on a retiré de la matrice matrix_All_ClosePrice
    
    #On est sorti du for et on reset les index de la df compo_Indice_Date_t
    compo_Indice_Date_t.reset_index(drop = True, inplace=True)
    #print(compo_Indice_Date_t)

    #print(matrix_All_ClosePrice)
    column_Names = compo_Indice_Date_t.iloc[:,0]
    column_Names = list(column_Names)
    #print(column_Names)

    #On renomme les colonne de notre matrice avec uniquement les stocks ayant toutes les datas
    df_All_Yield = pd.DataFrame(matrix_All_Yields, columns=column_Names)
    mean_df_All_Yield = df_All_Yield.mean(axis = 1) #Moyenne des rendements du jour sur les lignes
    return mean_df_All_Yield



def Correlation_Matrix(df_stocks, df2):
    """ Methode de calcul de la matrice de correlation Stocks Vs Stocks && Stocks Vs Indice
    
    Inputs:
    df_stocks : Informations sur les stocks
        DataFrame pandas, float
    df2 : Informations sir l'indice
        Dataframe, float

    Return:
    corr_stocks_indice = Matrice de corrélation des Stocks Vs Indice
        Dataframe, float
    corr_stocks_stocks = Matrice de corrélation des Stocks Vs Stocks
        Dataframe, float   
    """

    matrix = pd.concat([df_stocks, df2.iloc[:,1]], axis = 1)
    #print(matrix)
    correlation_matrix = matrix.corr()
    #print(correlation_matrix)
    #plt.matshow(correlation_matrix.iloc[490:,490:], fignum="Corr_matshow")
    #plt.show()

    #Affichage de la matrice de correlation
    """
    plt.figure(num="Corr_heatmap")
    sb.heatmap(correlation_matrix.iloc[490:,490:], 
            xticklabels=correlation_matrix.iloc[490:,490:].columns,
            yticklabels=correlation_matrix.iloc[490:,490:].columns,
            cmap='RdBu_r',
            annot=True,
            linewidth=0.5)
    plt.show()
    """

    corr_stocks_indice = correlation_matrix.iloc[:len(correlation_matrix)-1, len(correlation_matrix.columns)-1] #Correlation entre les stocks et l'indice
    corr_stocks_stocks = correlation_matrix.iloc[0:len(correlation_matrix)-1, 0:len(correlation_matrix.columns)-1] #Correlation entre les stocks et eux-meme

    return corr_stocks_indice, corr_stocks_stocks

def Rendements_Indice(dbConn, end_Date, window_days):
    """ Methode permettant de calculer les rendement de l'indice entre 2 dates

    Inputs:
    * dbConn = Nom du stream avec MySQL
    * end_Date = Date de fin (format STRING YYYY-MM-DD)
        string
    * window_Days = Taille de la fenetre
        int
    
    Return:
    * rendement = rendement de l'indice entre 2 dates
        float
    """

    #Creation de start_Date qui est end_Date - Window_Days
    start_Date = datetime.datetime.strptime(end_Date, '%Y-%m-%d')
    start_Date = start_Date - datetime.timedelta(days = window_days)
    start_Date = start_Date.strftime('%Y-%m-%d')
    ref = tuple((start_Date, end_Date))

    frame = pd.read_sql("SELECT trade_Date, AVG(close_Value) FROM datas WHERE (trade_Date BETWEEN %s AND %s) GROUP BY datas.trade_Date ;", dbConn, params=ref)
    pd.set_option('display.expand_frame_repr', False)

    rendement = 100 * (frame.iloc[len(frame)-1, 1] - frame.iloc[0, 1])/frame.iloc[0, 1] # =(Valeur indice final - Valeur indice initiale) / Valeur indice initiale
    #On voudra savoir si le rendement est négatif ou positif
    return rendement #Rendement en pourcentage


#region Selection des stocks

def Select_Corr_Stocks_Stocks(df, seuil, panier_size):
    """Selection des stocks en fonction de la corrélation entre stocks

    Inputs :
    * df = dataframe de la correlation Stock vs Indice
        Dataframe pandas, float
    *seuil = Seuille de corrélation entre les stocks. Au-dessus de ce seuils les stocks ne sont pas pris
        float
    * panier_size = taille du panier
        int

    Return :
        Retourne les numéros des stocks composant le portefeuille suivant la taille du panier
    """

    temp = df.apply(lambda x: np.abs(x))
    col_sum = temp.sum(axis = 1, skipna = True)
    col_sum = col_sum / col_sum.size
    
    stocksToBeKept = col_sum[col_sum[:] < seuil]
    
    return stocksToBeKept.iloc[ : panier_size]

def Select_Corr_Stocks_Indice(df, panier_size):
    """Selection des stocks en fonction de la corrélation entre stocks et indice
    
    Inputs :
    * df = dataframe d'1 colonne de la correlation Stock vs Indice
        Dataframe pandas, float
    * panier_size = taille du panier
        int

    Return :
        Retourne les numéros des stocks composant le portefeuille suivant la taille du panier
    """
    #Tri de la dataframe qui est la amtrice de correlation Stocks-Indice
    #Tri decroissant pour avoir les plus hautes correlation en debut de dataframe
    df_sorted = df.sort_values('index', ascending = False)
    return df_sorted.iloc[ : panier_size]

#endregion


def Rendement_Percent(df):
    """Methode pour calculer le rendement en pourcentage et pouvoir le comparer
    
    Input :
    * df = Dataframe ayant 1 colonne
        Dataframe pandas d'1 colonne, floats

    Return :
    * df = Dataframe contenant l'évolution des rendements
        Dataframe pandas d'1 colonne, floats
    """
    initial_Value = 100
    for i in df.index:
        df.loc[i] = initial_Value*(1 + (df.loc[i])/100)
        initial_Value = df.loc[i]
    return df

def Plot_Indice_Stocks(list_df, list_label_Df, fig_title):
    """Methode de plot des rendements de l'indice
    
    Inputs :
    *list_df = liste de toutes les dataframes a plot sur la meme figure
        list[Dataframes]
    *lis_tabel_Df = nom (label) de chaque courbe. De MEME TAILLE que list_df
        list[string]
    *fig_title = Titre de la figure et du plot
        string

    Return :
        Ne retourne rien
    """
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.suptitle(fig_title)
    for i in range(len(list_df)):
        ax.plot(list_df[i], label = list_label_Df[i])
    ax.grid(True)
    ax.legend(loc='best')
    plt.show()


def Rendement_List_Stocks(dbConn, end_Date, window_days, list_Stocks):
    """Methode retournant la moyenne des rendements d'une liste de stocks entre 2 dates"""
    
    #Creation de start_Date qui est end_Date - Window_Days
    start_Date = datetime.datetime.strptime(end_Date, '%Y-%m-%d')
    start_Date = start_Date - datetime.timedelta(days = window_days)
    start_Date = start_Date.strftime('%Y-%m-%d')

    #Jointure des valeurs de list_Stocks en 1 string unique
    for i in range(len(list_Stocks)):
        list_Stocks[i] = str(list_Stocks[i])    
    str_List_Stocks = ','.join(list_Stocks)

    ref = tuple((start_Date, end_Date, str_List_Stocks))

    frame = pd.read_sql("SELECT AVG(rendement) FROM datas WHERE (trade_Date BETWEEN %s AND %s) AND datas.num_Stock IN (%s) GROUP BY datas.trade_Date ;", dbConn, params=ref)
    pd.set_option('display.expand_frame_repr', False)

    return frame #Moyenne des rendements

def Rendement_List_Stocks_Plus(dbConn, end_Date, window_days, list_Stocks,alpha):
    """Methode retournant la moyenne des rendements d'une liste de stocks entre 2 dates"""
    
    #Creation de start_Date qui est end_Date - Window_Days
    start_Date = datetime.datetime.strptime(end_Date, '%Y-%m-%d')
    start_Date = start_Date - datetime.timedelta(days = window_days)
    start_Date = start_Date.strftime('%Y-%m-%d')

    #Jointure des valeurs de list_Stocks en 1 string unique
    for i in range(len(list_Stocks)):
        list_Stocks[i] = str(list_Stocks[i])    
    str_List_Stocks = ','.join(list_Stocks)

    ref = tuple((start_Date, end_Date, str_List_Stocks))

    frame = pd.read_sql("SELECT AVG(rendement) FROM datas WHERE (trade_Date BETWEEN %s AND %s) AND datas.num_Stock IN (%s) GROUP BY datas.trade_Date ;", dbConn, params=ref)
    pd.set_option('display.expand_frame_repr', False)
    frame['AVG(rendement)']=frame['AVG(rendement)']+alpha/252
    return frame #Moyenne des rendements

def Rendement_List_Stocks_Minus(dbConn, end_Date, window_days, list_Stocks,alpha):
    """Methode retournant la moyenne des rendements d'une liste de stocks entre 2 dates"""
    
    #Creation de start_Date qui est end_Date - Window_Days
    start_Date = datetime.datetime.strptime(end_Date, '%Y-%m-%d')
    start_Date = start_Date - datetime.timedelta(days = window_days)
    start_Date = start_Date.strftime('%Y-%m-%d')

    #Jointure des valeurs de list_Stocks en 1 string unique
    for i in range(len(list_Stocks)):
        list_Stocks[i] = str(list_Stocks[i])    
    str_List_Stocks = ','.join(list_Stocks)

    ref = tuple((start_Date, end_Date, str_List_Stocks))

    frame = pd.read_sql("SELECT AVG(rendement) FROM datas WHERE (trade_Date BETWEEN %s AND %s) AND datas.num_Stock IN (%s) GROUP BY datas.trade_Date ;", dbConn, params=ref)
    pd.set_option('display.expand_frame_repr', False)
    frame['AVG(rendement)']=frame['AVG(rendement)']-alpha/252
    return frame #Moyenne des rendements


#region Section Rebalancement

#! Idee amelioration : Proposer le choix de methode de selection du portfolio (Correlation Stock Vs Stock, Stock Vs Indice, autre)
def Rebalancement_UnCycle(dbConn, previous_Active_Stocks, actual_Date, rebalancing_size, taille_panier, fenetre_Analyse, benchmark):
    """Methode faisant un unique rebalancement et renvoyant la nouvelle liste des stocks
    
    Inputs :
    * dbConn = connexion a la base de donne MySQL
    * previous_Active_Stocks = Liste des stocks actuellement dans le portefeuille et devant être changés
    * actual_Date = date actuelle lors du rebalancement
    * rebalancing_size = nombre de jours entre chaque rebalancement = Frequence de rebalancement
    * taille_panier = taille du portefeuille
    * fenetre_Analyse = taille de la fenêtre sur laquelle nous allons baser notre analyse pour rebalancer notre portfeuille (ne pas la prendre trop petite ou trop grande, 200 est bien)
    * benchmark = recration de l'indice

    Return :
    * list_NumStocks = Numéro des stocks composant le nouveau portefeuille sur 1 cycle
        list[float]
    * df_Rendements_portfolio_lastPeriod = Rendementsdu portefeuille passé en input (ancien qui a été rebalancé)
        DataFrame pandas, Float
    * df_Rendements_indice_lastPeriod = Rendementsdu de l'indice passé en input
        DataFrame pandas, Float
    """

    #Etape 1 :
    # Recuperer le rendement moyen de chaque jour depuis le dernier rebalancement en fonction des stocks composant le panier actuellement (ces stocks n'ont pas encore ete changes)
    #Rendements du portefeuille
    df_Rendements_portfolio_lastPeriod = Rendement_List_Stocks(dbConn, actual_Date, rebalancing_size, previous_Active_Stocks)

    #Elements pour determiner les rendements de l'indice sur la meme periode
    indice_previous_Date = datetime.datetime.strptime(actual_Date, '%Y-%m-%d')#transformation en type datetime
    indice_previous_Date = indice_previous_Date - datetime.timedelta(days = rebalancing_size)#On se replace une semaine avant pour connaitre la composition de l'indice a ce moment la
    indice_previous_Date = indice_previous_Date.strftime('%Y-%m-%d')#Convertion en string
    #On determine la composition de l'indice au debut de la periode precedente
    compo_Indice_Date_t = Composition_Indice_Date_t(dbConn, indice_previous_Date) #Composition de l'indice à une date donnée (Environ 500 stocks)
    list_compo_indice = []
    list_compo_indice.extend(compo_Indice_Date_t.values[i,0] for i in range(len(compo_Indice_Date_t.values))) 
    #On obtient une liste et non une dataframe (besoin d'une liste pour la methode > Rendement_List_Stocks() )

    #Rendements de l'indice
    df_Rendements_indice_lastPeriod = Rendement_List_Stocks(dbConn, actual_Date, rebalancing_size, list_compo_indice)


    #Etape 2 :
    #Rebalancer le portefeuille
    print("\n\n*** Parametres ***\n")
    print(f"Date > {actual_Date} \nFenetre d'analyse > {fenetre_Analyse} jours \nPanier de {taille_panier} stocks")

    compo_Indice_Date_t = Composition_Indice_Date_t(dbConn, actual_Date) #Composition de l'indice à une date donnée (Environ 500 stocks)
    #print(compo_Indice_Date_t)
    #Nombre de jours où il y a eu des trades sur la fenetre (ne compte pas les weekends et jours feriés)
    nb_J = Nb_Jours_De_Trade(dbConn, actual_Date, fenetre_Analyse) #nb_J est un integer 
    #print(nb_J)
    print("Nombre de jours de trade effectifs sur la fenetre d'analyse > ", nb_J,"\n\n")

    all_Close_Price = Extract_LogClosePrice_Stocks_Btw_2Dates(dbConn, actual_Date, fenetre_Analyse, nb_J)
    #print(all_Close_Price)

    matrix_X_ClosePrice = Create_Df_ClosePrice(all_Close_Price, compo_Indice_Date_t, nb_J)
    #print("\ncompo_Indice_Date_t > \n", compo_Indice_Date_t)
    #pd.set_option('display.max_columns', 10) #Pour n'afficher que 6 colonnes (index compris)
    #print("\nMatrice des Close prices > \n", matrix_X_ClosePrice)

    matrix_Y_Benchmark = Benchmark_Btw_2Dates(benchmark, actual_Date, fenetre_Analyse)
    #print("\nMatrice du Benchmark sur la periode > \n", matrix_Y_Benchmark)


    #! Verification de la COINTEGRATION : Portefeuille stationnaire
    percentage_Test_Train = 0.4 #40%
    X_train, X_test, Y_train, Y_test = Split_Df_Train_Test(matrix_X_ClosePrice, matrix_Y_Benchmark, percentage_Test_Train)

    #We fit our model    
    ourModel = Fit_Model(X_train, Y_train)
    #Make predictions
    predictions = ourModel.predict(X_test)

    #Affichage des resultats
    #Print_Results(ourModel, X_test, Y_test, predictions)
    
    #TEST ADF
    cointegrated = ADF(Y_test, predictions) #True = Portfolio is Stationary and the stocks are cointegrated with more than 95% certainty
    
    if(cointegrated == True):
        #Cointegration OK
        print("\n*** Cointegration OK ***")
        #Selection des stocks 
        corr_stocks_indice, corr_stocks_stocks = Correlation_Matrix(X_train, Y_train)
        #print("\n* Correlation Stocks Vs Indice\n", corr_stocks_indice)
        #print("\n\n* Correlation Stocks Vs Stocks\n", corr_stocks_stocks)

        #Panier construit a partir d'une selection basee sur la correlation Stocks Vs Indice
        panier_Select_Corr_Stocks_Indice = Select_Corr_Stocks_Indice(corr_stocks_indice, taille_panier)
        print("**Panier construit a partir d'une selection basee sur les plus fortes correlations positives Stocks Vs Indice \n"
                +f"*Pour un panier de taille > {taille_panier} \n")
        #print(panier_Select_Corr_Stocks_Indice)

        #Panier construit a partir d'une selection basee sur la correlation Stocks Vs Stocks
        """
        critere_Correlation_Stocks_Stocks = 0.6 #= 60%
        panier_Select_Corr_Stocks_Indice = Select_Corr_Stocks_Stocks(corr_stocks_stocks, critere_Correlation_Stocks_Stocks, taille_panier)
        print("**Panier construit a partir d'une selection basee sur les plus fortes correlations positives Stocks Vs Indice \n"
                +f"*Pour un panier de taille > {taille_panier} \n")
        """

        list_NumStocks = sorted(list(panier_Select_Corr_Stocks_Indice.index.values)) #Liste triee uniquement des numeros des stocks composant le panier
        print("*Ce qui donne la liste de stocks :\n", list_NumStocks)

        #On retourne la nouvelle composition du portefeuille (numeros des stocks) + les rendements lors de la precedente fenetre de rebalancement
        return list_NumStocks, df_Rendements_portfolio_lastPeriod, df_Rendements_indice_lastPeriod
    
    else :
        #Processus non stationnaire 
        list_NumStocks = previous_Active_Stocks #On ne change pas le portefeuille car il n'y a PAS de Cointegration
        print("\n**** !!! ****\nPas de cointegration sur cette période de rebalancement\n\tLe portefeuille n'a donc pas été modifié")
        return list_NumStocks, df_Rendements_portfolio_lastPeriod, df_Rendements_indice_lastPeriod
        

def Rebalancement_UnCycle_Plus(dbConn, previous_Active_Stocks, actual_Date, rebalancing_size, taille_panier, fenetre_Analyse, benchmark, alpha):
    """Methode faisant un unique rebalancement et renvoyant la nouvelle liste des stocks"""
    """
    * dbConn = connexion a la base de donne MySQL
    * previous_Active_Stocks = Liste des stocks actuellement dans le portefeuille et devant être changés
    * actual_Date = date actuelle lors du rebalancement
    * rebalancing_size = nombre de jours entre chaque rebalancement = Frequence de rebalancement
    * taille_panier = taille du portefeuille
    * fenetre_Analyse = taille de la fenêtre sur laquelle nous allons baser notre analyse pour rebalancer notre portfeuille (ne pas la prendre trop petite ou trop grande, 200 est bien)
    * benchmark = recration de l'indice
    """

    #Etape 1 :
    # Recuperer le rendement moyen de chaque jour depuis le dernier rebalancement en fonction des stocks composant le panier actuellement (ces stocks n'ont pas encore ete changes)
    #Rendements du portefeuille
    df_Rendements_portfolio_lastPeriod = Rendement_List_Stocks_Plus(dbConn, actual_Date, rebalancing_size, previous_Active_Stocks, alpha)

    #Elements pour determiner les rendements de l'indice sur la meme periode
    indice_previous_Date = datetime.datetime.strptime(actual_Date, '%Y-%m-%d')#transformation en type datetime
    indice_previous_Date = indice_previous_Date - datetime.timedelta(days = rebalancing_size)#On se replace une semaine avant pour connaitre la composition de l'indice a ce moment la
    indice_previous_Date = indice_previous_Date.strftime('%Y-%m-%d')#Convertion en string
    #On determine la composition de l'indice au debut de la periode precedente
    compo_Indice_Date_t = Composition_Indice_Date_t(dbConn, indice_previous_Date) #Composition de l'indice à une date donnée (Environ 500 stocks)
    list_compo_indice = []
    list_compo_indice.extend(compo_Indice_Date_t.values[i,0] for i in range(len(compo_Indice_Date_t.values))) 
    #On obtient une liste et non une dataframe (besoin d'une liste pour la methode > Rendement_List_Stocks() )

    #Rendements de l'indice
    df_Rendements_indice_lastPeriod = Rendement_List_Stocks(dbConn, actual_Date, rebalancing_size, list_compo_indice)


    #Etape 2 :
    #Rebalancer le portefeuille
    print("\n\n*** Parametres ***\n")
    print(f"Date > {actual_Date} \nFenetre d'analyse > {fenetre_Analyse} jours \nPanier de {taille_panier} stocks")

    compo_Indice_Date_t = Composition_Indice_Date_t(dbConn, actual_Date) #Composition de l'indice à une date donnée (Environ 500 stocks)
    #print(compo_Indice_Date_t)
    #Nombre de jours où il y a eu des trades sur la fenetre (ne compte pas les weekends et jours feriés)
    nb_J = Nb_Jours_De_Trade(dbConn, actual_Date, fenetre_Analyse) #nb_J est un integer 
    #print(nb_J)
    print("Nombre de jours de trade effectifs sur la fenetre d'analyse > ", nb_J,"\n\n")

    all_Close_Price = Extract_LogClosePrice_Stocks_Btw_2Dates(dbConn, actual_Date, fenetre_Analyse, nb_J)
    #print(all_Close_Price)

    matrix_X_ClosePrice = Create_Df_ClosePrice(all_Close_Price, compo_Indice_Date_t, nb_J)
    #print("\ncompo_Indice_Date_t > \n", compo_Indice_Date_t)
    #pd.set_option('display.max_columns', 10) #Pour n'afficher que 6 colonnes (index compris)
    #print("\nMatrice des Close prices > \n", matrix_X_ClosePrice)

    matrix_Y_Benchmark = Benchmark_Btw_2Dates(benchmark, actual_Date, fenetre_Analyse)
    #print("\nMatrice du Benchmark sur la periode > \n", matrix_Y_Benchmark)


    #Recupérer les 500 stocks (ou 474) de la matrice + le Y pour l'index 5
    #On predit le y^ avec le model 
    #On compare le Y et y^ (ex: MSE ou autre)

    percentage_Test_Train = 0.4 #40%
    X_train, X_test, Y_train, Y_test = Split_Df_Train_Test(matrix_X_ClosePrice, matrix_Y_Benchmark, percentage_Test_Train)

    #CORRELATION
    corr_stocks_indice, corr_stocks_stocks = Correlation_Matrix(X_train, Y_train)
    #print("\n* Correlation Stocks Vs Indice\n", corr_stocks_indice)
    #print("\n\n* Correlation Stocks Vs Stocks\n", corr_stocks_stocks)

    #Panier construit a partir d'une selection basee sur la correlation Stocks Vs Indice
    panier_Select_Corr_Stocks_Indice = Select_Corr_Stocks_Indice(corr_stocks_indice, taille_panier)
    print("**Panier construit a partir d'une selection basee sur les plus fortes correlations positives Stocks Vs Indice \n"
            +f"*Pour un panier de taille > {taille_panier} \n")
    #print(panier_Select_Corr_Stocks_Indice)
    list_NumStocks = sorted(list(panier_Select_Corr_Stocks_Indice.index.values)) #Liste triee uniquement des numeros des stocks composant le panier
    print("*Ce qui donne la liste de stocks :\n", list_NumStocks)

    #On retourne la nouvelle composition du portefeuille (numeros des stocks) + les rendements lors de la precedente fenetre de rebalancement
    return list_NumStocks, df_Rendements_portfolio_lastPeriod, df_Rendements_indice_lastPeriod

def Rebalancement_UnCycle_Minus(dbConn, previous_Active_Stocks, actual_Date, rebalancing_size, taille_panier, fenetre_Analyse, benchmark, alpha):
    """Methode faisant un unique rebalancement et renvoyant la nouvelle liste des stocks"""
    """
    * dbConn = connexion a la base de donne MySQL
    * previous_Active_Stocks = Liste des stocks actuellement dans le portefeuille et devant être changés
    * actual_Date = date actuelle lors du rebalancement
    * rebalancing_size = nombre de jours entre chaque rebalancement = Frequence de rebalancement
    * taille_panier = taille du portefeuille
    * fenetre_Analyse = taille de la fenêtre sur laquelle nous allons baser notre analyse pour rebalancer notre portfeuille (ne pas la prendre trop petite ou trop grande, 200 est bien)
    * benchmark = recration de l'indice
    """

    #Etape 1 :
    # Recuperer le rendement moyen de chaque jour depuis le dernier rebalancement en fonction des stocks composant le panier actuellement (ces stocks n'ont pas encore ete changes)
    #Rendements du portefeuille
    df_Rendements_portfolio_lastPeriod = Rendement_List_Stocks_Minus(dbConn, actual_Date, rebalancing_size, previous_Active_Stocks, alpha)

    #Elements pour determiner les rendements de l'indice sur la meme periode
    indice_previous_Date = datetime.datetime.strptime(actual_Date, '%Y-%m-%d')#transformation en type datetime
    indice_previous_Date = indice_previous_Date - datetime.timedelta(days = rebalancing_size)#On se replace une semaine avant pour connaitre la composition de l'indice a ce moment la
    indice_previous_Date = indice_previous_Date.strftime('%Y-%m-%d')#Convertion en string
    #On determine la composition de l'indice au debut de la periode precedente
    compo_Indice_Date_t = Composition_Indice_Date_t(dbConn, indice_previous_Date) #Composition de l'indice à une date donnée (Environ 500 stocks)
    list_compo_indice = []
    list_compo_indice.extend(compo_Indice_Date_t.values[i,0] for i in range(len(compo_Indice_Date_t.values))) 
    #On obtient une liste et non une dataframe (besoin d'une liste pour la methode > Rendement_List_Stocks() )

    #Rendements de l'indice
    df_Rendements_indice_lastPeriod = Rendement_List_Stocks(dbConn, actual_Date, rebalancing_size, list_compo_indice)


    #Etape 2 :
    #Rebalancer le portefeuille
    print("\n\n*** Parametres ***\n")
    print(f"Date > {actual_Date} \nFenetre d'analyse > {fenetre_Analyse} jours \nPanier de {taille_panier} stocks")

    compo_Indice_Date_t = Composition_Indice_Date_t(dbConn, actual_Date) #Composition de l'indice à une date donnée (Environ 500 stocks)
    #print(compo_Indice_Date_t)
    #Nombre de jours où il y a eu des trades sur la fenetre (ne compte pas les weekends et jours feriés)
    nb_J = Nb_Jours_De_Trade(dbConn, actual_Date, fenetre_Analyse) #nb_J est un integer 
    #print(nb_J)
    print("Nombre de jours de trade effectifs sur la fenetre d'analyse > ", nb_J,"\n\n")

    all_Close_Price = Extract_LogClosePrice_Stocks_Btw_2Dates(dbConn, actual_Date, fenetre_Analyse, nb_J)
    #print(all_Close_Price)

    matrix_X_ClosePrice = Create_Df_ClosePrice(all_Close_Price, compo_Indice_Date_t, nb_J)
    #print("\ncompo_Indice_Date_t > \n", compo_Indice_Date_t)
    #pd.set_option('display.max_columns', 10) #Pour n'afficher que 6 colonnes (index compris)
    #print("\nMatrice des Close prices > \n", matrix_X_ClosePrice)

    matrix_Y_Benchmark = Benchmark_Btw_2Dates(benchmark, actual_Date, fenetre_Analyse)
    #print("\nMatrice du Benchmark sur la periode > \n", matrix_Y_Benchmark)


    #Recupérer les 500 stocks (ou 474) de la matrice + le Y pour l'index 5
    #On predit le y^ avec le model 
    #On compare le Y et y^ (ex: MSE ou autre)

    percentage_Test_Train = 0.4 #40%
    X_train, X_test, Y_train, Y_test = Split_Df_Train_Test(matrix_X_ClosePrice, matrix_Y_Benchmark, percentage_Test_Train)

    #CORRELATION
    corr_stocks_indice, corr_stocks_stocks = Correlation_Matrix(X_train, Y_train)
    #print("\n* Correlation Stocks Vs Indice\n", corr_stocks_indice)
    #print("\n\n* Correlation Stocks Vs Stocks\n", corr_stocks_stocks)

    #Panier construit a partir d'une selection basee sur la correlation Stocks Vs Indice
    panier_Select_Corr_Stocks_Indice = Select_Corr_Stocks_Indice(corr_stocks_indice, taille_panier)
    print("**Panier construit a partir d'une selection basee sur les plus fortes correlations positives Stocks Vs Indice \n"
            +f"*Pour un panier de taille > {taille_panier} \n")
    #print(panier_Select_Corr_Stocks_Indice)
    list_NumStocks = sorted(list(panier_Select_Corr_Stocks_Indice.index.values)) #Liste triee uniquement des numeros des stocks composant le panier
    print("*Ce qui donne la liste de stocks :\n", list_NumStocks)

    #On retourne la nouvelle composition du portefeuille (numeros des stocks) + les rendements lors de la precedente fenetre de rebalancement
    return list_NumStocks, df_Rendements_portfolio_lastPeriod, df_Rendements_indice_lastPeriod


def Rebalancement(dbConn, start_Date, end_Date, frequence_rebalancement, taille_panier, fenetre_Analyse, benchmark):
    """Methode generale faisant tous les rebalancements sur une periode donnee
    
    Inputs
    * dbConn = connexion a la base de donne MySQL
    * start_Date = date du debut du rebalancement
    * end_Date = date de fin du rebalancement
    * frequence_rebalancement = nombre de jours entre chaque rebalancement
    * taille_panier = taille du portefeuille
    * fenetre_Analyse = taille de la fenêtre sur laquelle nous allons baser notre analyse pour rebalancer notre portfeuille (ne pas la prendre trop petite ou trop grande, 200 est bien)
    * benchmark = recration de l'indice

    Returns
    *list_Historical_Composition_Portfolio = Composition dans le temps du portefeuille (numero des stocks)
        list[int]
    *df_Historical_Yield_Portfolio = Rendements du portefeuille sur la période
        DataFrame Pandas, float
    *df_Historical_Yield_Indice = Rendements de l'indice sur la période
        DataFrame Pandas, float
    """
    list_Historical_Composition_Portfolio = [[-1]*taille_panier] #List de la composition du portefeuille pour chaque periode de rebalancement
    #On initialise la liste à -1 pour pouvoir faire le 1e boucle du while
    df_Historical_Yield_Portfolio = None #DataFrame contenant les rendement du portefeuille pour chaque jour sur toute la duree du rebalancement 
    df_Historical_Yield_Indice = None #DataFrame contenant les rendement de l'indice pour chaque jour sur toute la duree du rebalancement 


    #On boucle sur nos rebalancement tant qu'on n'a pas atteint la date finale

    start_Date = datetime.datetime.strptime(start_Date, '%Y-%m-%d') #Transformation en objet datetime
    end_Date = datetime.datetime.strptime(end_Date, '%Y-%m-%d') #Transformation en objet datetime
    
    while(start_Date < end_Date):
        str_Start_Date = start_Date.strftime('%Y-%m-%d')
        new_Portfolio, pd_Last_Yield_portfolio, pd_Last_Yield_indice  = Rebalancement_UnCycle(dbConn, list_Historical_Composition_Portfolio[-1], str_Start_Date, frequence_rebalancement, taille_panier, fenetre_Analyse, benchmark)
        #list_Historical_Composition_Portfolio[-1] sera le dernier element de la liste list_Historical_Composition_Portfolio (ici une sous-liste)
        list_Historical_Composition_Portfolio.append(new_Portfolio)
        #print(f"\n## pd_Last_Yield_portfolio > \n {pd_Last_Yield_portfolio}")
        #print(f"\n## df_Historical_Yield_Portfolio > \n {df_Historical_Yield_Portfolio}")
        df_Historical_Yield_Portfolio = pd.concat([df_Historical_Yield_Portfolio, pd_Last_Yield_portfolio], ignore_index = True)
        #print(f"\n## NEW df_Historical_Yield_Portfolio > \n {df_Historical_Yield_Portfolio}")
        if(len(df_Historical_Yield_Portfolio.values) != 0):
            #Sinon on a un decalage d'une periode car a la premiere iteration df_Historical_Yield_Portfolio est toujours vide
            df_Historical_Yield_Indice = pd.concat([df_Historical_Yield_Indice, pd_Last_Yield_indice], ignore_index = True)

        start_Date = start_Date + datetime.timedelta(days = frequence_rebalancement) 

    del list_Historical_Composition_Portfolio[0] #Delete la sous-liste d'index 0 car initialisee a -1
    return list_Historical_Composition_Portfolio, df_Historical_Yield_Portfolio, df_Historical_Yield_Indice


def Rebalancement_Plus(dbConn, start_Date, end_Date, frequence_rebalancement, taille_panier, fenetre_Analyse, benchmark, alpha):
    """Methode generale faisant tous les rebalancements sur une periode donnee pour la LONG Leg
    
    * dbConn = connexion a la base de donne MySQL
    * start_Date = date du debut du rebalancement
    * end_Date = date de fin du rebalancement
    * frequence_rebalancement = nombre de jours entre chaque rebalancement
    * taille_panier = taille du portefeuille
    * fenetre_Analyse = taille de la fenêtre sur laquelle nous allons baser notre analyse pour rebalancer notre portfeuille (ne pas la prendre trop petite ou trop grande, 200 est bien)
    * benchmark = recration de l'indice
    """
    list_Historical_Composition_Portfolio = [[-1]*taille_panier] #List de la composition du portefeuille pour chaque periode de rebalancement
    #On initialise la liste à -1 pour pouvoir faire le 1e boucle du while
    df_Historical_Yield_Portfolio = None #DataFrame contenant les rendement du portefeuille pour chaque jour sur toute la duree du rebalancement 
    df_Historical_Yield_Indice = None #DataFrame contenant les rendement de l'indice pour chaque jour sur toute la duree du rebalancement 


    #On boucle sur nos rebalancement tant qu'on n'a pas atteint la date finale

    start_Date = datetime.datetime.strptime(start_Date, '%Y-%m-%d') #Transformation en objet datetime
    end_Date = datetime.datetime.strptime(end_Date, '%Y-%m-%d') #Transformation en objet datetime
    
    while(start_Date < end_Date):
        str_Start_Date = start_Date.strftime('%Y-%m-%d')
        new_Portfolio, pd_Last_Yield_portfolio, pd_Last_Yield_indice  = Rebalancement_UnCycle_Plus(dbConn, list_Historical_Composition_Portfolio[-1], str_Start_Date, frequence_rebalancement, taille_panier, fenetre_Analyse, benchmark, alpha)
        #list_Historical_Composition_Portfolio[-1] sera le dernier element de la liste list_Historical_Composition_Portfolio (ici une sous-liste)
        list_Historical_Composition_Portfolio.append(new_Portfolio)
        #print(f"\n## pd_Last_Yield_portfolio > \n {pd_Last_Yield_portfolio}")
        #print(f"\n## df_Historical_Yield_Portfolio > \n {df_Historical_Yield_Portfolio}")
        df_Historical_Yield_Portfolio = pd.concat([df_Historical_Yield_Portfolio, pd_Last_Yield_portfolio], ignore_index = True)
        #print(f"\n## NEW df_Historical_Yield_Portfolio > \n {df_Historical_Yield_Portfolio}")
        if(len(df_Historical_Yield_Portfolio.values) != 0):
            #Sinon on a un decalage d'une periode car a la premiere iteration df_Historical_Yield_Portfolio est toujours vide
            df_Historical_Yield_Indice = pd.concat([df_Historical_Yield_Indice, pd_Last_Yield_indice], ignore_index = True)

        start_Date = start_Date + datetime.timedelta(days = frequence_rebalancement) 

    del list_Historical_Composition_Portfolio[0] #Delete la sous-liste d'index 0 car initialisee a -1
    return list_Historical_Composition_Portfolio, df_Historical_Yield_Portfolio, df_Historical_Yield_Indice

def Rebalancement_Minus(dbConn, start_Date, end_Date, frequence_rebalancement, taille_panier, fenetre_Analyse, benchmark, alpha):
    """Methode generale faisant tous les rebalancements sur une periode donnee pour la SHORT leg
    
    * dbConn = connexion a la base de donne MySQL
    * start_Date = date du debut du rebalancement
    * end_Date = date de fin du rebalancement
    * frequence_rebalancement = nombre de jours entre chaque rebalancement
    * taille_panier = taille du portefeuille
    * fenetre_Analyse = taille de la fenêtre sur laquelle nous allons baser notre analyse pour rebalancer notre portfeuille (ne pas la prendre trop petite ou trop grande, 200 est bien)
    * benchmark = recration de l'indice
    """
    list_Historical_Composition_Portfolio = [[-1]*taille_panier] #List de la composition du portefeuille pour chaque periode de rebalancement
    #On initialise la liste à -1 pour pouvoir faire le 1e boucle du while
    df_Historical_Yield_Portfolio = None #DataFrame contenant les rendement du portefeuille pour chaque jour sur toute la duree du rebalancement 
    df_Historical_Yield_Indice = None #DataFrame contenant les rendement de l'indice pour chaque jour sur toute la duree du rebalancement 


    #On boucle sur nos rebalancement tant qu'on n'a pas atteint la date finale

    start_Date = datetime.datetime.strptime(start_Date, '%Y-%m-%d') #Transformation en objet datetime
    end_Date = datetime.datetime.strptime(end_Date, '%Y-%m-%d') #Transformation en objet datetime
    
    while(start_Date < end_Date):
        str_Start_Date = start_Date.strftime('%Y-%m-%d')
        new_Portfolio, pd_Last_Yield_portfolio, pd_Last_Yield_indice  = Rebalancement_UnCycle_Minus(dbConn, list_Historical_Composition_Portfolio[-1], str_Start_Date, frequence_rebalancement, taille_panier, fenetre_Analyse, benchmark, alpha)
        #list_Historical_Composition_Portfolio[-1] sera le dernier element de la liste list_Historical_Composition_Portfolio (ici une sous-liste)
        list_Historical_Composition_Portfolio.append(new_Portfolio)
        #print(f"\n## pd_Last_Yield_portfolio > \n {pd_Last_Yield_portfolio}")
        #print(f"\n## df_Historical_Yield_Portfolio > \n {df_Historical_Yield_Portfolio}")
        df_Historical_Yield_Portfolio = pd.concat([df_Historical_Yield_Portfolio, pd_Last_Yield_portfolio], ignore_index = True)
        #print(f"\n## NEW df_Historical_Yield_Portfolio > \n {df_Historical_Yield_Portfolio}")
        if(len(df_Historical_Yield_Portfolio.values) != 0):
            #Sinon on a un decalage d'une periode car a la premiere iteration df_Historical_Yield_Portfolio est toujours vide
            df_Historical_Yield_Indice = pd.concat([df_Historical_Yield_Indice, pd_Last_Yield_indice], ignore_index = True)

        start_Date = start_Date + datetime.timedelta(days = frequence_rebalancement) 

    del list_Historical_Composition_Portfolio[0] #Delete la sous-liste d'index 0 car initialisee a -1
    return list_Historical_Composition_Portfolio, df_Historical_Yield_Portfolio, df_Historical_Yield_Indice
#endregion


def InfoRatio(Portfolio,Benchmark):
    """ Methode de calcul de l'information Ratio
    Inputs:
    * Portfolio : Rendements du portefeuille 
        DataFrame pandas, float
    * Benchmark : Rendements de l'indice
        DataFrame pandas, float

    Return:
    * Valeur de l'information ratio
        float    
    """

    data = Portfolio["AVG(rendement)"]-Benchmark["AVG(rendement)"]
    moy = data.mean()
    vol = data.std()
    
    return moy/vol

def SharpeRatio(Portfolio, rf):
    """ Methode de calcul du Sharpe Ratio
    Inputs:
    * Portfolio : Rendements du portefeuille 
        DataFrame pandas, float
    * rf : Taux sans risque
        float

    Return:
    * Valeur du Sharpe ratio
        float    
    """
    vol=Portfolio["AVG(rendement)"].std()
    ret=Portfolio["AVG(rendement)"].mean()
    
    return (ret-rf)/vol


def LongShort(benchmark,alpha):
    """ Methode d'execution du Long-Short
    Inputs:
    * benchmark :
        DataFrame pandas, float
    * alpha
        float

    Return:
    * df_plus
        Rendements du portefeuille long
    * df_minus
        Rendements du portfeuille short    
    """
    
    df_plus, df_minus = None, None
    df_plus, df_minus = benchmark.copy(), benchmark.copy()
    df_plus['log(AVG(close_Value))'] = df_plus['log(AVG(close_Value))'] + alpha/252
    df_minus['log(AVG(close_Value))'] = df_minus['log(AVG(close_Value))'] - alpha/252

    return df_plus, df_minus


#? MAIN
if __name__=='__main__' :

    #! A adpater suivant vos ID / Psw / Ports de connexion / Nom de database
    dbConnection = mysql.connector.connect(host= "127.0.0.1", port="3306",
                                    user="root", password="root",
                                    database="allDatas", use_pure=True)

    #region Version test du main

    #Requette test
    #Requete_Test_SelectAll(dbConnection) 

    
#    myDate_End = "2019-01-21"
#    windowSize = 20 #Nombre de jours sur le calendrier (compte les weekends et jours feriés)
#    taille_panier = 30 #Taille du panier de stocks
#
#    print("\n*** Parametres ***\n")
#    print("Date > ", myDate_End,"\nFenetre de > ", windowSize," jours","\nPanier de ",taille_panier," stocks")
#
#    compo_Indice_Date_t = Composition_Indice_Date_t(dbConnection, myDate_End) #Composition de l'indice à une date donnée (Environ 500 stocks)
#    #print(compo_Indice_Date_t)
#    
#    benchmark = Recreation_Indice(dbConnection)
#    df_plus, df_minus=LongShort(benchmark, 0.05)
#    #print(benchmark)
#
#    #Nombre de jours où il y a eu des trades sur la fenetre (ne compte pas les weekends et jours feriés)
#    nb_J = Nb_Jours_De_Trade(dbConnection, myDate_End, windowSize) #nb_J est un integer 
#    #print(nb_J)
#    print("Nombre de jours de trade effectifs > ", nb_J,"\n\n")
#    
#
#    all_Close_Price = Extract_LogClosePrice_Stocks_Btw_2Dates(dbConnection, myDate_End, windowSize, nb_J)
#    #print(all_Close_Price)
#
#    matrix_X_ClosePrice = Create_Df_ClosePrice(all_Close_Price, compo_Indice_Date_t, nb_J)
#    #print("\ncompo_Indice_Date_t > \n", compo_Indice_Date_t)
#    #pd.set_option('display.max_columns', 10) #Pour n'afficher que 6 colonnes (index compris)
#    #print("\nMatrice des Close prices > \n", matrix_X_ClosePrice)
#
#    matrix_Y_Benchmark = Benchmark_Btw_2Dates(benchmark, myDate_End, windowSize)
#    #print("\nMatrice du Benchmark sur la periode > \n", matrix_Y_Benchmark)
#
#
#    #Recupérer les 500 stocks (ou 474) de la matrice + le Y pour l'index 5
#    #On predit le y^ avec le model 
#    #On compare le Y et y^ (ex: MSE ou autre)
#
#    percentage_Test_Train = 0.4 #40%
#    X_train, X_test, Y_train, Y_test = Split_Df_Train_Test(matrix_X_ClosePrice, matrix_Y_Benchmark, percentage_Test_Train)
#
#    
#    #We fit our model
#    
#    ourModel = Fit_Model(X_train, Y_train)
#    #Make predictions
#    #predictions = ourModel.predict(X_train)
#    predictions = ourModel.predict(X_test)
#
#    #Affichage des resultats
#    #Print_Results(ourModel, X_train, Y_train, predictions)
#    Print_Results(ourModel, X_test, Y_test, predictions)
#
#
#    print("\n$$ ANALYSE >\n")
#    print("* Le MSE est très proche de 0\n",
#            "* Le R-squared error est très proche de 1\n",
#            "* Le MAE Mean Absolute Error est très proche de 0\n",
#            "** ==> Bon modèle\n")
#    print("\n/!\ > Tous les p-values sont < 0.05\n",
#            "\t-Soit tous les stocks sont significatifs (ou alors leurs log(Price))\n",
#            "\t-Soit il y a un problème dans le modèle ou l'approche\n")
#    
#
#    matrix_MeanYield = Select_Rendement(dbConnection, myDate_End, windowSize, nb_J, compo_Indice_Date_t)
#    pd.set_option('display.max_columns', 10) #Pour n'afficher que 6 colonnes (index compris)
#    #print("\n*Matrice des rendements sur la période :\n", matrix_MeanYield) 
#
#    
#    #TEST ADF
#    #cointegrated = ADF(Y_test, predictions) #True = Portfolio is Stationary and the stocks are cointegrated with more than 95% certainty
#
#
#    sigma_plus=df_plus.std()
#    sigma_minus=df_minus.std()
#    sigma_benchmark=benchmark.std()
#    
#    corr_plus=df_plus.corrwith(benchmark)
#    corr_minus=df_minus.corrwith(benchmark)
#    
#
#    #CORRELATION
#    corr_stocks_indice, corr_stocks_stocks = Correlation_Matrix(X_train, Y_train)
#    #print("\n* Correlation Stocks Vs Indice\n", corr_stocks_indice)
#    #print("\n\n* Correlation Stocks Vs Stocks\n", corr_stocks_stocks)
#
#    rendement_Indice = Rendements_Indice(dbConnection, myDate_End, windowSize)
#    #print("\n*Rendement de l'indice", rendement_Indice,"%\n")
#    
#
#    #Panier construit a partir d'une selection basee sur la correlation Stocks Vs Stocks
#    critere_Correlation_Stocks_Stocks = 0.6 #= 60%
#    panier_Select_Corr_Stocks_Stocks = Select_Corr_Stocks_Stocks(corr_stocks_stocks, critere_Correlation_Stocks_Stocks, taille_panier)
#    
#    print("\n**Panier construit a partir d'une selection basee sur la correlation Stocks Vs Stocks \n" 
#            +f"*Pour une correlation inferieure a > {critere_Correlation_Stocks_Stocks} \n"
#            +f"*Pour un panier de taille > {taille_panier} \n")
#    
#    #print(panier_Select_Corr_Stocks_Stocks)
#    list_NumStocks = sorted(list(panier_Select_Corr_Stocks_Stocks.index.values)) #Liste triee uniquement des numeros des stocks composant le panier
#    #print("\n*Ce qui donne la liste de stocks :\n", list_NumStocks)
#
#    #panier construit a partir d'une selection basee sur la correlation Stocks Vs Indice
#    panier_Select_Corr_Stocks_Indice = Select_Corr_Stocks_Indice(corr_stocks_indice, taille_panier)
#    #print("\n**Panier construit a partir d'une selection basee sur les plus fortes correlations positives Stocks Vs Indice \n"
#    #        +f"*Pour un panier de taille > {taille_panier} \n")
#    #print(panier_Select_Corr_Stocks_Indice)
#    list_NumStocks = sorted(list(panier_Select_Corr_Stocks_Indice.index.values)) #Liste triee uniquement des numeros des stocks composant le panier
#    #print("\n*Ce qui donne la liste de stocks :\n", list_NumStocks)

    #endregion

    #region Rebalancement
    date_debutRebalancement = "2019-01-01"
    date_finRebalancement = "2019-01-31"
    frequence_rebalancement = 7 #7 jours = Toutes les semaines
    windowSize = 200 #Nombre de jours sur le calendrier (compte les weekends et jours feriés)
    taille_panier = 30 #Taille du panier de stocks

    benchmark = Recreation_Indice(dbConnection)
    #print(benchmark)

    #Rebalancement du portefeuille entre la date de debut et de fin
    list_Historical_Composition_Portfolio, df_Historical_Yield_Portfolio, df_Rendements_indice_lastPeriod = Rebalancement(dbConnection, date_debutRebalancement, date_finRebalancement, frequence_rebalancement, taille_panier, windowSize, benchmark )
    print("\nInformation Ratio : " + str(InfoRatio(df_Historical_Yield_Portfolio,df_Rendements_indice_lastPeriod)) + "\n")
    print("\nSharpe Ratio : " + str(SharpeRatio(df_Historical_Yield_Portfolio,0.05)) + "\n")
    #df_plus, df_minus = LongShort(benchmark, 0.05)
    
    #list_Historical_Composition_Portfolio_plus, df_Historical_Yield_Portfolio_plus, df_Rendements_indice_lastPeriod_plus = Rebalancement_Plus(dbConnection, date_debutRebalancement, date_finRebalancement, frequence_rebalancement, taille_panier, windowSize, benchmark, 0.05 )
    #list_Historical_Composition_Portfolio_minus, df_Historical_Yield_Portfolio_minus, df_Rendements_indice_lastPeriod_minus = Rebalancement_Minus(dbConnection, date_debutRebalancement, date_finRebalancement, frequence_rebalancement, taille_panier, windowSize, benchmark, 0.05 )

    
    #Recuperation des rendements de l'indice et du portfolio 
    #matrix_MeanYield = Select_Rendement(dbConnection, date_finRebalancement, windowSize, nb_J, compo_Indice_Date_t)
    #matrix_MeanYield = Rendement_Percent(matrix_MeanYield)
    
    df_Rendements_indice_lastPeriod = Rendement_Percent(df_Rendements_indice_lastPeriod)
    df_Historical_Yield_Portfolio = Rendement_Percent(df_Historical_Yield_Portfolio)
    
    #df_Rendements_indice_lastPeriod_plus = Rendement_Percent(df_Rendements_indice_lastPeriod_plus)
    #df_Historical_Yield_Portfolio_plus = Rendement_Percent(df_Historical_Yield_Portfolio_plus)
    
    #df_Rendements_indice_lastPeriod_minus = Rendement_Percent(df_Rendements_indice_lastPeriod_minus)
    #df_Historical_Yield_Portfolio_minus = Rendement_Percent(df_Historical_Yield_Portfolio_minus)

#    Plot_Indice_Stocks([df_Rendements_indice_lastPeriod, df_Rendements_indice_lastPeriod_plus, df_Rendements_indice_lastPeriod_minus], ["Indice","Plus", "Minus"], "Long-Short")
#    
    #print(df_Rendements_indice_lastPeriod)
    #print(df_Historical_Yield_Portfolio)

#    df_plus = Rendement_Percent(df_plus)
#    df_minus = Rendement_Percent(df_minus)

#    Plot_Indice_Stocks([df_Rendements_indice_lastPeriod, df_plus, df_minus], ["Indice","Plus", "Minus"], "Long-Short")

    #Plot des rendements de l'Indice sur la période
    #Plot_Indice_Stocks([matrix_MeanYield, df_Historical_Yield_Portfolio], ["Indice","Portfolio"], "Replication de l'indice")
    Plot_Indice_Stocks([df_Rendements_indice_lastPeriod, df_Historical_Yield_Portfolio], ["Indice","Portfolio"], "Replication de l'indice")
    #Plot_Indice_Stocks([df_Rendements_indice_lastPeriod_plus, df_Historical_Yield_Portfolio_plus], ["Indice","Portfolio"], "Replication de l'indice plus")
    #Plot_Indice_Stocks([df_Rendements_indice_lastPeriod_minus, df_Historical_Yield_Portfolio_minus], ["Indice","Portfolio"], "Replication de l'indice minus")


    #endregion
    
    
    
    dbConnection.close() #Fermeture du stream avec MySql
