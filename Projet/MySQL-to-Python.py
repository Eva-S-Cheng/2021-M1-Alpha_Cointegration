# -*- coding: utf-8 -*-

# Librairies
#* py -m pip install mysql-connector
#* py -m pip install mysql-connector-python
import mysql.connector
import pandas as pd
import numpy as np
import datetime

#Pour linear models
#* py -m pip install sklearn
#* py -m pip install regressors
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error #metrics pr MSE, R², MAE
from sklearn.model_selection import train_test_split
from regressors import stats



def Requete_Test_SelectAll(dbConn):
    
    frame = pd.read_sql("SELECT * FROM alldatas.composition;", dbConn)

    pd.set_option('display.expand_frame_repr', False)
    print(frame)


#Methode retournant la composition de l'indice pour une date donnee
#! DEMANDER A REMI CAR PRBL DE QUANTITE DE STOCKS (709 pr une date precise et non 500)
def Composition_Indice_Date_t(dbConn, myDate):
    # dbConn est la connexion a la data base
    # myDate est la date a laquelle on cherche la composition de l'indice
    #! myDate DOIT être un STRING de la forme YYYY-MM-DD
    ref = tuple((myDate, myDate))

    frame = pd.read_sql("SELECT num_Stock FROM composition WHERE start_Date < %s AND end_Date > %s;", dbConn, params=ref)
    pd.set_option('display.expand_frame_repr', False)
    return frame


#Méthode permettant de recréer l'indice
def Recreation_Indice(dbConn):
    frame = pd.read_sql("SELECT trade_Date, AVG(log(close_Value)) FROM datas GROUP BY datas.trade_Date ;", dbConn)
    pd.set_option('display.expand_frame_repr', False)
    return frame



def Benchmark_Btw_2Dates(df, end_Date, window_Days):
    start_Date = datetime.datetime.strptime(end_Date, '%Y-%m-%d')
    start_Date = start_Date - datetime.timedelta(days = window_Days)
    start_Date = start_Date.date()

    end_Date = datetime.datetime.strptime(end_Date, '%Y-%m-%d').date()
    #start_Date = start_Date.strftime('%Y-%m-%d')

    benchmark_d1_to_d2 = df[(df['trade_Date'] >= start_Date) & (df['trade_Date']<= end_Date)]
    return benchmark_d1_to_d2


# Methode retournant le nombre de jours où il y a eu des trades sur la fenetre donnee (ne compte pas les weekends et jours feriés)
def Nb_Jours_De_Trade(dbConn, end_Date, window_Days):
    #end_Date DOIT être un STRING de la forme YYYY-MM-DD

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
def Extract_LogClosePrice_Stocks_Btw_2Dates(dbConn, end_Date, window_days, nb_Of_Stocks):
    #end_Date DOIT être un STRING de la forme YYYY-MM-DD
    
    #Creation de start_Date qui est end_Date - Window_Days
    start_Date = datetime.datetime.strptime(end_Date, '%Y-%m-%d')
    start_Date = start_Date - datetime.timedelta(days = window_days)
    start_Date = start_Date.strftime('%Y-%m-%d')

    ref = tuple((start_Date, end_Date, end_Date, end_Date))
    frame = pd.read_sql("SELECT log(close_Value) FROM datas WHERE (trade_Date BETWEEN %s AND %s) AND datas.num_Stock IN (SELECT num_Stock FROM composition WHERE start_Date < %s AND end_Date > %s);", dbConn, params=ref)
    
    frameList = np.array_split(frame, nb_Of_Stocks)

    return frameList


#Methode permettant de creer la matrice des Close prices en gardant uniquement les colonnes n'ayant pas de datas manquantes
# Ajoute en nom de colonne les numéros des stocks utilisés
def Create_Df_ClosePrice(all_Close_Price, compo_Indice_Date_t):
    
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


#Methode pour diviser un dataset en 2 sous-dataset avec des pourcentages précis
def Split_Df_Train_Test(df_x, df_y, percent_Test):
    #* df_x, df_y sont les dataframes à diviser
    #* percent_Test sont les pourcentage (0.80 pour 80%) de séparation des Train et Test datas
    train_x, test_x, train_y, test_y = train_test_split(df_x, df_y, test_size = percent_Test, random_state = 3, shuffle = True)
    return train_x, test_x, train_y, test_y



def Fit_Model(X_train, Y_train, X_test):
    model = LinearRegression(fit_intercept=True)
    model.fit(X_train, Y_train['AVG(log(close_Value))'])

    return model


def Print_Results(model, X_test, Y_test, Pred):
    print("\n** INPUTS > \n",X_test)
    print("\n** Our PREDICTIONS > \n",Pred)
    print("\n** OUTPUTS WANTED > \n", Y_test)

    #Coefficients
    coeff_Beta_model = model.coef_
    print("\n* Coefficients du model > \n", coeff_Beta_model, "\n\n")

    #Calcul du MSE pr les predictions et true_Outputs
    MSE_model = mean_squared_error(Y_test['AVG(log(close_Value))'], Pred)
    print("\n$ MSE = ", MSE_model)
    R2_model = r2_score(Y_test['AVG(log(close_Value))'], Pred)
    print("$ R2 = ", R2_model)
    MAE_model = mean_absolute_error(Y_test['AVG(log(close_Value))'], Pred)
    print("$ MAE = ", MAE_model)

    list_p_value = stats.coef_pval(model, X_train, Y_train['AVG(log(close_Value))'])
    print("\n* p_values > \n", list_p_value)



#? MAIN
if __name__=='__main__' :

    #! A adpater suivant vos ID / Psw / Ports de connexion / Nom de database
    dbConnection = mysql.connector.connect(host= "127.0.0.1", port="3306",
                                    user="root", password="root",
                                    database="allDatas")

    #Requette test
    #Requete_Test_SelectAll(dbConnection) 

    myDate_End = "2018-01-21"
    windowSize = 100

    compo_Indice_Date_t = Composition_Indice_Date_t(dbConnection, myDate_End) #Composition de l'indice à une date donnée (Environ 500 stocks)
    #print(compo_Indice_Date_t)
    
    benchmark = Recreation_Indice(dbConnection)
    #print(benchmark)
    

    #Nombre de jours où il y a eu des trades sur la fenetre (ne compte pas les weekends et jours feriés)
    nb_J = Nb_Jours_De_Trade(dbConnection, myDate_End, windowSize) #nb_J est un integer 
    #print(nb_J)
    
    all_Close_Price = Extract_LogClosePrice_Stocks_Btw_2Dates(dbConnection, myDate_End, windowSize, len(compo_Indice_Date_t))
    #print(all_Close_Price)

    matrix_X_ClosePrice = Create_Df_ClosePrice(all_Close_Price, compo_Indice_Date_t)
    print("\ncompo_Indice_Date_t > \n", compo_Indice_Date_t)
    pd.set_option('display.max_columns', 10) #Pour n'afficher que 6 colonnes (index compris)
    print("\nMatrice des Close prices > \n", matrix_X_ClosePrice)

    matrix_Y_Benchmark = Benchmark_Btw_2Dates(benchmark, myDate_End, windowSize)
    print("\nMatrice du Benchmark sur la periode > \n", matrix_Y_Benchmark)


    #Recupérer les 500 stocks (ou 474) de la matrice + le Y pour l'index 5
    #On predit le y^ avec le model 
    #On compare le Y et y^ (ex: MSE ou autre)

    percentage_Test_Train = 0.2 #20%
    X_train, X_test, Y_train, Y_test = Split_Df_Train_Test(matrix_X_ClosePrice, matrix_Y_Benchmark, percentage_Test_Train)

    #We fit our model
    ourModel = Fit_Model(X_train, Y_train, X_test)
    #Make predictions
    predictions = ourModel.predict(X_test)
    #Affichage des resultats
    Print_Results(ourModel, X_test, Y_test, predictions)

    print("\n$$ ANALYSE >\n")
    print("* Le MSE est très proche de 0\n",
            "* Le R-squared error est très proche de 1\n",
            "* Le MAE Mean Absolte Error est très proche de 0\n",
            "** ==> Bon modèle\n")
    print("\n/!\ > Tous les p-values sont < 0.05\n",
            "\t-Soit tous les stocks sont significatifs (ou alors leurs log(Price))\n",
            "\t-Soit il y a un problème dans le modèle ou l'approche\n")


    dbConnection.close() #Fermeture du strem avec MySql
