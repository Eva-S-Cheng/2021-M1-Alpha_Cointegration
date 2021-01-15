# -*- coding: utf-8 -*-

# Librairies
#* py -m pip install mysql-connector
#* py -m pip install mysql-connector-python
import mysql.connector
import pandas as pd
import numpy as np
import datetime


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
def Creat_Df_ClosePrice(all_Close_Price, compo_Indice_Date_t):
    
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



#? MAIN
if __name__=='__main__' :

    #! A adpater suivant vos ID / Psw / Ports de connexion / Nom de database
    dbConnection = mysql.connector.connect(host= "127.0.0.1", port="3306",
                                    user="root", password="root",
                                    database="allDatas")

    #Requette test
    #Requete_Test_SelectAll(dbConnection) 

    myDate = "2019-01-21"
    compo_Indice_Date_t = Composition_Indice_Date_t(dbConnection, myDate) #Composition de l'indice à une date donnée (Environ 500 stocks)
    #print(compo_Indice_Date_t)
    
    #benchmark = Recreation_Indice(dbConnection)
    #print(benchmark)

    #Nombre de jours où il y a eu des trades sur la fenetre (ne compte pas les weekends et jours feriés)
    nb_J = Nb_Jours_De_Trade(dbConnection, "2019-01-21", 20) #nb_J est un integer 
    #print(nb_J)
    
    all_Close_Price = Extract_LogClosePrice_Stocks_Btw_2Dates(dbConnection, "2019-01-21", 20, len(compo_Indice_Date_t))
    #print(all_Close_Price)

    matrix_X_ClosePrice = Creat_Df_ClosePrice(all_Close_Price, compo_Indice_Date_t)
    pd.set_option('display.max_columns', 6) #Pour n'afficher que 6 colonnes (index compris)
    print(matrix_X_ClosePrice)


    dbConnection.close() #Fermeture du strem avec MySql
