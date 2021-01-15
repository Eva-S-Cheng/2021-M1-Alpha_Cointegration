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
    #frame = pd.read_sql("SELECT num_Stock, log(close_Value) FROM datas WHERE (trade_Date BETWEEN %s AND %s) AND datas.num_Stock IN (SELECT num_Stock FROM composition WHERE start_Date < %s AND end_Date > %s);", dbConn, params=ref)
    
    frameList = np.array_split(frame, nb_Of_Stocks)
    #frameList = split(frame, frame['trade_Date'])
    #frameList = frame.groupby(by='num_Stock')['log(close_Value)']
    #print(frame)
    #print(frameList)

    #frameList = frame.to_numpy()
    #print(frameList)
    #new_DF_Frame = frameList.reshape(15,502)
    #new_Df_X = pd.DataFrame(new_DF_Frame)
    #print(new_Df_X)

    return frameList



if __name__=='__main__' :

    dbConnection = mysql.connector.connect(host= "127.0.0.1", port="3306",
                                    user="root", password="root",
                                    database="allDatas")

    #Requete_Test_SelectAll(dbConnection) 

    myDate = "2019-01-21"
    compo_Indice_Date_t = Composition_Indice_Date_t(dbConnection, myDate)
    
    #benchmark = Recreation_Indice(dbConnection)
    #print(benchmark)

    nb_J = Nb_Jours_De_Trade(dbConnection, "2019-01-21", 20) #nb_J est un integer 
    print(nb_J)
    
    all_Close_Price = Extract_LogClosePrice_Stocks_Btw_2Dates(dbConnection, "2019-01-21", 20, len(compo_Indice_Date_t))
    #print(all_Close_Price)

    #A RENOMER
    final_Df = np.asarray(all_Close_Price[0])

    for i in range(len(compo_Indice_Date_t)-1):

        if len(all_Close_Price[i+1]) == nb_J :
            #compo_Indice_Date_t.drop([i+1], inplace = True)
            final_Df= np.append(final_Df,(np.asarray(all_Close_Price[i+1])),axis=1)

    print(final_Df)
    
    column_Name = compo_Indice_Date_t.values.tolist()
    new_Df_X = pd.DataFrame(final_Df, columns=column_Name)

    print(new_Df_X.iloc[:,0])

    dbConnection.close()
