# -*- coding: utf-8 -*-

# Librairies
import mysql.connector
import pandas as pd


def Requete_Test_SelectAll(dbConn):
    
    frame = pd.read_sql("SELECT * FROM alldatas.composition;", dbConn)

    pd.set_option('display.expand_frame_repr', False)
    print(frame)



def Composition_Indice_Date_t(dbConn, myDate):
    # dbConn est la connexion a la data base
    # myDate est la date a laquelle on cherche la composition de l'indice
    #! myDate DOIT être un STRING de la forme YYYY-MM-DD
    ref = tuple((myDate, myDate))
    #SELECT num_Stock FROM alldatas.composition WHERE start_Date < "2010-01-12" AND end_Date > "2010-01-12";
    #VALUES(%s, %s, %s, %s);""", ref
    frame = pd.read_sql("SELECT num_Stock FROM alldatas.composition WHERE start_Date < %s AND end_Date > %s;", dbConn, params=ref)
    pd.set_option('display.expand_frame_repr', False)
    print(frame)


if __name__=='__main__' :

    dbConnection = mysql.connector.connect(host= "127.0.0.1", port="3306",
                                    user="root", password="root",
                                    database="allDatas")

    #Requete_Test_SelectAll(dbConnection) 

    myDate = "2010-01-12"
    Composition_Indice_Date_t(dbConnection, myDate)

    dbConnection.close()
