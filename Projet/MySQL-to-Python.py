# -*- coding: utf-8 -*-

# Librairies
import mysql.connector
import pandas as pd


def Requete_Test_SelectAll():
    dbConnection = mysql.connector.connect(host= "127.0.0.1", port="3306",
                                    user="root", password="root",
                                    database="allDatas")

    frame = pd.read_sql("SELECT * FROM alldatas.composition;", dbConnection)

    pd.set_option('display.expand_frame_repr', False)

    print(frame)

    dbConnection.close()

    

if __name__=='__main__' :

    Requete_Test_SelectAll()