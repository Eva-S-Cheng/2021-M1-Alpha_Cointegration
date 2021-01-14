# -*- coding: utf-8 -*-

# Librairies
#* py -m pip install mysql-connector
import mysql.connector
import datetime



def Create_Tables_SQL(cursor):
    #? Connexion au serveur

    #* Creation d'une table de données MySql
    cursor.execute("""use allDatas;""")
    # Creation de COMPOSITION
    cursor.execute("""drop table if exists allDatas.Composition;""")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS allDatas.Composition (
        id INT(10) NOT NULL,
        num_Stock INT NOT NULL,
        start_Date DATE NULL,
        end_Date DATE NULL,
        PRIMARY KEY (id));
    """)


    #Creation de DATAS
    cursor.execute("""drop table if exists allDatas.Datas;""")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS allDatas.Datas (
        id INT(50) NOT NULL,
        num_Stock INT NOT NULL,
        trade_Date DATE NULL,
        open_Value FLOAT NOT NULL,
        max_Value FLOAT NOT NULL,
        min_Value FLOAT NOT NULL,
        close_Value FLOAT NOT NULL,
        rendement FLOAT NOT NULL,
        PRIMARY KEY (id));
    """)
    
    #myConn.close()


def Insert_Datas_In_Composition(cursor,ref):
    #Ref doit etre un tuple
    cursor.execute("""INSERT INTO  allDatas.Composition(id, num_Stock, start_Date, end_Date) VALUES(%s, %s, %s, %s);""", ref)

def Insert_Datas_In_Datas(cursor, ref):
    #Ref doit etre un tuple
    cursor.execute(""" INSERT INTO alldatas.Datas(id, num_Stock,trade_Date,open_Value,max_Value,min_Value,close_Value,rendement) VALUES(%s, %s, %s, %s, %s, %s, %s, %s);""", ref)



#Extarction des donnees du fichier Composition.csv 
def Extraire_Dataset_Composition(path, cursor):
    composition = []
    myFile = open(path,"r")
    for line in myFile:
        tab_UneLigne = line.replace('\n','').split(';') #Type string
        #! 1e valeur = Numero du stock && 2e valeur =  Date d'entree du benchmark && 3e valeur = Date de sortie du benchmark
        #print(tab_UneLigne)
        if(tab_UneLigne[0] ==""):
            break #On a atteind la derniere ligne du fichier donc on arret de le lire
        tab_UneLigne[0] = int(tab_UneLigne[0])
        tab_UneLigne[1] = datetime.datetime.strptime(tab_UneLigne[1], '%d/%m/%Y')
        tab_UneLigne[1] = tab_UneLigne[1].date() #Récupère uniquement la date et pas l'heure
        tab_UneLigne[1] = tab_UneLigne[1].strftime("%Y-%m-%d") #Convertion en string pour insertion dans la database

        tab_UneLigne[2] = datetime.datetime.strptime(tab_UneLigne[2], '%d/%m/%Y')
        tab_UneLigne[2] = tab_UneLigne[2].date() #Récupère uniquement la date et pas l'heure
        tab_UneLigne[2] = tab_UneLigne[2].strftime("%Y-%m-%d") #Convertion en string pour insertion dans la database

        composition.append(tab_UneLigne)  
    myFile.close()

    
    #Ajout a la database
    for i in range(len(composition)):
        reference = composition[i]
        reference.insert(0,i) #On insert l'id en premiere position. ici l'ID est la valeur de i
        reference = tuple(reference) #Convertion d'une list en tuple pour insertion dans SQl
        Insert_Datas_In_Composition(cursor, reference)
        
    #return composition #Retourne le contenu du fichier Composition.csv


#Extarction des donnees des fichiers data1 a data5
def Extraire_Dataset_Data(path, cursor, first_Id_Value):
    data = []
    myFile = open(path,"r")
    for line in myFile:
        tempo = []
        tab_UneLigne = line.replace('\n','').split(';') #Type string
        #! 1e valeur = Numero du stock && 2e valeur =  Date d'exercice && 3e valeur = Val Ouverture
        #! 4e valeur = Val. Max && 5e valeur = Val. Min && 6e Valeur = Val.Fermeture && 7e valeur = Rendement
        #print(tab_UneLigne)
        if(tab_UneLigne[0] !="" and tab_UneLigne[1] !="" and tab_UneLigne[5] !="" and tab_UneLigne[6] !=""):
            #On a atteind une ligne avec des donnees manquantes
            tempo.extend([float(tab_UneLigne[0])])
            if('/' in tab_UneLigne[1]):
                #Cas de format de date avec des '/' et DD/MM/YYYY
                tab_UneLigne[1] = datetime.datetime.strptime(tab_UneLigne[1], '%d/%m/%Y')
            elif('-' in tab_UneLigne[1]):
                #Cas de format de date avec des '-' et YYYY-MM-DD
                tab_UneLigne[1] = datetime.datetime.strptime(tab_UneLigne[1], '%Y-%m-%d')

            tab_UneLigne[1] = tab_UneLigne[1].date()#Récupère uniquement la date et pas l'heure
            tempo.extend([tab_UneLigne[1].strftime("%Y-%m-%d")])  #Convertion en string pour insertion dans la database

            tempo.extend([float(tab_UneLigne[2])])
            tempo.extend([float(tab_UneLigne[3])])
            tempo.extend([float(tab_UneLigne[4])])
            tempo.extend([float(tab_UneLigne[5])])
            tempo.extend([float(tab_UneLigne[6])])

            data.append(tempo)          
    myFile.close()


    #Insertion dans la database
    for i in range(len(data)):
        reference = data[i]
        reference.insert(0,first_Id_Value + i) #On insert l'id en premiere position. ici l'ID est la valeur de i
        reference = tuple(reference) #Convertion d'une list en tuple pour insertion dans SQl
        Insert_Datas_In_Datas(cursor, reference)

    #return data #Retourne le contenu du fichier Composition.csv
    return len(data) #Retourne la taille du tableau qui nous servira de premier index pour le prochain id
    #Comme ca on garde une continuité des index (et des id) dans la database Datas




if __name__=='__main__' :

    #! Il faut en amont activer son serveur SQL
    #! Remplir avec les user et passwords de chacun
    myConn = mysql.connector.connect(host= "127.0.0.1", port="3306",
                                    user="root", password="root",
                                    database="allDatas")
    cursor = myConn.cursor()
    
    #* Section des operations a realiser sur la database

    #? Create_Tables_SQL(cursor) #Creation des tables Composition et Datas
    
    """
    reference = (1,1,"2007-01-01","2020-09-24")
    Insert_Datas_In_Composition(cursor, reference) #Ajout d'une valeur dans la table Composition
    reference =(1, 1, "2009-01-01", 31.5257,32.6318,31.3677,31.8493,0.4986)
    Insert_Datas_In_Datas(cursor,reference) #Ajout d'une valeur dans la table Datas
    """
    PATH_General = "Datas/"
    #PATH_General = "Projet/Datas/"
    PATH_Compositon = PATH_General + "Composition.csv"
    #? Extraire_Dataset_Composition(PATH_Compositon, cursor)

    PATH_Data1 = PATH_General+"data1.csv"
    PATH_Data2 = PATH_General+"data2.csv"
    PATH_Data3 = PATH_General+"data3.csv"
    PATH_Data4 = PATH_General+"data4.csv"
    PATH_Data5 = PATH_General+"data5.csv"
    first_Id = 0
    #Le first_Id sera l'id (la cle unique) du stock dans la database Datas
    #Il faut donc une continuité entre chaque lignes des differentes feuilles Excel
    #La méthode Extraire_Dataset_Data() renvoie donc la taille du tableau des datas 
    #La valeur de cette taille sera la valeur du premier id de la première ligne de la feuille Excel suivante
    """
    first_Id += Extraire_Dataset_Data(PATH_Data1,cursor,first_Id)
    first_Id += Extraire_Dataset_Data(PATH_Data2,cursor,first_Id)
    first_Id += Extraire_Dataset_Data(PATH_Data3,cursor,first_Id)
    first_Id += Extraire_Dataset_Data(PATH_Data4,cursor,first_Id)
    first_Id += Extraire_Dataset_Data(PATH_Data5,cursor,first_Id)
    """
    
    #Acceder a des données
    #Compter le nombre de stocks dans le fichier Composition
    cursor.execute(""" SELECT COUNT(DISTINCT(num_Stock)) FROM alldatas.composition; """)
    #cursor.execute(""" SELECT trade_Date, AVG(log(close_Value)) FROM alldatas.datas WHERE (trade_Date BETWEEN "2009-01-01" AND "2020-01-21") GROUP BY alldatas.datas.trade_Date ; """)
    
    results = cursor.fetchall()
    print(results)

    myConn.commit() #! Pour sauvegarder les données dans la table
    myConn.close() #Ferme le flux

