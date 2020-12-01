# -*- coding: utf-8 -*-
"""
@author: LANGLE Armand - Eq-46
"""

#Librairies
import datetime


#Extarction des donnees du fichier Composition.csv 
def Extraire_Dataset_Composition(path):
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
        tab_UneLigne[2] = datetime.datetime.strptime(tab_UneLigne[2], '%d/%m/%Y')
        tab_UneLigne[2] = tab_UneLigne[2].date() #Récupère uniquement la date et pas l'heure

        composition.append(tab_UneLigne)  
    myFile.close()

    return composition #Retourne le contenu du fichier Composition.csv


#Extarction des donnees des fichiers data1 a data5
def Extraire_Dataset_Data(path):
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

            tempo.extend([tab_UneLigne[1].date()]) #Récupère uniquement la date et pas l'heure
            tempo.extend([float(tab_UneLigne[5])])
            tempo.extend([float(tab_UneLigne[6])])

            data.append(tempo)          
    myFile.close()

    return data #Retourne le contenu du fichier Composition.csv

#Methode premettant de séparer un dataset en sous-dateset contenant uniquement les datas entre 2 dates (ces 2 dates comprises)
def Divide_Dataset_Composition_Btw_Dates(myData, date_Searched):
    dataset_Btw_Dates=[]
    for i in range(len(myData)):
        if(date_Searched >= myData[i][1] and date_Searched <= myData[i][2]):
            dataset_Btw_Dates.extend([myData[i][0]])
            #Si ma date recherchee est coprise dans les dates ou le stock etait present dans le benchmark, alors je l'ajoute au tableau des dataset
    
    return dataset_Btw_Dates #Tableau contenant tous les stocks actifs pour la date donnee

#Recupere toutes les infos des stocks presents dans l'indice pour une certaine date donnee
def Extraire_Stocks_From_Data(tab_Data, tab_Stocks_ID, date_Searched):
    #tab_Data : Tableau de toutes les datas
    #tab_Stock_ID : ID des stocks recherches (leur numero)
    dataset_Stocks_Actifs = []
    for oneData in tab_Data:
        if(oneData[0] in tab_Stocks_ID and oneData[1] == date_Searched):
            #oneData[0] : ID du stock
            #oneData[1] : Date d'exercice
            dataset_Stocks_Actifs.append(oneData)
    
    return dataset_Stocks_Actifs


def Prix_Journalier_Indice(data_Stocks):
    mean_Result = 0
    for oneStock in data_Stocks:
        #oneData[2] : Valeur de cloture
        mean_Result+=oneStock[2]

    return (mean_Result/len(data_Stocks))


def Write_Stocks_inCSV(tab_Values, file_Name):
    colonnes_Names_Csv = ['ID_Stock', 'Date','Close_Price','Price_Return']
    try:
        myWrtiteFile = open(file_Name, 'w')
        enteteColumn = ";".join(colonnes_Names_Csv)+"\n"
        myWrtiteFile.write(enteteColumn)

        for oneLigne in tab_Values:
            #ligneToWrite = ";".join(oneLigne)+"\n"
            ligneToWrite = str(oneLigne[0]) + ";"
            ligneToWrite += (oneLigne[1].isoformat()) + ";"
            ligneToWrite += str(oneLigne[2]) + ";"
            ligneToWrite += str(oneLigne[3]) + "\n"
            myWrtiteFile.write(ligneToWrite)
        
        myWrtiteFile.close()
    except:
        print("Erreur lors de l'écriture du fichier csv")

    

if __name__=='__main__' :

    print("\n*** Projet PI2 - Equipe 46 ***")
    PATH_General = "Datas/"
    PATH_Compositon = PATH_General + "Composition.csv"
    data_Composition = Extraire_Dataset_Composition(PATH_Compositon)
    search_Date = datetime.date(2015, 7, 17) #17 juillet 2015
    data_Stocks_ID_Actifs = Divide_Dataset_Composition_Btw_Dates(data_Composition, search_Date)
    #print(data_Composition)
    print("\nTaille de data_Composition (tous les stocks du fichiers composition > " + str(len(data_Composition)))
    print("\nTaille de data_Stocks_ID_Actifs (les stocks présent dans l'indice à la date du : "+
            search_Date.isoformat() +") > ", str(len(data_Stocks_ID_Actifs)))
    
    
    PATH_Data1 = PATH_General+"data1.csv"
    PATH_Data2 = PATH_General+"data2.csv"
    PATH_Data3 = PATH_General+"data3.csv"
    PATH_Data4 = PATH_General+"data4.csv"
    PATH_Data5 = PATH_General+"data5.csv"
    all_Datas_Stocks = Extraire_Dataset_Data(PATH_Data1)
    all_Datas_Stocks.extend(Extraire_Dataset_Data(PATH_Data2))
    all_Datas_Stocks.extend(Extraire_Dataset_Data(PATH_Data3))
    all_Datas_Stocks.extend(Extraire_Dataset_Data(PATH_Data4))
    all_Datas_Stocks.extend(Extraire_Dataset_Data(PATH_Data5))

    #Liste contenant les ID+Date+Prix de cloture+Rendement des stocks actifs pour la date donnee.
    #Donc des stocks selectionnes dans 'data_Stocks_ID_Actifs'
    data_Stocks_Actifs = Extraire_Stocks_From_Data(all_Datas_Stocks, data_Stocks_ID_Actifs, search_Date)
    print("\nTaille de data_Stocks_Actifs (les stocks présent dans l'indice à la date du : "+
        search_Date.isoformat() +") > ", str(len(data_Stocks_Actifs)))

    indice_Prix_Journalier = Prix_Journalier_Indice(data_Stocks_Actifs)
    print("\n\n*Le prix journalier de l'indice en date du "+search_Date.isoformat() +" est de "+str(indice_Prix_Journalier))

    
    name_Csv = "Stocks_Actifs_One-Date_PI2_CSV.csv"
    Write_Stocks_inCSV(data_Stocks_Actifs, name_Csv)

    print("Fin d'execution du programme")

