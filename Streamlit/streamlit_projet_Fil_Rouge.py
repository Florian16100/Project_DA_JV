import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("vgsales_cleaned_franchise_random.csv")

st.title("Projet Data Analyse : Jeux Vidéo")
st.image("streamlit_image_jeu_video.jpeg")
st.write("L’industrie du jeu vidéo est une manne importante, riche en données à exploiter et où la concurrence est forte. Notre ambition à travers ce projet est de proposer une analyse des données du secteur, de reconnaître des corrélations et des disparités entre éditeurs, plateformes, distributeurs et de pouvoir élaborer un algorithme de machine-learning pouvant prédire le nombre de ventes d’un jeu vidéo.")
st.write("Les principaux objectifs de ce projet sont:")
st.write("1.	Exploration, visualisation et pre-processing  du jeu de données")
st.write("2.	Entraînement et évaluation des modèles de machine-learning pour la prédiction du nombre de ventes")

st.sidebar.title("Sommaire")
pages=["Exploration", "Data Visualisation", "Modélisation"]
page=st.sidebar.radio("Aller vers", pages)

if page == pages[0] : 
    st.write("### Exploration")
    st.write("L’équipe de DataScientest nous a dirigé vers un jeu de données disponible sur Kaggle, issu d'un scrap du site vgchartz.com. Le site répertorie les ventes totales de jeux-vidéos pour toutes les principales plateformes, allant des années 70 à aujourd’hui. Le site fournit également des variables additionnelles pour chaque jeu, tel que l’éditeur, le développeur ou bien encore le genre.")
    st.write("Le jeu de données en question était toutefois quelque peu limité. Le document ayant été créé en 2016, il ne pouvait pas inclure les ventes de jeux sortis plus récemment. La variable développeur manquait également au jeu de données. Nous avons donc fait un travail de recherche supplémentaire pour trouver un scrap plus complet.") 
    st.write("Nous avons jugé pertinent d’identifier la franchise à laquelle appartiennent les jeux de notre dataset. La franchise en elle-même est un argument marketing de poids et contribue grandement à la communication autour d’un jeu par les éditeurs. Grâce à cette méthode, nous avons ajouté la variable “Franchise” à notre jeu de données qui prend le nom de la franchise, si reconnue.")
    st.write("Finalement, afin de mieux comprendre les variables importantes de notre dataset, nous avons ajouté une colonne aléatoire contenant des chiffres random.")
    st.dataframe(df.head(10))
    st.write(df.shape)
    st.dataframe(df.describe())

    if st.checkbox("Afficher les NA") :
        st.dataframe(df.isna().sum())

if page == pages[1] : 
    st.write("### Data Visualisation")
    st.write("Pour la partie visualisation, nous nous sommes penchés sur les relations entre les variables, notamment le genre du jeu, les publishers, les plateformes et les années. La visualisation était particulièrement centrée sur la compréhension entre le nombre de jeux, les ventes totales par publisher mais aussi quels genres de jeux étaient le plus vendus afin d’avoir un aperçu des forces en présence.")
    
    st.write("Le premier graphique est l'évolution des ventes totales par année:")
    release_year = df['Year'].value_counts()
    x_values = release_year.values
    fig = plt.figure(figsize=(10,10))
    sns.lineplot(data = release_year)
    plt.xlabel("Years")
    plt.ylabel("Estimated_Sales")
    st.pyplot(fig)

    df_Estimated_Sales = df.groupby(['Year']).agg({'Estimated_Sales': 'sum'})


    
    st.write("Ce graphique montre les ventes des différents genres de jeu au fil du temps:")
    # Line plot for different sales categories over time
    fig = plt.figure(figsize=(7,7))
    sns.lineplot(data=df, x='Year', y='Genre', hue='Genre', palette='tab10')
    plt.xlabel('Year')
    plt.ylabel('Ventes (en millions)')
    plt.legend(bbox_to_anchor=(1.1,1), loc="upper left")
    st.pyplot(fig)


    st.write("Ici, il s'agit de comprendre quels types de plateforme reviennent le plus dans notre dataset:")
    fig = plt.figure(figsize=(10,10))
    df_platform2=df["Platform"].value_counts().head(20)
    sns.barplot(y=df_platform2.index, x=df_platform2.values);
    st.pyplot(fig)

    global_sales2 = df.groupby(['Publisher']).agg({'Estimated_Sales': 'sum'})
    global_games2 = df.groupby(['Publisher']).agg({'Name': 'count'})
    global_sales2 = global_sales2.join(global_games2).sort_values(
    by='Estimated_Sales', ascending=False).head(15)

    global_sales2["Estimated_Sales"].sort_values(ascending=False).head(10)
    st.write("Enfin, nous avons étudié respectivement la répartition des ventes globales pour les 10 plus grands publishers mondiaux et la répartition selon le nombre de jeux sortis par ces mêmes publishers")
    fig = plt.figure(figsize=(8,8))
    plt.pie(global_sales2.head(10).Estimated_Sales, labels=["Nintendo", "Activision", "Electronic Arts", "Sony", "EA Sports", "Ubisoft", "THQ", "Sega", "Rockstar Games", "Capcom"],
       colors=["#6c5f32","#f9f4ce","#9fc184","#97c8d9","pink","#432f0f","#a5a202","#0f68b8","#f7e560"], explode=[0.1,0.1,0.1,0,0,0,0,0,0,0],
        autopct=lambda x:round(x,2).astype(str)+"%", pctdistance=0.7, labeldistance=1.1)
    plt.title=('Répartition des ventes globales par publisher')
    plt.legend(bbox_to_anchor=(1.1,1), loc="upper left")
    plt.show()
    st.pyplot(fig)

    global_sales2.Name.sort_values(ascending=False).head(10)

    fig = plt.figure(figsize=(10,10))
    plt.pie(global_sales2.Name.sort_values(ascending=False).head(10), labels=["Activision", "Ubisoft", "EA", "Konami", "Nintendo", "THQ", "Sega", "Sony", "EA Sports", "Capcom"],
       colors=["#74e0aa","#bee893","#fbfeb2","#dbbf9e","#c4d9a9","#ddd8c4","#c5a5b8","#cccccc","#bdb3a6","#7c908a"], explode=[0.1,0.1,0.1,0,0,0,0,0,0,0],
        autopct=lambda x:round(x,2).astype(str)+"%", pctdistance=0.7, labeldistance=1.1)
    plt.title=('Répartition du nombre de jeux par éditeurs')
    plt.legend(bbox_to_anchor=(1,1.1), loc="upper left")
    plt.show()
    st.pyplot(fig)


    #fig = sns.catplot(x='Pclass', y='Survived', data=df, kind='point')
    #st.pyplot(fig)

    #fig = sns.lmplot(x='Age', y='Survived', hue="Pclass", data=df)
    #st.pyplot(fig)

if page == pages[2] : 
    st.write("### Modélisation")
    st.write("Notre problème de machine-learning s’apparente à une régression : prédire un chiffre de ventes à partir de variables catégorielles. Pour y répondre, nous avons entraîné et testé les modèles suivant, en comparant la performance de chacun sur la base du score de train et du score de test :")
    st.write("-	Régression Linéaire")
    st.write("-	Arbre de Régression")
    st.write("-	Random Forest")
    st.write("Les scores de la régression linéaire et de l'arbre de régression étant trop faibles, nous avons retenu le Random Forest comme modèle le plus adapté à notre objectif.")

    # Liste déroulante
    no_model = "Sélectionnez un modèle"
    model_1 = "Random Forest Regressor - Max Depth 2"
    model_2 = "Random Forest Regressor - Max Depth 4"
    model_3 = "Random Forest Regressor - No Max Depth"
    model_options = [no_model, model_1, model_2, model_3]
    selected_model = st.selectbox('Selection du modèle:', model_options)
    
    # Si sélection d'un modèle
    if selected_model != no_model:
        
        # Variables
        if selected_model == model_1:
            model_type = "Random Forest Regressor"
            model_depth = "2"
            model_loaded = joblib.load("vgsales_RandomForestReg_MaxDepth2.joblib")
        if selected_model == model_2:
            model_type = "Random Forest Regressor"
            model_depth = "4"
            model_loaded = joblib.load("vgsales_RandomForestReg_MaxDepth4.joblib")
        if selected_model == model_3:
            model_type = "Random Forest Regressor"
            model_depth = "Max"
            model_loaded = joblib.load("vgsales_RandomForestReg_NoMaxDepth.joblib")
        
        # Présentation du Modèle
        st.write('### Présentation du modèle')
        st.write('Type de Modèle:', model_type)
        st.write('Profondeur:', model_depth)
        
        # Checkbox
        st.write("### Options:")
        FeatImp_button_status = st.checkbox("Afficher Feature Importances")
        Xtest_button_status = st.checkbox("Charger un jeu de test et faire une prédiction")
        PersPred_button_status = st.checkbox("Faire une prédiction personnalisée")

        # Feature Importances Matrix
        if FeatImp_button_status == True:
            st.write('### Feature Importances Matrix')
            X_train_columns = joblib.load("X_train_columns.joblib")
            feature_importances = pd.DataFrame({'Variable' : X_train_columns, 'Importance' : model_loaded.feature_importances_}).sort_values('Importance', ascending = False)
            st.dataframe(feature_importances[feature_importances['Importance'] > 0.02])

        # Chargement du jeu de test
        if Xtest_button_status == True:
            X_test = joblib.load("vgsales_RandomForestReg_Xtest.joblib")
            # X_test = pd.read_csv("vgsales_RandomForestReg_Xtest.csv", index_col = 0)
            y_test = pd.read_csv("vgsales_RandomForestReg_Ytest.csv", index_col = 0)
            X_test_decoded = pd.read_csv("vgsales_RandomForestReg_XtestDecoded.csv", index_col = 0)
            st.write('### Présentation du jeu de test')
            st.write('Nombre de jeux listés:', X_test.shape[0])
            st.write("Extrait du jeu de données encodé:")
            st.dataframe(X_test.head(5))

            # Prédiction sur jeu de test
            pred_button_status = st.button("Faire une prédiction")
            
            if pred_button_status == True:
                st.write("Score de précision:", model_loaded.score(X_test, y_test))
                y_pred = model_loaded.predict(X_test)
                X_test_decoded['Estimated Sales - Predicted'] = y_pred
                X_test_decoded['Estimated Sales - Real'] = y_test.reset_index(drop = True)
                X_test_decoded['Squared Error'] = (X_test_decoded["Estimated Sales - Predicted"] - X_test_decoded["Estimated Sales - Real"]) ** 2
                
                col21, col22 = st.columns(2)

                with col21:
                    st.write("##### Top 100 des prédictions")
                    st.dataframe(X_test_decoded.nsmallest(100, 'Squared Error'))

                with col22:
                    st.write("##### Flop 100 des prédictions")
                    st.dataframe(X_test_decoded.nlargest(100, 'Squared Error'))

        # Faire une prédiction personnalisée
        if PersPred_button_status == True:

            # Définition des valeurs
            franchise_options = joblib.load("franchise_options.joblib")
            genre_options = joblib.load("genre_options.joblib")
            platform_options = joblib.load("platform_options.joblib")
            publisher_options = joblib.load("publisher_options.joblib")
            developer_options = joblib.load("developer_options.joblib")
            year_options = [year for year in range(1970, 2019)]

            input_franchise = st.selectbox("Franchise", franchise_options)
            input_genre = st.selectbox("Genre", genre_options)
            input_platform = st.selectbox("Platform", platform_options)
            input_publisher = st.selectbox("Publisher", publisher_options)
            input_developer = st.selectbox("Developer", developer_options)
            input_year = st.select_slider("Release Year", year_options)

            # Faire une prédiction
            perspred_button_status = st.button("Faire une prédiction")

            if perspred_button_status == True:
                vgsales_perso = pd.DataFrame({'Franchise' : [input_franchise],
                                              'Genre' : [input_genre],
                                              'Platform' : [input_platform],
                                              'Publisher' : [input_publisher],
                                              'Developer' : [input_developer],
                                              'Year' : [input_year]})
                
                # Encodage
                vgsales_perso_ohe = vgsales_perso[['Franchise', 'Genre', 'Platform', 'Publisher', 'Developer']]
                vgsales_perso_sc = np.asarray(vgsales_perso['Year']).reshape(-1,1)

                ohe = joblib.load("FitOneHotEncoder.joblib")
                sc = joblib.load("FitStandardScaler.joblib")

                st.write("##### Résumé")
                st.dataframe(vgsales_perso_ohe)
                vgsales_perso_sc = pd.DataFrame(sc.transform(vgsales_perso_sc), columns = ['Year'])
                vgsales_perso_ohe = pd.DataFrame(ohe.transform(vgsales_perso_ohe).toarray(), columns = ohe.get_feature_names_out())

                vgsales_perso_encoded = pd.concat([vgsales_perso_ohe, vgsales_perso_sc], ignore_index = False, axis = 1)

                # Prédiction
                y_perso_pred = int(np.round(model_loaded.predict(vgsales_perso_encoded) * 1000000)[0])
                y_perso_pred = "{:,.0f}".format(y_perso_pred)
                st.metric("Predicted Sales in 2019", y_perso_pred)