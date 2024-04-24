import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("vgsales_cleaned_franchise_random.csv")

st.title("Projet fil rouge")
st.sidebar.title("Sommaire")
pages=["Exploration", "DataVisualization", "Modélisation"]
page=st.sidebar.radio("Aller vers", pages)

if page == pages[0] : 
    st.write("### Introduction")
    st.dataframe(df.head(10))
    st.write(df.shape)
    st.dataframe(df.describe())

    if st.checkbox("Afficher les NA") :
        st.dataframe(df.isna().sum())

if page == pages[1] : 
    st.write("### DataVisualization")

    
    release_year = df['Year'].value_counts()
    x_values = release_year.values
    fig = plt.figure()
    sns.lineplot(data = release_year)
    plt.xlabel("Years")
    plt.ylabel("Estimated_Sales")
    st.pyplot(fig)

    df_Estimated_Sales = df.groupby(['Year']).agg({'Estimated_Sales': 'sum'})


    
    
    # Line plot for different sales categories over time
    fig = plt.figure()
    sns.lineplot(data=df, x='Year', y='Genre', hue='Genre', palette='tab10')
    plt.xlabel('Year')
    plt.ylabel('Ventes (en millions)')
    st.pyplot(fig)



    fig = plt.figure()
    df_platform2=df["Platform"].value_counts().head(20)
    sns.barplot(y=df_platform2.index, x=df_platform2.values);
    st.pyplot(fig)

    global_sales2 = df.groupby(['Publisher']).agg({'Estimated_Sales': 'sum'})
    global_games2 = df.groupby(['Publisher']).agg({'Name': 'count'})
    global_sales2 = global_sales2.join(global_games2).sort_values(
    by='Estimated_Sales', ascending=False).head(15)

    global_sales2["Estimated_Sales"].sort_values(ascending=False).head(10)

    fig = plt.figure()
    plt.pie(global_sales2.head(10).Estimated_Sales, labels=["Nintendo", "Activision", "Electronic Arts", "Sony", "EA Sports", "Ubisoft", "THQ", "Sega", "Rockstar Games", "Capcom"],
       colors=["#6c5f32","#f9f4ce","#9fc184","#97c8d9","pink","#432f0f","#a5a202","#0f68b8","#f7e560"], explode=[0.1,0.1,0.1,0,0,0,0,0,0,0],
        autopct=lambda x:round(x,2).astype(str)+"%", pctdistance=0.7, labeldistance=1.1)
    plt.title=('Répartition des ventes globales par publisher')
    plt.legend(bbox_to_anchor=(1.1,1), loc="upper left")
    plt.show()
    st.pyplot(fig)

    global_sales2.Name.sort_values(ascending=False).head(10)

    fig = plt.figure()
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

    df = df.drop(['Rank','Name','basename',	'Genre','Platform',	'Publisher','Developer','Year'], axis=1)
    y = df['Estimated_Sales']
    #X_cat = df[['Pclass', 'Sex',  'Embarked']]
    #X_num = df[['Age', 'Fare', 'SibSp', 'Parch']]

    #for col in X_cat.columns:
        #X_cat[col] = X_cat[col].fillna(X_cat[col].mode()[0])
    #for col in X_num.columns:
        #X_num[col] = X_num[col].fillna(X_num[col].median())
    #X_cat_scaled = pd.get_dummies(X_cat, columns=X_cat.columns)
    #X = pd.concat([X_cat_scaled, X_num], axis = 1)

    #from sklearn.model_selection import train_test_split
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    #from sklearn.preprocessing import StandardScaler
    #scaler = StandardScaler()
    #X_train[X_num.columns] = scaler.fit_transform(X_train[X_num.columns])
    #X_test[X_num.columns] = scaler.transform(X_test[X_num.columns])
                                            
    #def prediction(classifier):
        #if classifier == 'Random Forest':
           #clf = RandomForestClassifier()
        #elif classifier == 'SVC':
             #clf = SVC()
        #elif classifier == 'Logistic Regression':
             #clf = LogisticRegression()
        #clf.fit(X_train, y_train)
        #return clf

    #def scores(clf, choice):
        #if choice == 'Accuracy':
            #return clf.score(X_test, y_test)
        #elif choice == 'Confusion matrix':
            #return confusion_matrix(y_test, clf.predict(X_test))

    #choix = ['Random Forest', 'SVC', 'Logistic Regression']
    #option = st.selectbox('Choix du modèle', choix)
    #st.write('Le modèle choisi est :', option)

    #clf = prediction(option)

    #display = st.radio('Que souhaitez-vous montrer ?', ('Accuracy', 'Confusion matrix'))
    #if display == 'Accuracy':
        #st.write(scores(clf, display))
    #elif display == 'Confusion matrix':
        #st.dataframe(scores(clf, display))
