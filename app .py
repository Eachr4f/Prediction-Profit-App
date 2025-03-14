import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn import metrics
from sklearn import preprocessing
from sklearn import compose
from sklearn import impute
import category_encoders as ce

def format_number(num):
    if num >= 1_000_000:
        return f'{num / 1_000_000:.1f}M'
    elif num >= 1_000:
        return f'{num / 1_000:.1f}K'
    else:
        return f'{num:,.0f}'



# Tete d application
st.title("🤖 Machine learning App")
st.subheader("🔔 Projet d'application d analyse fichier csv")
st.info("L'utilisation de la régression linéaire multiple")

uploaded_file = st.file_uploader("Téléchargez votre fichier CSV", type=["csv"])

if uploaded_file is not None:
    tab1, tab2, tab3 = st.tabs(["📊 Visualization", "🧹 Cleaning", " 🎯Prediction"])

    df = pd.read_csv(uploaded_file)
    # Séparer les colonnes numériques et catégorielles
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    categorical_columns = df.select_dtypes(include=[object]).columns
    with tab1:
        with st.expander('Data filtered'):
            st.header("Choose Filters")

            filters = {}
            for column in categorical_columns:
                selected_values = st.multiselect(
                    f"Select {column}",
                    options=df[column].unique(),
                    default=df[column].unique()
                )
                filters[column] = selected_values

            # Sliders for numerical columns
            numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
            for column in numerical_columns:
                min_val = df[column].min()
                max_val = df[column].max()
                selected_range = st.slider(
                    f"Select {column}",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val)
                )
                filters[column] = selected_range

            st.header("Data Filtered")
            df1 = df.copy()
            for column, value in filters.items():
                if column in categorical_columns:
                    df1 = df1[df1[column].isin(value)]
                elif column in numerical_columns:
                    df1 = df1[(df1[column] >= value[0]) & (df1[column] <= value[1])]
            st.dataframe(df1)
            st.subheader("Dimensions du jeu de données filtrer")
            st.write(f"Nombre d'instances (lignes) : {df1.shape[0]}")
            st.write(f"Nombre d'attributs (colonnes) : {df1.shape[1]}")
        with st.expander('📊 Visualisation des données'):
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("Choisissez la colonne pour l'axe X", df.columns)
            with col2:
                y_axis = st.selectbox("Choisissez la colonne pour l'axe Y", df.columns)

            # Choix du type de graphique
            graph_type = st.selectbox(
                "Choisissez le type de graphique",
                ["Nuage de points", "Ligne", "Barre", "Histogramme"]
            )
            # Affichage du graphique
            if st.button("Générer le graphique"):
                st.subheader(f"Graphique : {graph_type}")
                if graph_type == "Nuage de points":
                    fig, ax = plt.subplots()
                    sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax)
                    st.pyplot(fig)

            elif graph_type == "Ligne":
                fig, ax = plt.subplots()
                sns.lineplot(data=df, x=x_axis, y=y_axis, ax=ax)
                st.pyplot(fig)

            elif graph_type == "Barre":
                fig, ax = plt.subplots()
                sns.barplot(data=df, x=x_axis, y=y_axis, ax=ax,orient='h')
                st.pyplot(fig)

            elif graph_type == "Histogramme":
                fig, ax = plt.subplots()
                sns.histplot(data=df, x=x_axis, kde=True, ax=ax)
                st.pyplot(fig)

            # Option pour utiliser Plotly (plus interactif)
            if graph_type == "Nuage de points":
                fig = px.scatter(df, x=x_axis, y=y_axis, title="Nuage de points avec Plotly")
                st.plotly_chart(fig)
            elif graph_type == "Ligne":
                fig = px.line(df, x=x_axis, y=y_axis, title="Graphique en ligne avec Plotly")
                st.plotly_chart(fig)
            elif graph_type == "Barre":
                fig = px.bar(df, x=x_axis, y=y_axis,orient='h', title="Graphique en barres avec Plotly")
                st.plotly_chart(fig)
            st.markdown("---") 
        Col_cibler = st.selectbox(
            "Select the target column (Y)",
            options=df.columns,
            index=df.shape[1]-1#Mettre le dernier column comme un valeur default
        )
        #ajouter les statitics de Profits
        if pd.api.types.is_numeric_dtype(df[Col_cibler]):
            total_profit=df[Col_cibler].sum()
            mean_cb=df[Col_cibler].mean()
            mode_cb=df[Col_cibler].mode().iloc[0]
            median_cb=df[Col_cibler].median()
            total_profit_formatted = format_number(total_profit)
            mean_cb_formatted = format_number(mean_cb)
            mode_cb_formatted = format_number(mode_cb)
            median_cb_formatted = format_number(median_cb)

            # Display in Streamlit
            total_1, total_2, total_3, total_4 = st.columns(4, gap='large')
            with total_1:
                st.info(f'Total {Col_cibler}', icon='📌')
                st.metric(label='Total', value=total_profit_formatted)
            with total_2:
                st.info(f'Mean {Col_cibler}', icon='📌')
                st.metric(label='Mean', value=mean_cb_formatted)
            with total_3:
                st.info(f'Mode {Col_cibler}', icon='📌')
                st.metric(label='Mode', value=mode_cb_formatted)
            with total_4:
                st.info(f'Median {Col_cibler}', icon='📌')
                st.metric(label='Median', value=median_cb_formatted)
        if not categorical_columns.empty:
            if pd.api.types.is_numeric_dtype(df[Col_cibler]):
                selected_categorical=st.selectbox("Choisissez une colonne catégorique :",
                categorical_columns)
                col_1, col_2 = st.columns(2)
                with col_1:
                    fig = px.pie(
                    df, names=selected_categorical, values=Col_cibler,
                    title=f"Répartition par {selected_categorical}"
                    )
                    st.plotly_chart(fig)
                with col_2:
                    grouped_data = df.groupby(selected_categorical)[Col_cibler].sum().reset_index()
                    grouped_data = grouped_data.sort_values(by=Col_cibler, ascending=False)
                    fig_bar = px.bar(
                    grouped_data,
                    x=Col_cibler, y=selected_categorical,
                    orientation='h',  # Horizontal
                    title=f"Bar Plot Horizontal pour {Col_cibler}"
                )       
                    st.plotly_chart(fig_bar)

            st.markdown("---") 

        Y = df[Col_cibler]  
        Xtest = df.drop(columns=[Col_cibler])  
        st.warning(f"Colonnes Variables : {', '.join(Xtest)}")
        st.warning(f"Colonne Cibler : {Y.name}")
        st.markdown("---") 
        #Afficher Les premiers ligne de Dataframe
        st.subheader("Preview sur le Data")
        st.write(df.head())
        st.markdown("---") 
        st.warning(f"Nombre total d'instances (lignes) : {df1.shape[0]}")
        st.warning(f"Nombre total d'attributs (colonnes) : {df1.shape[1]}")
        st.markdown("---") 
        # Affichage des colonnes contenant des valeurs manquantes
        st.subheader("Colonnes contenant des valeurs manquantes")
        missing_values = df.columns[df.isnull().any()]
        st.write(missing_values)
        st.markdown("---") 
        # Affichage du nombre de lignes contenant des valeurs manquantes
        st.subheader("Nombre de lignes contenant des valeurs manquantes")
        st.write(df.isnull().sum().sum())
        st.markdown("---") 
    #Sidebar pour Choisir (Nettoyage,Encodage,Pca)
    with st.sidebar:
        st.title('⚙ Choisir les Parametres ')
        st.image("C:/Users/AdMin/Desktop/Cours iddl/School project/application/logo-universite-abdelmalek-essaadi-tetouan-uae-7.png",caption="UAE FSTH", width=150)
        st.markdown("---") 
        st.subheader("Remplacement des valeurs manquantes")
        # Option pour le remplacement
        replace_option = st.selectbox(
            "Comment voulez-vous remplacer les valeurs manquantes ?",
            ["Moyenne", "Médiane", "Modalité fréquente (catégorielles)", "Suppression","Rien"]
        )
        if replace_option == "Moyenne":
            st.info("Remplacement des valeurs manquantes par la moyenne.")
            df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
        elif replace_option == "Médiane":
            st.info("Remplacement des valeurs manquantes par la médiane.")
            df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        elif replace_option == "Modalité fréquente (catégorielles)":
            st.info("Remplacement des valeurs manquantes par la modalité la plus fréquente.")
            for col in categorical_columns:
                df[col] = df[col].fillna(df[col].mode()[0])
        elif replace_option == "Suppression":
            st.info("Suppression des lignes contenant des valeurs manquantes.")
            df.dropna(inplace=True)
        elif replace_option == "Suppression":
            st.info("Rien est changer")
        st.markdown("---") 

        st.subheader("Encodage des colonnes catégorielles")
        encoding_option = st.selectbox(
            "Choisissez une méthode d'encodage",
            ["One-Hot Encoding", "Binary Encoding", "Label Encoding"],
            index=2
        )
        Y=df[Col_cibler]
        X=df.drop(columns=[Col_cibler])
        # Appliquer l'encodage en fonction de l'option choisie
        if encoding_option == "One-Hot Encoding":
            st.info("Application de l'encodage One-Hot avec ColumnTransformer.")
            # Utiliser ColumnTransformer pour One-Hot Encoding
            ct = compose.ColumnTransformer(
                transformers=[('encoder', OneHotEncoder(), categorical_columns)],
                remainder='passthrough'
            )   
            X = ct.fit_transform(X)
            # Convertir le résultat en DataFrame pour l'affichage
            X = pd.DataFrame(X, columns=ct.get_feature_names_out())
        elif encoding_option == "Binary Encoding":
            st.info("Application de l'encodage Binaire.")
            ct=compose.ColumnTransformer(transformers=[('encoder',ce.BinaryEncoder(),
                                          categorical_columns)],remainder='passthrough')
            X=ct.fit_transform(X)
            X = pd.DataFrame(X, columns=ct.get_feature_names_out())   
        elif encoding_option == "Label Encoding":
            st.info("Application de l'encodage Label.")
            label_encoder = LabelEncoder()
            for col in categorical_columns:
                X[col] = label_encoder.fit_transform(X[col])
            X = X
        X1=X.copy()
        st.subheader("Normalisation des colonnes numériques")
        normalization_option = st.selectbox(
            "Choisissez une méthode de normalisation",
            ["Pas de normalisation", "MinMaxScaler", "StandardScaler", "MaxAbsScaler", "RobustScaler"],
            index=1
        )
        numeric_columns_X = X.select_dtypes(include=[np.number]).columns.tolist()
        # Appliquer la normalisation en fonction de l'option choisie
        if normalization_option == "MinMaxScaler":
            st.info("Application de MinMaxScaler.")
            scaler = preprocessing.MinMaxScaler()
            X1[numeric_columns_X] = scaler.fit_transform(X1[numeric_columns_X])
        elif normalization_option == "StandardScaler":
            st.info("Application de StandardScaler.")
            scaler = preprocessing.StandardScaler()
            X1[numeric_columns_X] = scaler.fit_transform(X1[numeric_columns_X])
        elif normalization_option == "MaxAbsScaler":
            st.info("Application de MaxAbsScaler.")
            scaler = preprocessing.MaxAbsScaler()
            X1[numeric_columns_X] = scaler.fit_transform(X1[numeric_columns_X])
        elif normalization_option == "RobustScaler":
            st.info("Application de RobustScaler.")
            scaler = preprocessing.RobustScaler()
            X1[numeric_columns_X] = scaler.fit_transform(X1[numeric_columns_X])
        st.markdown("---") 
        st.subheader("Réduction de dimension avec PCA")
        pca_option = st.radio(
            "Souhaitez-vous appliquer une PCA ?",
            ("Oui", "Non")
        )
        if pca_option == "Oui":
            n_components = st.slider(
                "Nombre de composantes principales",
                min_value=1,
                max_value=X1.shape[1],
                value=2
            )
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X1)
            df_pca = pd.DataFrame(X_pca)
            df_pca[Y.name] = df[Y.name]
            st.info(f"Réduction de dimension appliquée avec {n_components} composantes principales.")
        else:
            st.info("Aucune réduction de dimension appliquée.")


        st.markdown("---")  # Add a horizontal line for separation
        st.write("Traitement des données terminé avec succès ! ✅")
    with tab2:
        st.subheader("DataFrame après traitement des valeurs manquantes")
        st.dataframe(df)
        st.markdown("---") 
        # Afficher les statistiques des valeurs manquantes après traitement
        st.subheader("Statistiques des valeurs manquantes après traitement")
        st.write(df.isnull().sum())
        st.markdown("---") 
        st.subheader("Les Variables apres Encodage")
        st.dataframe(X)
        st.markdown("---") 
        st.subheader("Les Variables apres Normalization")
        st.dataframe(X1)


    # Visualisation de la matrice de corrélation
        st.subheader("Matrice de corrélation")
        corr_matrix = X[numeric_columns_X].corr()
        st.write("Matrice de corrélation avec Plotly")
        fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r')
        st.plotly_chart(fig)
    with tab3:
        if pca_option == "Oui":
            st.subheader("Données après réduction de dimension (PCA) :")
            st.write(df_pca.head())
            st.subheader("Visualisation des deux premières composantes principales")
        st.markdown("---") 
        test_size = st.slider("Choisissez le pourcentage de l'ensemble de test", 0.1, 0.9, 0.2)
        st.subheader("Séparation des données")
        if pca_option == "Oui":
            X_train, X_test, y_train, y_test = train_test_split(df_pca.iloc[:,:-1], Y, test_size=test_size, random_state=42)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X1, Y, test_size=test_size, random_state=42)
        st.write(f"Ensemble d'entraînement : {len(X_train)} lignes")
        st.write(f"Ensemble de test : {len(X_test)} lignes")
        st.markdown("---") 
        # 14. Entraînement du modèle avec gestion de l'état
        if "model" not in st.session_state:
            st.session_state.model = LinearRegression()

        # 14.1 Bouton pour lancer l'entraînement du modèle
        st.subheader("Entraînement du modèle")
        if st.button("Entraîner le modèle"):
            st.session_state.model.fit(X_train, y_train)
            st.write("Modèle entraîné avec succès!")

        # 15.1 Bouton pour tester le modèle sans rafraîchir
        st.subheader("Tester le modèle")
        if st.button("Tester le modèle"):
            y_pred = st.session_state.model.predict(X_test)
            st.write("MSE",metrics.mean_squared_error(y_test,y_pred))
            st.write("MAE",metrics.mean_absolute_error(y_test,y_pred))
            st.write("R2 score",metrics.r2_score(y_test,y_pred))
            st.write("MedAe",metrics.median_absolute_error(y_test,y_pred))
            st.write("Explainedvariance score",metrics.explained_variance_score(y_test,y_pred))
            st.write("RMSE",np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
            # 16. Affichage d'un graphe montrant les valeurs réelles et les prédites
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, color='blue', label='Prédictions')
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Ligne idéale')
            ax.set_xlabel('Valeurs de test')
            ax.set_ylabel('Valeurs prédites')
            ax.set_title('Test vs Prédictions')
            ax.legend()
            st.pyplot(fig)
        st.markdown("---") 

        st.subheader("Prédiction avec de nouvelles données")
        new_data = {}

        for col in X.columns:
            if col in categorical_columns:
                unique_values = df[col].unique()
                selected_value = st.selectbox(f"Sélectionnez une valeur pour {col}", unique_values)
        
                # Handle encoding based on the selected encoding method
                if encoding_option == "One-Hot Encoding":
                    # For One-Hot Encoding, create binary columns for each unique value
                    for value in unique_values:
                        new_data[f"{col}_{value}"] = 1 if selected_value == value else 0
                elif encoding_option == "Binary Encoding":
                    # For Binary Encoding, transform the selected value using BinaryEncoder
                    encoder = ce.BinaryEncoder(cols=[col])
                    encoded_value = encoder.transform(pd.DataFrame({col: [selected_value]}))
                    for encoded_col in encoded_value.columns:
                        new_data[encoded_col] = encoded_value[encoded_col].values[0]
                elif encoding_option == "Label Encoding":
                    # For Label Encoding, transform the selected value using LabelEncoder
                    label_encoder = LabelEncoder()
                    label_encoder.fit(df[col])
                    new_data[col] = label_encoder.transform([selected_value])[0]
            else:
                # For numerical columns, allow the user to input a value
                new_data[col] = st.number_input(f"Entrez la valeur pour {col}")

        # Create a DataFrame with the new data
        new_df = pd.DataFrame(new_data, index=[0])

        # Make a prediction
        if st.button("Prédire le profit"):
            if pca_option == "Oui":
                # If PCA was applied, transform the new data using the same PCA
                new_df_pca = pca.transform(new_df)
                prediction = st.session_state.model.predict(new_df_pca)
            else:
                # If no PCA was applied, predict directly
                prediction = st.session_state.model.predict(new_df)
            st.success(f"✅ {Y.name} prédit pour ces données est :  {format_number(prediction[0])}")
else:
    st.error("Please upload a CSV file to proceed.")
