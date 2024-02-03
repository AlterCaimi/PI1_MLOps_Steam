from fastapi import FastAPI
import pandas as pd 
import warnings
warnings.filterwarnings('ignore')
from recommender_item_item import item_item_recom

app = FastAPI()

#http://127.0.0.1:8000

#-----------------------------------------INICIO-------------------------------------------#
@app.get("/")
def index():
    return 'API desarrollada para el PI1 MLOps por Alter Caimi'

df_steam = pd.read_parquet('../CleanData/steam_games.parquet')
df_user = pd.read_parquet('../CleanData/users_items.parquet')
df_reviews = pd.read_parquet('../CleanData/reviews.parquet')
df_sent = pd.read_parquet('../sentiment_analysis_2.parquet')

#-----------------------------------------ENDPOINT 1---------------------------------------#
@app.get('/developer/{desarrollador}')
def developer(desarrollador: str):
    
    """
    Recupera información sobre un desarrollador específico.

    Args:
        desarrollador (str): Nombre del desarrollador.

    Returns:
        pd.DataFrame: DataFrame con información sobre el desarrollador.
    """
    
    if not isinstance(desarrollador, str):
        return {'Mensaje': 'El argumento "desarrollador" debe ser una cadena de texto (str).'}
    
    df_steam['free'] = df_steam['price'].apply(lambda x: 1 if x == 0 else 0)
    df_steam['Año'] = df_steam['release_date'].dt.year
    
    desarrolladores = df_steam['developer'].str.lower().unique()
    desarrollador = desarrollador.lower()

    if desarrollador not in desarrolladores:
        return {'Mensaje': 'Desarrollador no encontrado. Inserte un desarrollador válido'}

    df = df_steam.groupby(['Año', 'developer']).agg(
                            {'id': 'count', 'free': lambda x: (x.sum() / x.count()) * 100}
                        ).reset_index().rename(columns={'free': 'Contenido Free', 'id': 'Cantidad de Items'})
    
    df['developer'] = df['developer'].apply(lambda x: x.lower())
    
    df = df[df['developer'] == desarrollador].drop(columns= 'developer').sort_values(by= 'Año', ascending=False)
    df['Contenido Free'] = round(df['Contenido Free'], 2)
    df['Contenido Free'] = df['Contenido Free'].apply(lambda x: str(x) + '%')
    df.reset_index(drop=True, inplace=True)

    resultado = {
        f'Año {int(df.loc[i,"Año"])}': {'Cantidad de Items': int(df.loc[i, 'Cantidad de Items']), 'Contenido Free:': df.loc[i, 'Contenido Free']}
        for i in range(len(df))
    }

    return resultado

#-----------------------------------------ENDPOINT 2---------------------------------------#
@app.get('/userdata/{user_id}')
def userdata(user_id: str):

    if not isinstance(user_id, str):
        return {'Mensaje': 'El argumento user_id debe ser una cadena de texto.'}

    df_user_n = df_user[['user_id', 'item_id', 'items_count']]
    
    usuarios = df_user_n['user_id'].unique()

    if user_id not in usuarios:
        return {'Mensaje': 'Usuario no encontrado. Por favor ingrese un usuario válido'}
    
    df_steam_n = df_steam[['id', 'price']]
    df_reviews_n = df_reviews[['user_id', 'recommend']]
    
    df_reviews_n = df_reviews_n.groupby('user_id').agg('sum').reset_index()

    df = df_user_n.merge(df_steam_n, how='left', left_on= 'item_id', right_on= 'id')
    df = df.merge(df_reviews_n, how = 'left')
    df = df.drop(columns=['item_id', 'id'])
    df = df.groupby('user_id').agg('max').reset_index()
    df = df[df['user_id'] == user_id]

    df['recommend'] = df['recommend'].fillna(0)
    df['recommend'] = round(df['recommend'] / df['items_count'], 2)
    df['recommend'] = df['recommend'].apply(lambda x: str(x) + ' %')
    df.rename(columns= {'recommend': '% de recomendación'}, inplace=True)
    
    df.reset_index(inplace=True, drop=True)

    resultado = {
        'Usuario': df.loc[0, 'user_id'],
        'Dinero gastado': str(df.loc[0, 'price']) + 'USD',
        '% de recomendación': df.loc[0, '% de recomendación'],
        'Cantidad de Items': int(df.loc[0, 'items_count'])
    }
    return resultado

#-----------------------------------------ENDPOINT 3---------------------------------------#
@app.get('/UserForGenre/{genero}')
def UserForGenre(genero: str):
    
    if not isinstance(genero, str):
        return {'Mensaje': 'El género ingresado debe ser una cadena de texto (string)'}

    genero = 'genre_' + genero
    genero = genero.lower()

    df_steam_n = df_steam.rename(columns= lambda x: x.lower())

    columnas = list(df_steam_n.columns)

    if genero not in columnas:
        generos = [col.replace('genre_', '') for col in columnas if 'genre_' in col]
        return {'Mensaje': 'Género no encontrado. Ingrese un género válido',
                'Géneros disponibles': generos}

    df_user_n = df_user[['user_id', 'item_id', 'playtime_forever']]

    df_steam_n = df_steam_n[df_steam_n[genero] == 1][['id','release_date']]
    df_steam_n['Año'] = df_steam_n['release_date'].dt.year
    df_steam_n.drop(columns= 'release_date', inplace= True)

    df = df_user_n.merge(df_steam_n, how= 'left', left_on= 'item_id', right_on= 'id')
    df = df.dropna()
    df = df.groupby(['user_id', 'Año']).agg({'playtime_forever': 'sum'}).reset_index()
    df['playtime_forever'] = round(df['playtime_forever']/60, 2)

    df_1 = df.groupby('user_id').agg({'playtime_forever' : 'sum'}).reset_index()
    ind = df_1['playtime_forever'].idxmax()
    usuario = df_1.loc[ind, 'user_id']
    df = df[df['user_id'] == usuario]
    df.rename(columns= {'playtime_forever': 'Horas jugadas'}, inplace=True)

    df.reset_index(inplace=True, drop=True)
    
    resultado = {
        f'Usuario con mas horas jugadas para el género {genero.replace("genre_", "")}:': df.loc[0,'user_id'],
        'Horas jugadas:': [{'Año:': int(df.loc[i,'Año']), 'Horas:': int(df.loc[i,'Horas jugadas'])} for i in range(len(df))]
    }

    return resultado

#-----------------------------------------ENDPOINT 4---------------------------------------#
@app.get('/best_developer_year/{anio}')
def best_developer_year(anio: int):

    try:
        anio = int(anio)
    except Exception as e:
        return {f'Error {e}': 'Debe insertar un número entero.'}
    
    df_sent_n = df_sent[['item_id', 'sentiment_analysis_2', 'recommend', 'Año']]

    anios = list(df_sent['Año'].unique())

    if anio not in anios:
        anios = [int(x) for x in anios]
        return {'Mensaje': f'No hay registros del año {anio}',
                'Los años disponibles son:': anios}
    
    df_steam_n = df_steam[['id', 'developer']]

    df = df_sent_n.merge(df_steam_n, how='left', left_on='item_id', right_on='id')
    df.drop(columns=['item_id', 'id'], inplace=True)
    df.rename(columns= {'sentiment_analysis_2': 'rating'}, inplace=True)

    df = df.groupby(['developer', 'Año']).agg({'rating': (lambda x: (x == 2).sum()), 'recommend': 'sum'}).reset_index()
    df = df[df['Año'] == anio]
    
    df['puntaje'] = df['rating'] + df['recommend']

    df.sort_values(by= 'puntaje', ascending=False, inplace=True)
    df.reset_index(inplace=True, drop= True)
    
    result = {
                'Puesto 1': df.loc[0, 'developer'],
                'Puesto 2': df.loc[1, 'developer'],
                'Puesto 3': df.loc[2, 'developer']
            }

    return result

#-----------------------------------------ENDPOINT 5---------------------------------------#
@app.get('/developer_reviews_analysis/{desarrolladora}')
def developer_reviews_analysis(desarrolladora: str):

    if not isinstance(desarrolladora, str):
        return {'Mensaje': 'Debe ingresar una cadena de texto'}
    
    desarrolladora = desarrolladora.lower()
    
    df_steam_n = df_steam[['id', 'developer']]
    df_steam_n['developer'] = df_steam_n['developer'].apply(lambda x: x.lower())

    developers = list(df_steam_n['developer'].unique())
    
    if desarrolladora not in developers:
        return {'Mensaje': 'Desarrolladora no encontrada. Por favor ingrese una desarrolladora válida.',
                'Desarrolladoras disponibles': developers}
    
    df_sent_n = df_sent[['item_id', 'sentiment_analysis_2']]

    df = df_sent_n.merge(df_steam_n,how= 'left', left_on= 'item_id', right_on='id')
    df.drop(columns=['id', 'item_id'], inplace=True)

    resultado = df[df['developer'] == desarrolladora]['sentiment_analysis_2'].value_counts()

    resultado_dic = {
        desarrolladora: [f'Negative = {resultado[0]}', f'Positive = {resultado[2]}']
    }

    return resultado_dic

#-----------------------------------------ENDPOINT 6---------------------------------------#
@app.get('/recomendacion_juego/{id_producto}')
def recomendacion_juego(id_producto: int, n_recom: int = 5, umbral: float = 0.999):
    return item_item_recom(id_producto, n_recom, umbral)