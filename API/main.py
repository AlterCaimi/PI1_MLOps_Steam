from fastapi import FastAPI
import pandas as pd 
from recommender_item_item import item_item_recom

app = FastAPI()

#http://127.0.0.1:8000

#-----------------------------------------INICIO-------------------------------------------#
@app.get("/")
def index():
    return 'API desarrollada para el PI1 MLOps por Alter Caimi'

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
    
    df_steam = pd.read_parquet('../CleanData/steam_games.parquet', columns= ['id', 'developer', 'release_date', 'price'])

    df_steam['developer'] = df_steam['developer'].apply(lambda x: x.lower())
    desarrolladores = df_steam['developer'].unique()
    desarrollador = desarrollador.lower()

    if desarrollador not in desarrolladores:
        del df_steam
        return {'Mensaje': f'Desarrollador no encontrado. Inserte un desarrollador válido. Desarrolladores: {desarrolladores}'}
    
    df_steam = df_steam[df_steam['developer'] == desarrollador]
    
    df_steam['free'] = df_steam['price'].apply(lambda x: 1 if x == 0 else 0)
    df_steam['Año'] = df_steam['release_date'].dt.year

    df = df_steam.groupby(['Año', 'developer']).agg(
                            {'id': 'count', 'free': lambda x: (x.sum() / x.count()) * 100}
                        ).reset_index().rename(columns={'free': 'Contenido Free', 'id': 'Cantidad de Items'})
    del df_steam
    df['developer'] = df['developer'].apply(lambda x: x.lower())
    
    df = df[df['developer'] == desarrollador].drop(columns= 'developer').sort_values(by= 'Año', ascending=False)
    df['Contenido Free'] = round(df['Contenido Free'], 2)
    df['Contenido Free'] = df['Contenido Free'].apply(lambda x: str(x) + '%')
    df.reset_index(drop=True, inplace=True)

    resultado = {
        f'Año {int(df.loc[i,"Año"])}': {'Cantidad de Items': int(df.loc[i, 'Cantidad de Items']), 'Contenido Free:': df.loc[i, 'Contenido Free']}
        for i in range(len(df))
    }

    del df
    return resultado

#-----------------------------------------ENDPOINT 2---------------------------------------#
@app.get('/userdata/{user_id}')
def userdata(user_id: str):

    if not isinstance(user_id, str):
        return {'Mensaje': 'El argumento user_id debe ser una cadena de texto.'}

    df = pd.read_parquet('../CleanData/userdata.parquet')
    df= df[df['user_id'].isin([user_id])].reset_index(drop=True)
    
    if df.empty:
        del df
        return {'Mensaje': 'Usuario no encontrado. Por favor ingrese un usuario válido'}
    
    df = df.reset_index(drop=True)

    resultado = {
        'Usuario': user_id,
        'Dinero gastado': str(round(df.loc[0,'dinero_gastado'], 2)) + ' USD',
        '% de recomendación': str(df.loc[0,'porcentaje_recom']) + ' %',
        'Cantidad de Items': int(df.loc[0,'items_count'])
    }
    del df
    return resultado

#-----------------------------------------ENDPOINT 3---------------------------------------#
@app.get('/UserForGenre/{genero}')
def UserForGenre(genero: str):
    
    if not isinstance(genero, str):
        return {'Mensaje': 'El género ingresado debe ser una cadena de texto (string)'}
    
    genero = genero.lower()

    try:
        df = pd.read_parquet('../CleanData/userforgenre2.parquet')
        df = df[df['genres'].isin(['action'])].drop(columns='genres')
    except Exception:
        return {'Error': 'Género no encontrado. Ingrese un género válido'}

    usuario_max_horas = df.groupby('user_id').agg({'playtime_forever': 'sum'}).sort_values('playtime_forever').tail(1).index[0]
    df = df[df['user_id'].isin([usuario_max_horas])].reset_index(drop=True)

    resultado = {
        f'Usuario con mas horas jugadas para el género {genero}:': usuario_max_horas,
        'Horas jugadas:': [{'Año:': int(df.loc[i,'Año']), 'Horas:': float(df.loc[i,'playtime_forever'])} for i in range(len(df))]
    }
    del df
    return resultado

#-----------------------------------------ENDPOINT 4---------------------------------------#
@app.get('/best_developer_year/{anio}')
def best_developer_year(anio: int):

    try:
        anio = int(anio)
    except Exception as e:
        return {f'Error {e}': 'Debe insertar un número entero.'}
    
    df_sent = pd.read_parquet('../sentiment_analysis_2.parquet', columns= ['item_id', 'sentiment_analysis_2', 'recommend', 'Año'])
    anios = list(df_sent['Año'].unique())
    anios = [int(x) for x in anios]
    df_sent = df_sent[df_sent['Año'] == anio]
    if df_sent.empty:
        del df_sent
        return {'Mensaje': f'No hay registros del año {anio}',
                'Los años disponibles son:': anios}

    df_steam = pd.read_parquet('../CleanData/steam_games.parquet', columns= ['id', 'developer'])

    df = df_sent.merge(df_steam, how='left', left_on='item_id', right_on='id')

    del df_sent
    del df_steam

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
    del df
    return result

#-----------------------------------------ENDPOINT 5---------------------------------------#
@app.get('/developer_reviews_analysis/{desarrolladora}')
def developer_reviews_analysis(desarrolladora: str):

    if not isinstance(desarrolladora, str):
        return {'Mensaje': 'Debe ingresar una cadena de texto'}
    
    desarrolladora = desarrolladora.lower()
    
    df_steam = pd.read_parquet('../CleanData/steam_games.parquet', columns= ['id', 'developer'])
    df_steam['developer'] = df_steam['developer'].apply(lambda x: x.lower())
    developers = list(df_steam['developer'].unique()) 

    df_steam = df_steam[df_steam['developer'] == desarrolladora]
    if df_steam.empty:
        del df_steam
        return {f'Desarrolladora no encontrada: {desarrolladora}.': f'Desarrolladoras disponibles {", ".join(developers)}'}
    
    df_sent = pd.read_parquet('../sentiment_analysis_2.parquet', columns= ['item_id', 'sentiment_analysis_2'])

    df = df_sent.merge(df_steam,how= 'left', left_on= 'item_id', right_on='id')

    del df_sent
    del df_steam

    df.drop(columns=['id', 'item_id'], inplace=True)

    resultado = df[df['developer'] == desarrolladora]['sentiment_analysis_2'].value_counts()

    del df

    resultado_dic = {
        desarrolladora: [f'Negative = {resultado[0]}', f'Positive = {resultado[2]}']
    }

    return resultado_dic

#-----------------------------------------ENDPOINT 6---------------------------------------#
@app.get('/recomendacion_juego/{id_producto}')
def recomendacion_juego(id_producto: int, n_recom: int = 5, umbral: float = 0.999):
    return item_item_recom(id_producto, n_recom, umbral)