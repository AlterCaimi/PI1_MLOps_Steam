from fastapi import FastAPI
import pandas as pd 
from recommender_item_item import item_item_recom

app = FastAPI()

#http://127.0.0.1:8000

#-----------------------------------------INICIO-------------------------------------------#

@app.get("/")
def index():                                                    # mensaje de inicio de la API
    return 'API desarrollada para el PI1 MLOps por Alter Caimi'

#-----------------------------------------ENDPOINT 1---------------------------------------#

@app.get('/developer/{desarrollador}')
def developer(desarrollador: str):
    
    """
    Recupera información sobre un desarrollador específico.

    Args:
        desarrollador (str): Nombre del desarrollador.

    Returns:
        dict: Diccionario con información sobre el desarrollador.
    """
    
    if not isinstance(desarrollador, str):
        return {'Mensaje': 'El argumento "desarrollador" debe ser una cadena de texto (str).'}
    
    df = pd.read_parquet('../CleanData/developer.parquet')      # cargamos dataframe

    desarrollador = desarrollador.lower()                       # pasamos desarrollador ingresado a minúsculas
    desarrolladores = list(df['developer'].unique())            # obtenemos lista de todos los desarrolladores
    df = df[df['developer'].isin([desarrollador])].reset_index(drop=True)       # filtramos dataframe dejando info del desarrolador

    if len(df) == 0:                                            # si el desarrollador está mal ingresado o no existe
        del df
        return {'Mensaje': f'Desarrollador no encontrado. Inserte un desarrollador válido. Desarrolladores: {", ".join(desarrolladores)}'}

    resultado = {                                               # armamos diccionario con los resultados
        f'Año {int(df.loc[i,"Año"])}': {'Cantidad de Items': int(df.loc[i, 'Cantidad de Items']), 'Contenido Free:': str(df.loc[i, 'Contenido Free']) + ' %'}
        for i in range(len(df))
    }

    del df
    return resultado

#-----------------------------------------ENDPOINT 2---------------------------------------#

@app.get('/userdata/{user_id}')
def userdata(user_id: str):

    '''
    Recupera información de un usuario específico.

    Args:
        user_id (str): id del usuario
    
    Returns:
        dict: Diccionario con información del usuario
    
    '''

    if not isinstance(user_id, str):
        return {'Mensaje': 'El argumento user_id debe ser una cadena de texto.'}

    df = pd.read_parquet('../CleanData/userdata.parquet')               # cargamos dataframe desde archivo
    df= df[df['user_id'].isin([user_id])].reset_index(drop=True)        # filtramos dataframe por usuario ingresado
    
    if df.empty:                # si el usuario ingresado está mal o no existe
        del df
        return {'Mensaje': 'Usuario no encontrado. Por favor ingrese un usuario válido'}
    
    df = df.reset_index(drop=True)                                      # reset index luego de filtrar

    resultado = {                                                       # armamos diccionario con los resultados
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

    '''
    Recupera información de un género específico.

    Args:
        género (str): género de videojuego
    
    Returns:
        dict: Diccionario con información del género
    
    '''

    if not isinstance(genero, str):
        return {'Mensaje': 'El género ingresado debe ser una cadena de texto (string)'}
    
    genero = genero.lower()

    try:
        df = pd.read_parquet('../CleanData/userforgenre.parquet')
        df = df[df['genres'].isin([genero])].drop(columns='genres')
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

    '''
    Recupera información de un año específico.

    Args:
        anio (int): Año a considerar
    
    Returns:
        dict: Diccionario con el top 3 de desasrrolladores para ese año
    
    '''

    try:
        anio = int(anio)
    except Exception as e:
        return {f'Error {e}': 'Debe insertar un número entero.'}
    
    df = pd.read_parquet('../CleanData/best_developer_year.parquet')
    anios = list(df['Año'].unique())
    anios = [int(x) for x in anios]
    
    df = df[df['Año'].isin([anio])]
    if len(df) == 0:
        del df
        return {'Mensaje': f'No hay registros del año {anio}',
                'Los años disponibles son:': anios}

    df = df.sort_values('rating', ascending=False).reset_index()
    
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

    '''
    Recupera información de un desarrollador específico.

    Args:
        desarrolladora (str): Desarrolladora a considerar.
    
    Returns:
        dict: Diccionario con la cantidad de reviews positivos y negativos para esa desarrolladora.
    
    '''

    if not isinstance(desarrolladora, str):
        return {'Mensaje': 'Debe ingresar una cadena de texto'}
    
    desarrolladora = desarrolladora.lower()
    
    df = pd.read_parquet('../CleanData/developer_reviews_analysis.parquet')
    
    developers = list(df['developer'].unique()) 

    df = df[df['developer'].isin([desarrolladora])].reset_index()
    if len(df) == 0:
        del df
        return {f'Desarrolladora no encontrada: {desarrolladora}.': f'Desarrolladoras disponibles {", ".join(developers)}'}

    resultado = {
        desarrolladora: [f'Negative = {int(df["negativos"].values[0])}', f'Positive = {int(df["positivos"].values[0])}']
    }

    del df

    return resultado

#-----------------------------------------ENDPOINT 6---------------------------------------#
@app.get('/recomendacion_juego/{id_producto}')
def recomendacion_juego(id_producto: int, n_recom: int = 5, umbral: float = 0.999):
    return item_item_recom(id_producto, n_recom, umbral)