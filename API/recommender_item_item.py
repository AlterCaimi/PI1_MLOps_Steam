import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from operator import itemgetter

def item_item_recom(item_id, n_recom=5, umbral= 0.999):
    
    df_names = pd.read_parquet('../Recommender System/item_names.parquet')      # cargamos el arghivo de nombres

    try: 
        item_id = int(item_id)
        indice = df_names[df_names['id'] == item_id].index[0]                   # verificamos existencia del id ingresado
    except Exception as e:
        return {'Mensaje': 'Ingrese un id de producto válido',                  # mensaje de error en caso de no existir id
                'Error': e}
    
    df_items = pd.read_parquet('../Recommender System/item_features_complete.parquet')  # cargamos archivo de features

    similaridades = {}                  # iniciamos diccionario de similaridades
    contador = 0                        # contador para realizar corte por umbral
    
    for i in range(len(df_items)):      # recorremos el dataframe

        if i != indice:                 # no tomamos el indice del juego de entrada

            sim = cosine_similarity(df_items.iloc[indice,:].values.reshape(1,-1), df_items.iloc[i,:].values.reshape(1,-1))[0][0]
            similaridades[i] = sim      # calclualmos la similaridad y la guardamos en el diccionario
            if sim > umbral:            # verificamos si la similaridad es mayor al umbral predeterimnado
                contador += 1           # si es mayor, sumamos 1 al contador
            if contador > n_recom:      # si se supera la cantidad de recomendaciones del contador, detenemos la búsqueda
                break
    
    similaridades_sorted = sorted(similaridades.items(), key= itemgetter(1), reverse=True)  # ordenamos las similaridades por valores

    items_recomendados = []             # creamos diccionario vacío para acumular los juegos recomendados

    for i in range(n_recom):            # nos quedamos con los indices de los n_recom primeros juegos
        items_recomendados.append(similaridades_sorted[i][0])

    resultado = {                       # generamos diccionario para retornar
                f'item_id: {df_names.loc[items_recomendados[i],"id"]}': df_names.loc[items_recomendados[i],"app_name" ]
                for i in range(len(items_recomendados))
    }

    return resultado                    # retornamos diccionario