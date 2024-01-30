import pandas as pd
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity
from operator import itemgetter

def item_item_recom(item_id, n_recom=5, umbral= 0.999):
    
    df_names = pd.read_parquet('../Recommender System/item_names.parquet')

    try: 
        item_id = int(item_id)
        indice = df_names[df_names['id'] == item_id].index[0]
    except Exception as e:
        return {'Mensaje': 'Ingrese un id de producto válido',
                'Error': e}
    
    df_items = pd.read_parquet('../Recommender System/item_features_complete.parquet')

    similaridades = {}
    contador = 0
    for i in range(len(df_items)):
        if i != indice:
            sim = cosine_similarity(df_items.iloc[indice,:].values.reshape(1,-1), df_items.iloc[i,:].values.reshape(1,-1))[0][0]
            similaridades[i] = sim
            if sim > umbral:
                contador += 1
            if contador > n_recom:
                break
    
    similaridades_sorted = sorted(similaridades.items(), key= itemgetter(1), reverse=True)

    items_recomendados = []

    for i in range(n_recom):
        items_recomendados.append(similaridades_sorted[i][0])

    resultado = {
                f'item_id: {df_names.loc[items_recomendados[i],"id"]}': df_names.loc[items_recomendados[i],"app_name" ]
                for i in range(len(items_recomendados))
    }

    return resultado