import pandas as pd
import os 
import zipfile

def preprocess_data(zip_path:str, station: str) -> pd.DataFrame:
    
    data_store = "data_ext"

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_store)

    df_collection = []

    for root, dirs, files in os.walk(data_store):
        for f in files:
            if f.endswith(".csv"):
                path = os.path.join(root, f)
                df = pd.read_csv(path)
                if "station" in df.columns and station in df["station"].values:
                    df_collection.append(df)
    
    df = pd.concat(df_collection)

    df[df.select_dtypes(include='number').columns] = df.select_dtypes(include='number').fillna(0)
    
    df.to_csv("air_quality_cleaned.csv")
    file_path = os.path.abspath("air_quality_cleaned.csv")