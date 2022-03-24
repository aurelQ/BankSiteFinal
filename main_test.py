# Import Needed Libraries
import joblib
import uvicorn
import numpy as np
import pandas as pd
from pydantic import BaseModel
import sklearn.metrics
import pickle
# FastAPI libray
from fastapi import FastAPI

# Initiate app instance
app = FastAPI(title='Placement Analytics', version='0.24.2',
              description='Lightgbm model is used for prediction')

# Initialize model artifacte files. This will be loaded at the start of FastAPI model server.
#grid_result_xgb_w  = joblib.load('../model/grid_result_xgb_w.joblib')
#clf = joblib.load('../model/clf_xgb_w.joblib')
X_test=pd.read_csv('model/X_train_ech.csv')
#explainer=pd.read_csv('../model/explainer_xgb.joblib')

# read pickle files
with open('model/score_objects2.pkl', 'rb') as handle:
    clf_xgb_w, explainer_xgb = pickle.load(handle)  

# This struture will be used for Json validation.
class Data(BaseModel):
    test_id: int


# Api root or home endpoint
@app.get('/')
@app.get('/home')
def read_home():
    """
     Home endpoint which can be used to test the availability of the application.
     """
    return {'message': 'System is healthy'}

# ML API endpoint for making prediction aganist the request received from client
@app.post("/predict")
def predict(data: Data):
    # Extract data in correct order
    data_dict = data.dict() #{data_dict={'test_id':3}} #    data_df = pd.DataFrame.from_dict([data_dict])
    var=data_dict['test_id']
    X_test_df=X_test[X_test["Unnamed: 0"]==var]   #  data_df_2=df_csv.loc[var]
    X_test_df=X_test_df.to_dict('series')
    del X_test_df['Unnamed: 0']
    X_test_df=pd.DataFrame.from_dict([X_test_df])
    print("X_test_df :", X_test_df )

    # Créer la prediction et le score
    seuil=0.2
    prediction_label = clf_xgb_w.predict_proba(X_test_df)[0,1]
    print("prediction_label", type(prediction_label)) #    prediction = clf.predict_proba(data_df_3)
    if prediction_label > seuil:
        prediction = 1
    else:
        prediction=0

    print("prediction :", prediction )
    score = float(prediction_label)
    print("score :", score )

    #Création du dictionnaire de données de sortie 
    result_dict={"prediction":prediction, "score":score}
    #result_dict={"prediction":[], "score":[]}
    #result_dict["prediction"].append(prediction)
    #result_dict["score"].append(score)
    print("result_dict :", result_dict )
    # Return response back to client
    return result_dict

#if __name__ == '__main__':
    #uvicorn.run("main_test:app", host="0.0.0.0", port=8000, reload=True)
 #   uvicorn.run("main_test:app" reload=True)
    