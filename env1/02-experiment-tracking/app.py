import json
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.model_selection import RepeatedStratifiedKFold, RandomizedSearchCV, GridSearchCV, train_test_split, KFold
 
app = FastAPI()
 
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("student-perform-experiment1")
 
class InputData(BaseModel):
    StudentID: int
    Age: int
    Gender: int
    Ethnicity: int
    ParentalEducation: int
    StudyTimeWeekly: float
    Absences: float
    Tutoring: int
    ParentalSupport: int
    Extracurricular: int
    Sports: int
    Music: int
    Volunteering: int
    GPA:float
    GradeClass:float

 
def get_latest_run_id(experiment_name):
    client = MlflowClient()
    experiment_id = '1'
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment:
        experiment_id = experiment.experiment_id
        runs = client.search_runs(experiment_id, order_by=["start_time DESC"], max_results=1)
        if runs:
            return runs[0].info.run_id
    return None
def load_model_and_preprocessor():
    experiment_name = 'student-perform-experiment1'
    run_id = get_latest_run_id(experiment_name)
    print("the run_id is ", run_id)
 
    if run_id is None:
        raise Exception(f"No runs found for experiment '{experiment_name}'")
 
    # Load the model from MLflow using the latest run ID
    logged_model = f'runs:/{run_id}/model'
    #preprocessor_uri = f"runs:/{run_id}/preprocessor"
    model = mlflow.sklearn.load_model(logged_model)
    #preprocessor = mlflow.sklearn.load_model(preprocessor_uri)
 
    return model
 
model = load_model_and_preprocessor()
 
 
# # Define the preprocessing pipeline
# categorical_features = ['Airline', 'Source', 'Destination']
# encoder = ColumnTransformer(
#     transformers=[
#         ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
#     ],
#     remainder='passthrough'
# )
# scaler = StandardScaler()
 
# preprocessor = Pipeline(steps=[
#     ('encoder', encoder),
#     ('scaler', scaler)
# ])
 
# Dummy DataFrame to fit the ColumnTransformer
# Adjust this based on your actual data for fitting purposes
# dummy_data = pd.DataFrame({
#     'Airline': ['Dummy_Airline'],
#     'Source': ['Dummy_Source'],
#     'Destination': ['Dummy_Destination']
# })
 
# # Fit the preprocessor pipeline with dummy data
# try:
#     preprocessor.fit(dummy_data)
# except Exception as e:
#     raise Exception(f"Error in fitting preprocessing pipeline: {str(e)}")
 
source_data= "data/Student_performance_data _.csv"

def data_load(source_file,unique_maxcount):
    #read the file to dataframe
    df = pd.read_csv(source_file)

    # Distinction is based on the number of different values in the column
    columns = list(df.columns)
    categoric_columns = []
    numeric_columns = []
    for i in columns:
        if len(df[i].unique()) > unique_maxcount:
            numeric_columns.append(i)   
        else:
            categoric_columns.append(i)
    # Assuming the first column is an ID or non-numeric feature
    numeric_columns = numeric_columns[1:]

    # Convert numeric columns to float64
    df[numeric_columns] = df[numeric_columns].astype('float64')

    return df

def features(df):
    # CHOOSE THE TARGET FEATURE HERE, IN THIS CASE IT IS 'GradeClass'
    X = df.drop(columns=['GradeClass', 'GPA', 'StudentID', 'Age'])
    y = df['GradeClass']

    # Splitting the data into training and testing sets (e.g., 80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    return X_train, X_test, y_train, y_test



@app.post("/predict/")
def predict(data: InputData):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([data.dict()])
   
    # Preprocess the input data
    # try:
    #     input_processed = preprocessor.transform(input_df)
    # except Exception as e:
    #     raise HTTPException(status_code=400, detail=f"Error in preprocessing: {str(e)}")
   
    # Make a prediction

    

    try:
        X_train, X_test, y_train, y_test = features(data_load(source_data, 5))
        model.fit(X_train, y_train)
        prediction =model.predict(X_train)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in prediction: {str(e)}")
   
    return {"predicted_score": prediction[0]}
 
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=5001)