
#This is a temporary script file. which has to ran on conda prompt window using the below 2 commands
#cd "C:\Users\Kishore kumar V\OneDrive\Desktop\MLAPI"
#uvicorn ml_api:app

from fastapi import FastAPI
from pydantic import BaseModel #The purpose of the base model is to up the format in which data will
                               #will be posted to our API
import pickle
import json

app = FastAPI()

class model_input(BaseModel):
      Pregnancies : int
      Glucose : float
      BloodPressure : float
      SkinThickness : float
      Insulin : float
      BMI : float
      DiabetesPedigreeFunction : float
      Age : int
      
      
#Loading the saved model
diabetes_model = pickle.load(open('diabetics.sav', 'rb'))

@app.post('/diabetes_prediction') #/diabetes_prediction it is end point for the URL
def diabetes_pred(input_parameters : model_input):
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    
    preg =input_dictionary['Pregnancies']
    glu =input_dictionary['Glucose']
    bp =input_dictionary['BloodPressure']
    skin =input_dictionary['SkinThickness']
    insulin =input_dictionary['Insulin']
    bmi =input_dictionary['BMI']
    dpf =input_dictionary['DiabetesPedigreeFunction']
    age =input_dictionary['Age']
    
    
    input_list= [preg,glu,bp,skin,insulin,bmi,dpf,age]
    
    prediction = diabetes_model.predict([input_list])
    
    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'
    
    