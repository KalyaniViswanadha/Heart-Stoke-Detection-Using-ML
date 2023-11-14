# import flask
#import pickle
# from flask import Flask, render_template, request, jsonify
#import joblib
# from sklearn.linear_model import LogisticRegression

# app = Flask(__name__)
import pickle
from flask import Flask,render_template,request
app=Flask(__name__)

from sklearn.linear_model import LogisticRegression
import numpy as np

model = pickle.load(open('heart_stroke.pkl','rb'))

# # Load or train your machine learning model
# # For this example, we'll use a simple Logistic Regression model
# try:
#     # Load the pre-trained model
#     model = joblib.load('heart_stroke.pkl')
# except FileNotFoundError:
#     # Train the model if the saved model file is not found
#     # Replace this with your actual model training code
#     X_train = [[35, 0, 0, 120, 25.5]]  # Example input data for training
#     y_train = [0]  # Example target labels for training

#     model = LogisticRegression()
#     model.fit(X_train, y_train)

#     # Save the trained model to a file
#     joblib.dump(model, 'heart_stroke.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
# def predict():
#     # Get data from the form
#     data = request.form.to_dict()

#     # Extract input features
#     features = [int(data['age']), int(data['hypertension']), int(data['heartDisease']),
#                 int(data['avgGlucoseLevel']), float(data['bmi'])]

#     # Use your model to make a prediction
#     prediction = model.predict([[request.form.get('age','hypertension','heartDisease','avgGlucoseLevel','bmi')]])
#     output=prediction[0]
#     # Return the prediction as JSON
#     #return jsonify({'prediction': prediction})
#     return render_template('index.html', predicted_output="This person falls under the class of"+output)
def predict():
    # Get data from the form
    data = request.form.to_dict()

    # Extract input features
    features = [int(data['age']), int(data['hypertension']), int(data['heartDisease']),
                int(data['avgGlucoseLevel']), float(data['bmi'])]

    # Use your model to make a prediction
    prediction = model.predict([features])
    output = prediction[0]

    # Return the prediction as JSON
    return render_template('index.html', predicted_output="This person falls under the class of " + str(output))
if __name__ == '__main__':
    app.run(debug=True)






# from flask import Flask, request, render_template
# import pickle
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()

# model = pickle.load(open("heart_stroke.pkl", 'rb'))

# app = Flask(__name__)
# @app.route('/analysis')
# def analysis():
#     return render_template("index.html")

# @app.route('/', methods=['GET', 'POST'])
# def home():
#     if request.method =="POST":
#         gender = request.form['gender']
#         age = int(request.form['age'])
#         hypertension = int(request.form['hypertension'])
#         disease = int(request.form['disease'])
#         married = request.form['married']
#         work = request.form['work']
#         residence = request.form['residence']
#         glucose = float(request.form['glucose'])
#         bmi = float(request.form['bmi'])
#         smoking = request.form['smoking']

#         # gender
#         if (gender == "Male"):
#             gender_male=1
#             gender_other=0
#         elif(gender == "Other"):
#             gender_male = 0
#             gender_other = 1
#         else:
#             gender_male=0
#             gender_other=0
        
#         # married
#         if(married=="Yes"):
#             married_yes = 1
#         else:
#             married_yes=0

#         # work  type
#         if(work=='Self-employed'):
#             work_type_Never_worked = 0
#             work_type_Private = 0
#             work_type_Self_employed = 1
#             work_type_children=0
#         elif(work == 'Private'):
#             work_type_Never_worked = 0
#             work_type_Private = 1
#             work_type_Self_employed = 0
#             work_type_children=0
#         elif(work=="children"):
#             work_type_Never_worked = 0
#             work_type_Private = 0
#             work_type_Self_employed = 0
#             work_type_children=1
#         elif(work=="Never_worked"):
#             work_type_Never_worked = 1
#             work_type_Private = 0
#             work_type_Self_employed = 0
#             work_type_children=0
#         else:
#             work_type_Never_worked = 0
#             work_type_Private = 0
#             work_type_Self_employed = 0
#             work_type_children=0

#         # residence type
#         if (residence=="Urban"):
#             Residence_type_Urban=1
#         else:
#             Residence_type_Urban=0

#         # smoking sttaus
#         if(smoking=='formerly smoked'):
#             smoking_status_formerly_smoked = 1
#             smoking_status_never_smoked = 0
#             smoking_status_smokes = 0
#         elif(smoking == 'smokes'):
#             smoking_status_formerly_smoked = 0
#             smoking_status_never_smoked = 0
#             smoking_status_smokes = 1
#         elif(smoking=="never smoked"):
#             smoking_status_formerly_smoked = 0
#             smoking_status_never_smoked = 1
#             smoking_status_smokes = 0
#         else:
#             smoking_status_formerly_smoked = 0
#             smoking_status_never_smoked = 0
#             smoking_status_smokes = 0

#         feature = scaler.fit_transform([[age, hypertension, disease, glucose, bmi, gender_male, gender_other, married_yes, work_type_Never_worked, work_type_Private, work_type_Self_employed, work_type_children, Residence_type_Urban,smoking_status_formerly_smoked, smoking_status_never_smoked, smoking_status_smokes]])

#         prediction = model.predict(feature)[0]
#         # print(prediction) 
#         # 
#         if prediction==0:
#             prediction = "NO" 
#         else:
#             prediction = "YES" 

#         return render_template("index.html", prediction_text="Chance of Stroke Prediction is --> {}".format(prediction))   
         

#     else:
#         return render_template("index.html")





# if __name__ == "__main__":
#     app.run(debug=True)