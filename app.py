import gradio as gr
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open("covid_model.pkl", "rb"))

# Define the prediction function
def predict_covid(Breathing_Problem, Fever, Dry_Cough, Sore_Throat, Running_Nose, Asthma, Chronic_Lung_Disease, 
                  Headache, Heart_Disease, Diabetes, Hyper_Tension, Fatigue, Gastrointestinal, Abroad_Travel, 
                  Contact_with_COVID_Patient, Attended_Large_Gathering, Visited_Public_Exposed_Places, 
                  Family_Working_in_Public_Exposed_Places, Wearing_Masks, Sanitization_from_Market):
    
    # Create the input array for the model
    input_data = np.array([[Breathing_Problem, Fever, Dry_Cough, Sore_Throat, Running_Nose, Asthma, Chronic_Lung_Disease, 
                            Headache, Heart_Disease, Diabetes, Hyper_Tension, Fatigue, Gastrointestinal, Abroad_Travel, 
                            Contact_with_COVID_Patient, Attended_Large_Gathering, Visited_Public_Exposed_Places, 
                            Family_Working_in_Public_Exposed_Places, Wearing_Masks, Sanitization_from_Market]])
    
    # Predict the result
    prediction = model.predict(input_data)[0]
    
    return "Infected with COVID-19" if prediction == 1 else "Not infected with COVID-19"

# Create the Gradio interface
demo = gr.Interface(
    fn=predict_covid,
    inputs=[
        gr.Number(label="Breathing Problem", min=0, max=1),
        gr.Number(label="Fever", min=0, max=1),
        gr.Number(label="Dry Cough", min=0, max=1),
        gr.Number(label="Sore Throat", min=0, max=1),
        gr.Number(label="Running Nose", min=0, max=1),
        gr.Number(label="Asthma", min=0, max=1),
        gr.Number(label="Chronic Lung Disease", min=0, max=1),
        gr.Number(label="Headache", min=0, max=1),
        gr.Number(label="Heart Disease", min=0, max=1),
        gr.Number(label="Diabetes", min=0, max=1),
        gr.Number(label="Hyper Tension", min=0, max=1),
        gr.Number(label="Fatigue", min=0, max=1),
        gr.Number(label="Gastrointestinal", min=0, max=1),
        gr.Number(label="Abroad Travel", min=0, max=1),
        gr.Number(label="Contact with COVID Patient", min=0, max=1),
        gr.Number(label="Attended Large Gathering", min=0, max=1),
        gr.Number(label="Visited Public Exposed Places", min=0, max=1),
        gr.Number(label="Family working in Public Exposed Places", min=0, max=1),
        gr.Number(label="Wearing Masks", min=0, max=1),
        gr.Number(label="Sanitization from Market", min=0, max=1)
    ],
    outputs="text",
    title="COVID-19 Prediction",
    description="Enter symptoms and habits to predict if the individual is infected with COVID-19"
)

# Launch the interface
demo.launch(share=True)  # Use share=True for a public URL
