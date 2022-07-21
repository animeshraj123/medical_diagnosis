from msilib.schema import Feature
import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.header("Medical Diagnosis DS250 Assignment_1C")
df = st.cache(pd.read_csv)("dataset.csv")
df1 = st.cache(pd.read_csv)("Symptom-severity.csv")
df2 = df1['Symptom'].unique()
df3 = st.cache(pd.read_csv)("symptom_Description.csv")
df4 = st.cache(pd.read_csv)("symptom_precaution.csv")

table = pickle.load(open('symptoms_table.pkl', 'rb'))
pickled_model = open('model.pkl', 'rb')
ans = pickle.load(pickled_model)



symptoms_streamlit = st.multiselect("Symptoms: " ,
                         ['itching', 'skin_rash', 'nodal_skin_eruptions',
       'continuous_sneezing', 'shivering', 'chills', 'joint_pain',
       'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting',
       'vomiting', 'burning_micturition', 'spotting_urination', 'fatigue',
       'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings',
       'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat',
       'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes',
       'breathlessness', 'sweating', 'dehydration', 'indigestion',
       'headache', 'yellowish_skin', 'dark_urine', 'nausea',
       'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain',
       'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever',
       'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure',
       'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes',
       'malaise', 'blurred_and_distorted_vision', 'phlegm',
       'throat_irritation', 'redness_of_eyes', 'sinus_pressure',
       'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
       'fast_heart_rate', 'pain_during_bowel_movements',
       'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus',
       'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity',
       'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes',
       'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties',
       'excessive_hunger', 'extra_marital_contacts',
       'drying_and_tingling_lips', 'slurred_speech', 'knee_pain',
       'hip_joint_pain', 'muscle_weakness', 'stiff_neck',
       'swelling_joints', 'movement_stiffness', 'spinning_movements',
       'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side',
       'loss_of_smell', 'bladder_discomfort', 'foul_smell_ofurine',
       'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching',
       'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain',
       'altered_sensorium', 'red_spots_over_body', 'belly_pain',
       'abnormal_menstruation', 'dischromic_patches',
       'watering_from_eyes', 'increased_appetite', 'polyuria',
       'family_history', 'mucoid_sputum', 'rusty_sputum',
       'lack_of_concentration', 'visual_disturbances',
       'receiving_blood_transfusion', 'receiving_unsterile_injections',
       'coma', 'stomach_bleeding', 'distention_of_abdomen',
       'history_of_alcohol_consumption', 'blood_in_sputum',
       'prominent_veins_on_calf', 'palpitations', 'painful_walking',
       'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling',
       'silver_like_dusting', 'small_dents_in_nails',
       'inflammatory_nails', 'blister', 'red_sore_around_nose',
       'yellow_crust_ooze', 'prognosis'])

st.write("You selected", len(symptoms_streamlit), 'symptoms')

def solve(symptoms_streamlit):

    print(symptoms_streamlit)
    p = table.columns
    our_col = list(p)[:-2]
    feature_vector = np.zeros(len(our_col))
    for i in range(len(our_col)):
        if our_col[i] in symptoms_streamlit:
            feature_vector[i] = 1
    prediction = ans.predict([feature_vector])[0]
    probability = ans.predict_proba([feature_vector])[0]
    max_proba = max(probability)
    
    return (prediction, max_proba)
    


if(st.button("Predict The Disease")):
    result = "Your Predicted Disease is !!!"
    st.success(result)
    
    p = solve(symptoms_streamlit)
    p1 = p[0]
    p2 = p[1]
    
    st.write(f"**Disease:** {p1} with probability {(p2)*100:.2f}%")


    desc = None
    for i in range(len(df3.index)):
        if df3.iloc[i][0] == p1:
            desc = df3.iloc[i][1]
            break
    st.warning("**Description of the predicted disease :**")
    st.write(desc)

    prec1 = None
    prec2 = None
    prec3 = None
    prec4 = None
    for i in range(len(df4.index)):
        if df4.iloc[i][0] == p1:
            prec1 = df4.iloc[i][1]
            prec2 = df4.iloc[i][2]
            prec3 = df4.iloc[i][3]
            prec4 = df4.iloc[i][4]
            break
    st.info("**Precautions you must take :**") 
    st.write("**Precaution 1 :**", prec1)
    st.write("**Precaution 2 :**", prec2)
    st.write("**Precaution 3 :**", prec3)
    st.write("**Precaution 4 :**", prec4)

##################################################################################################################