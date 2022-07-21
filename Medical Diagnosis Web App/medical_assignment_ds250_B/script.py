from msilib.schema import Feature
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import math

class DecisionTree:
    
    def __init__(self, max_depth=None):
        self.nodes = {}
        self.parentNode = 1
        self.leafNode = {}
        self.leafProbability = {}
        self.y_train = None
        self.b = None
        if max_depth is None:
          self.max_depth = np.inf
        else:
          self.max_depth = max_depth
    
    def fit(self, x_train, y_train, depth=0):
        self.parentNode = 1
        self.y_train = y_train
        self.b = len(np.unique(y_train))
        self.unique_labels = np.unique(y_train)
        self.dTree(self.parentNode, depth, x_train, y_train)
    
    def entropyCal(self, y):
        # Input: y (labels)
        # Output: entropy
         
        # number of labels      
        unique_labels = self.unique_labels
        n = self.b
        
        if len(y) == 0:
          return 0

        if n == 1:
            return 0
        
        e = 0
        for i in range(n):
            label = unique_labels[i]
            cnt = 0
            for i in y:
              if i == label:
                cnt += 1
            prob = cnt / len(y)
            if prob == 0:
              e += 0
            else:
              e += -1 * prob * math.log(prob, n)
        
        return e
    
    def dTree(self, node, depth, x_train=None, y_train=None):
        
        # if the node is a leaf node
        if (depth >= self.max_depth):
            
            # Probability of each label
            probs = {}
            for label in y_train:
                if label not in probs:
                    probs[label] = 0
                probs[label] += 1
            # Normalize
            for label in probs:
                probs[label] /= len(y_train)
            
            # Label with highest probability
            max_prob = 0
            max_label = None
            for label in probs:
                if probs[label] > max_prob:
                    max_prob = probs[label]
                    max_label = label
            
            self.leafNode[node] = max_label
            self.leafProbability[node] = probs
            
            return        
        
        if  len(np.unique(y_train)) == 1:
          
            # Probability of each label
            probs = {}
            for label in y_train:
                if label not in probs:
                    probs[label] = 0
                probs[label] += 1
            # Normalize
            for label in probs:
                probs[label] /= len(y_train)
            
            # Label with highest probability
            max_prob = 0
            max_label = None
            for label in probs:
                if probs[label] > max_prob:
                    max_prob = probs[label]
                    max_label = label
            
            self.leafNode[node] = max_label
            self.leafProbability[node] = probs
            
            return                  

        max_ig = -np.inf
        max_ig_index = -1
        dsLeft_x = []
        dsLeft_y = []
        dsRight_x = []
        dsRight_y = []
        
        if len(y_train) == 0:
            return
        
        number_of_possible_splits = len(x_train[0])
        
        for i in range(number_of_possible_splits):
             
            ds1_x = []
            ds1_y = []
            ds2_x = []
            ds2_y = []
            
            for j in range(len(x_train)):
                if x_train[j][i] == 1:
                    ds1_x.append(x_train[j])
                    ds1_y.append(y_train[j])
                else:
                    ds2_x.append(x_train[j])
                    ds2_y.append(y_train[j])
            
            # entropy of the current node
            current_entropy = self.entropyCal(y_train)
            
            # entropy of the left node
            left_entropy = self.entropyCal(ds1_y)
            
            # entropy of the right node
            right_entropy = self.entropyCal(ds2_y)
            
            # information gain
            ig = current_entropy - (len(ds1_y) / len(y_train)) * left_entropy - (len(ds2_y) / len(y_train)) * right_entropy
            
            # if information gain is greater than the previous one, update the max_ig and max_ig_index
            if ig > max_ig:
                max_ig = ig
                max_ig_index = i
                dsLeft_x = ds1_x
                dsLeft_y = ds1_y
                dsRight_x = ds2_x
                dsRight_y = ds2_y
            
            
        self.nodes[node] = max_ig_index

        if max_ig_index == -1:
            return
        
        # create the left node
        left_node = node * 2
        self.dTree(left_node, depth+1, dsLeft_x, dsLeft_y)
        
        # create the right node
        right_node = node * 2 + 1
        self.dTree(right_node, depth+1, dsRight_x, dsRight_y)
            
        return
    
    def predict(self, x_test):
        
        y_pred = []
         
        for i in range(len(x_test)):
            y_pred.append(self.predictLabel(x_test[i]))
        
        return y_pred

    def predictLabel(self, x_test):
            
            node = 1
            
            # traverse the tree
            while node not in self.leafNode:

                if x_test[self.nodes[node]] == 1:
                    node = node * 2
                else:
                    node = node * 2 + 1
            
            # return the label
            return self.leafNode[node]
        
    
    def predictProbability(self, x_test):
            
            y_pred = []
            
            for i in range(len(x_test)):
                y_pred.append(self.predictProbabilityLabel(x_test[i]))
            
            return y_pred

    def predictProbabilityLabel(self, x_test):
            
            node = 1
            
            # traverse the tree
            while node not in self.leafProbability:
                if x_test[self.nodes[node]] == 1:
                    node = node * 2
                else:
                    node = node * 2 + 1
            
            # return the probability
            return self.leafProbability[node]


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
    prediction_p = ans.predictProbability([feature_vector])[0]
    return (prediction, prediction_p)
    


if(st.button("Predict The Disease")):
    result = "Your Predicted Disease is !!!"
    st.success(result)
    
    p = solve(symptoms_streamlit)
    p1 = p[0]
    p2 = p[1]
    
    st.write(f"**Disease:** {p1} with probability {(p2[p1])*100:.2f}%")

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