import pandas as pd
import joblib
import streamlit as st
from sklearn.preprocessing import StandardScaler

# Load the model and mappings
model = joblib.load('log.pkl')
# Mapping dictionaries
Sex_mapping = {'Male': 1, 'Female': 0}
race_mapping = {'White': 0, 'Other (American Indian/AK Native, Asian/Pacific Islander)': 1, 'Black': 2}
Marital_status_mapping = {'Married': 1, 'Not married': 0}
AJCC_Stage_mapping= {'IIA': 0, 'IIB': 1, 'IVB': 2, 'IVA': 3, 'III': 4, 'IV': 5}
AJCC_T_stage_mapping= {'T1': 0, 'T2': 1, 'T3': 2, 'T2b': 3}
AJCC_N_stage_mapping= {'N0': 0, 'N1': 1}
AJCC_M_stage_mapping = {'M0': 0, 'M1b': 1, 'M1a': 2, 'M1': 3}

Primary_Site_mapping= {'C40.0-Long bones: upper limb, scapula, and associated joints': 0, 
                     'C41.4-Pelvic bones, sacrum, coccyx and associated joints': 1, 
                     'C41.3-Rib, sternum, clavicle and associated joints': 2, 'C40.2-Long bones of lower limb and associated joints': 3,
                       'C41.2-Vertebral column': 4, 'C41.0-Bones of skull and face and associated joints': 5,
                         'C40.3-Short bones of lower limb and associated joints': 6, 'C40.1-Short bones of upper limb and associated joints': 7,
                           'C41.1-Mandible': 8, 'C41.8-Overlap bones, joints, and art. cartilage': 9,
                      'C40.8-Overlap of bones, joints, and art. cartilage of limbs': 10, 'C38.1-Anterior mediastinum': 11}
Stage_mapping= {'Regional': 0, 'Distant': 1, 'Localized': 2}
Surgery_mapping= {'Yes': 1, 'No': 0}
Radiation_mapping= {'Yes': 1, 'No': 0}
Chemotherapy_mapping= {'No': 0, 'Yes': 1}


# Streamlit App
st.title ("5 years Survival prediction model for Ewing sarcoma)
Age = st.slider("Age (years)", min_value=0, max_value=100, value=50, step=1)
Sex = st.radio("Sex", ['Male', 'Female'])
Race = st.selectbox("Race", ['White', 'Other (American Indian/AK Native, Asian/Pacific Islander)', 'Black'])
Maritalstatus = st.radio("Marital status", ['Married', 'Not married'])
AJCC_Stage= st.selectbox("AJCC Stage",['IIA', 'IIB', 'IVB', 'IVA', 'III', 'IV'])
AJCC_T_stage= st.selectbox("AJCC T Stage", ['T1', 'T2', 'T3', 'T2b'])
AJCC_N_stage= st.radio("AJCC N Stage", ['N0', 'N1'])
AJCC_M_stage = st.selectbox("AJCC M Stage", ['M0', 'M1b', 'M1a', 'M1'])
Site = st.selectbox("Primary Site", ['C40.0-Long bones: upper limb, scapula, and associated joints', 
                     'C41.4-Pelvic bones, sacrum, coccyx and associated joints', 
                     'C41.3-Rib, sternum, clavicle and associated joints', 'C40.2-Long bones of lower limb and associated joints',
                       'C41.2-Vertebral column', 'C41.0-Bones of skull and face and associated joints',
                         'C40.3-Short bones of lower limb and associated joints', 'C40.1-Short bones of upper limb and associated joints',
                           'C41.1-Mandible', 'C41.8-Overlap bones, joints, and art. cartilage',
                      'C40.8-Overlap of bones, joints, and art. cartilage of limbs', 'C38.1-Anterior mediastinum'])
Stage = st.selectbox("Stage", ['Localized', 'Regional', 'Distant'])
Tumorsize = st.slider("tumor size", min_value=0, max_value=500, value=50, step=1)
Surgery= st.radio("Surgery", ['Yes', 'No'])
Radiation= st.radio("Radiation", ['Yes', 'No'])
Chemotherapy= st.radio("Chemotherapy", ['No', 'Yes'])

# Preprocess the user input using the same mappings
user_df = pd.DataFrame({
    'Sex': [Sex],
    'Race': [Race],
    'Age': [Age],
    'Marital status': [Maritalstatus],
    'AJCC Stage Group': [AJCC_Stage],
    'AJCC T': [AJCC_T_stage],
    'AJCC N':[AJCC_N_stage],
    'AJCC M': [AJCC_M_stage],
    'Surgery': [Surgery],
    'Radiation': [Radiation],
    'Chemotherapy': [Chemotherapy],
    'tumor size': [Tumorsize],
    'Primary Site': [Site],
    'Stage': [Stage],    

})

# Apply mappings to the dataframe
user_df['Sex'] = user_df['Sex'].map(Sex_mapping)
user_df['Race'] = user_df['Race'].map(race_mapping)
user_df['Marital status'] = user_df['Marital status'].map(Marital_status_mapping)
user_df['Primary Site'] = user_df['Primary Site'].map(Primary_Site_mapping)
user_df['AJCC Stage Group'] = user_df['AJCC Stage Group'].map(AJCC_Stage_mapping)
user_df['AJCC T'] = user_df['AJCC T'].map(AJCC_T_stage_mapping)
user_df['Stage'] = user_df['Stage'].map(Stage_mapping)
user_df['AJCC M'] = user_df['AJCC M'].map(AJCC_M_stage_mapping)
user_df['AJCC N'] = user_df['AJCC N'].map(AJCC_N_stage_mapping)
user_df['Chemotherapy'] = user_df['Chemotherapy'].map(Chemotherapy_mapping)
user_df['Radiation'] = user_df['Radiation'].map(Radiation_mapping)
user_df['Surgery'] = user_df['Surgery'].map(Surgery_mapping)
# Reshape 'Age' into a 2D array and apply the scaler
scaler = StandardScaler()
user_df['Age'] = scaler.fit_transform(user_df[['Age']])
# Reshape 'Age' into a 2D array and apply the scaler
user_df['tumor size'] = scaler.fit_transform(user_df[['tumor size']])

user_df['Age'] = user_df['Age'].astype(float)  # Convert 'Age' to float (you may adjust the data type)
# Reshape 'Age' into a 2D array and apply the scaler
user_df['tumor size'] = user_df['tumor size'].astype(float) 
if st.button("Make Prediction"):
    # Assuming 'model' is your pre-trained model
    prediction = model.predict(user_df)  # Make predictions using the model
    
    # Assuming the model is a classification model, predicting the survival status (alive or dead)
    if model.predict_proba:
        survival_probability = model.predict_proba(user_df)[0][1]  # Get the probability of being alive (1)

        # Decision rule: If survival probability is greater than 50%, patient is alive; otherwise dead
        if survival_probability > 0.5:
            st.write(f"The patient is predicted to be alive with a survival probability of {survival_probability * 100:.2f}%")
        else:
            st.write(f"The patient is predicted to be dead with a survival probability of {survival_probability * 100:.2f}%")
    else:
        # If the model doesn't support probabilities, you can just print the prediction result
        if prediction < 0:
            st.write("The predicted survival years for this patient is negative. Unable to provide accurate prediction.")
        else:
            st.write(f"The predicted survival years for this patient is: {prediction[0] * 12} months")


