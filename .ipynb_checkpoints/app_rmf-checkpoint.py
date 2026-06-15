import streamlit as st
import pandas as pd

from joblib import load

model = load('rmf_model.joblib')

education_encoder = load('rmf_education_encoder.joblib')

target_encoder = load('rmf_target_encoder.joblib')

mlb = load('rmf_mlb_encoder.joblib')

training_columns = load('rmf_training_columns.joblib')

st.set_page_config(
    page_title='AI Career Recommendation System',
    page_icon='💼'
)

# TITLE

st.title('💼 AI Career Recommendation System')

st.write(
    'Predict suitable job roles based on skills and education level.'
)

st.markdown('---')

# USER INPUTS

age = st.slider(
    'Age',
    18,
    60,
    25
)

education = st.selectbox(
    'Education Level',
    education_encoder.classes_
)


# SKILLS INPUT
all_skills = sorted(list(mlb.classes_))

selected_skills = st.multiselect(
    'Select Current Skills',
    all_skills
)

# JOB ROLE DETAILS


role_details = {

    'Web Developer': {
        'skill_gap': 'JavaScript, React, Git',
        'training': 'Frontend Development Program'
    },

    'Certified Plumber': {
        'skill_gap': 'Pipe Fitting, Safety Standards',
        'training': 'Advanced Plumbing Certification'
    },

    'Modern Farm Manager': {
        'skill_gap': 'Irrigation, Modern Agriculture',
        'training': 'Modern Agriculture & Agritech Training'
    },

    'Customer Support Executive': {
        'skill_gap': 'Professional Communication, CRM Tools',
        'training': 'BPO & Customer Service Training'
    },

    'Data Entry Operator': {
        'skill_gap': 'Advanced Excel, Documentation',
        'training': 'MS Office & Computer Basics Course'
    },

    'Sales Associate': {
        'skill_gap': 'Leadership, Retail Operations',
        'training': 'Retail Sales & Communication Course'
    },

    'Junior Accountant': {
        'skill_gap': 'GST Filing, Tally',
        'training': 'Accounting & Tally Certification'
    },

    'Retail Store Manager': {
        'skill_gap': 'Retail Analytics, Team Management',
        'training': 'Retail Management Certification'
    }
}

# PREDICTION BUTTON

if st.button('Predict Job Role'):

    if len(selected_skills) == 0:

        st.warning('Please select at least one skill')

    else:

        # CREATE INPUT DATAFRAME
        input_df = pd.DataFrame([{
            'age': age,
            'education_level': education
        }])

        # Encode education level

        input_df['education_level'] = education_encoder.transform(
            input_df['education_level']
        )

        # ENCODE SKILLS

        skills_encoded = mlb.transform([selected_skills])

        skills_df = pd.DataFrame(
            skills_encoded,
            columns=mlb.classes_
        )

        # FINAL INPUT

        final_input = pd.concat(
            [input_df.reset_index(drop=True), skills_df],
            axis=1
        )

        # Match training columns

        final_input = final_input.reindex(
            columns=training_columns,
            fill_value=0
        )

        # PREDICTION

        prediction = model.predict(final_input)

        predicted_role = target_encoder.inverse_transform(
            prediction
        )[0]

        # PREDICTION PROBABILITIES

        probabilities = model.predict_proba(final_input)[0]

        top_3_idx = probabilities.argsort()[-3:][::-1]

        # DISPLAY RESULT

        st.success(f'Predicted Job Role: {predicted_role}')

        # SKILL GAP + TRAINING

        details = role_details.get(predicted_role)

        if details:

            st.subheader('Predicted Skill Gap')
            st.info(details['skill_gap'])

            st.subheader('Recommended Training')
            st.success(details['training'])

        # TOP 3 PREDICTIONS

        st.subheader('Top 3 Predictions')

        for idx in top_3_idx:

            role = target_encoder.inverse_transform([idx])[0]

            prob = probabilities[idx] * 100

            st.write(f'{role}: {prob:.2f}%')

st.markdown('---')

st.caption('Built using Streamlit + Random Forest')