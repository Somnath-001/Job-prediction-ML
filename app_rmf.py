import streamlit as st
import pandas as pd
from joblib import load

# =========================================================
# LOAD MODEL FILES
# =========================================================

model = load('rmf_model.joblib')
education_encoder = load('rmf_education_encoder.joblib')
target_encoder = load('rmf_target_encoder.joblib')
mlb = load('rmf_mlb_encoder.joblib')
training_columns = load('rmf_training_columns.joblib')

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title='AI Career Recommendation System',
    page_icon='💼',
    layout='centered'
)

# =========================================================
# HEADER
# =========================================================

st.title("💼 AI Career Recommendation System")

st.write(
    "Predict suitable job roles based on your education, skills and profile."
)

st.markdown("---")

# =========================================================
# PERSONAL DETAILS
# =========================================================

st.subheader("👤 Personal Information")

age = st.slider(
    "Age",
    18,
    60,
    25
)

education = st.selectbox(
    "Education Level",
    education_encoder.classes_
)

academic_score = st.slider(
    "Academic Percentage",
    0,
    100,
    70
)

experience = st.slider(
    "Years of Experience",
    0,
    20,
    0
)

# =========================================================
# ADDITIONAL PROFILE DETAILS
# =========================================================

st.subheader("📋 Professional Profile")

employment_status = st.selectbox(
    "Employment Status",
    [
        "Student",
        "Unemployed",
        "Part-Time",
        "Full-Time"
    ]
)

location_type = st.selectbox(
    "Location Type",
    [
        "Rural",
        "Semi-Urban",
        "Urban"
    ]
)

english_level = st.selectbox(
    "English Proficiency",
    [
        "Beginner",
        "Intermediate",
        "Advanced"
    ]
)

career_interest = st.selectbox(
    "Preferred Career Domain",
    [
        "Technology",
        "Business",
        "Finance",
        "Agriculture",
        "Construction",
        "Customer Service"
    ]
)

# =========================================================
# SKILLS
# =========================================================

st.subheader("🛠 Skills")

all_skills = sorted(list(mlb.classes_))

selected_skills = st.multiselect(
    "Select Your Current Skills",
    all_skills
)

# =========================================================
# CERTIFICATIONS
# =========================================================

st.subheader("🏆 Certifications")

certifications = st.multiselect(
    "Select Certifications (Optional)",
    [
        "Python",
        "Excel",
        "Tally",
        "Data Analytics",
        "Digital Marketing",
        "AWS",
        "Networking"
    ]
)

# =========================================================
# ROLE DETAILS
# =========================================================

role_details = {

    'Web Developer': {
        'skill_gap': 'JavaScript, React, Git',
        'training': 'Frontend Development Program',
        'recommended': ['React', 'Node.js', 'Git']
    },

    'Certified Plumber': {
        'skill_gap': 'Pipe Fitting, Safety Standards',
        'training': 'Advanced Plumbing Certification',
        'recommended': ['Safety Standards', 'Blueprint Reading']
    },

    'Modern Farm Manager': {
        'skill_gap': 'Irrigation, Agritech',
        'training': 'Modern Agriculture Training',
        'recommended': ['Agritech', 'Farm Analytics']
    },

    'Customer Support Executive': {
        'skill_gap': 'CRM Tools, Communication',
        'training': 'Customer Service Program',
        'recommended': ['CRM', 'Communication']
    },

    'Data Entry Operator': {
        'skill_gap': 'Excel, Documentation',
        'training': 'MS Office Course',
        'recommended': ['Excel', 'Typing']
    },

    'Sales Associate': {
        'skill_gap': 'Leadership, Retail Operations',
        'training': 'Retail Sales Program',
        'recommended': ['Negotiation', 'Sales']
    },

    'Junior Accountant': {
        'skill_gap': 'GST, Tally',
        'training': 'Accounting Certification',
        'recommended': ['Tally', 'GST']
    },

    'Retail Store Manager': {
        'skill_gap': 'Retail Analytics',
        'training': 'Retail Management Program',
        'recommended': ['Analytics', 'Team Management']
    }
}

# =========================================================
# PREDICTION
# =========================================================

if st.button("🚀 Predict Job Role"):

    if len(selected_skills) == 0:

        st.warning(
            "Please select at least one skill."
        )

    else:

        # =====================================================
        # PROFILE SCORE
        # =====================================================

        readiness_score = min(
            100,
            len(selected_skills) * 8 +
            experience * 3 +
            (academic_score // 5)
        )

        st.subheader("📊 Career Readiness Score")

        st.progress(readiness_score)

        st.metric(
            "Readiness Score",
            f"{readiness_score}%"
        )

        # =====================================================
        # INPUT DATA
        # =====================================================

        input_df = pd.DataFrame([{
            'age': age,
            'education_level': education
        }])

        input_df['education_level'] = (
            education_encoder.transform(
                input_df['education_level']
            )
        )

        # =====================================================
        # SKILL ENCODING
        # =====================================================

        skills_encoded = mlb.transform(
            [selected_skills]
        )

        skills_df = pd.DataFrame(
            skills_encoded,
            columns=mlb.classes_
        )

        final_input = pd.concat(
            [
                input_df.reset_index(drop=True),
                skills_df
            ],
            axis=1
        )

        final_input = final_input.reindex(
            columns=training_columns,
            fill_value=0
        )

        # =====================================================
        # PREDICTION
        # =====================================================

        prediction = model.predict(
            final_input
        )

        predicted_role = (
            target_encoder.inverse_transform(
                prediction
            )[0]
        )

        st.success(
            f"🎯 Predicted Job Role: {predicted_role}"
        )

        # =====================================================
        # ROLE DETAILS
        # =====================================================

        details = role_details.get(
            predicted_role
        )

        if details:

            st.subheader(
                "📈 Skill Gap Analysis"
            )

            st.info(
                details['skill_gap']
            )

            st.subheader(
                "🎓 Recommended Training"
            )

            st.success(
                details['training']
            )

            st.subheader(
                "📚 Recommended Skills"
            )

            for skill in details['recommended']:
                st.write(
                    f"✅ {skill}"
                )

        # =====================================================
        # TOP 3 PREDICTIONS
        # =====================================================

        probabilities = model.predict_proba(
            final_input
        )[0]

        top_3_idx = (
            probabilities.argsort()[-3:][::-1]
        )

        st.subheader(
            "🏅 Top 3 Career Matches"
        )

        for idx in top_3_idx:

            role = (
                target_encoder
                .inverse_transform([idx])[0]
            )

            prob = (
                probabilities[idx] * 100
            )

            st.write(
                f"**{role}** : {prob:.2f}%"
            )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Built using Streamlit + Random Forest Machine Learning"
)

