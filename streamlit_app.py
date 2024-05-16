import streamlit as st
import json
import requests
import logging

# Setting up logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

st.title('Income Classification System')

# Sidebar
st.subheader('Predictors')

# Numeric values associated with options for other predictors
workclass_mapping = {'Private': 0, 'Self-emp-not-inc': 1, 'Self-emp-inc': 2, 'Federal-gov': 3, 'Local-gov': 4,
                     'State-gov': 5, 'Without-pay': 6, 'Never-worked': 7}
education_mapping = {'Bachelors': 0, 'Some-college': 1, '11th': 2, 'HS-grad': 3, 'Prof-school': 4, 'Assoc-acdm': 5,
                     'Assoc-voc': 6, '9th': 7, '7th-8th': 8, '12th': 9, 'Masters': 10, '1st-4th': 11, '10th': 12,
                     'Doctorate': 13, '5th-6th': 14, 'Preschool': 15}
marital_status_mapping = {'Married-civ-spouse': 0, 'Divorced': 1, 'Never-married': 2, 'Separated': 3, 'Widowed': 4,
                          'Married-spouse-absent': 5, 'Married-AF-spouse': 6}
occupation_mapping = {'Tech-support': 0, 'Craft-repair': 1, 'Other-service': 2, 'Sales': 3, 'Exec-managerial': 4,
                      'Prof-specialty': 5, 'Handlers-cleaners': 6, 'Machine-op-inspct': 7, 'Adm-clerical': 8,
                      'Farming-fishing': 9, 'Transport-moving': 10, 'Priv-house-serv': 11, 'Protective-serv': 12,
                      'Armed-Forces': 13}
relationship_mapping = {'Wife': 0, 'Own-child': 1, 'Husband': 2, 'Not-in-family': 3, 'Other-relative': 4,
                        'Unmarried': 5}
race_mapping = {'White': 0, 'Asian-Pac-Islander': 1, 'Amer-Indian-Eskimo': 2, 'Other': 3, 'Black': 4}
sex_mapping = {'Female': 0, 'Male': 1}
country_mapping = {'United-States': 0, 'Cambodia': 1, 'England': 2, 'Puerto-Rico': 3, 'Canada': 4, 'Germany': 5,
                   'Outlying-US(Guam-USVI-etc)': 6, 'India': 7, 'Japan': 8, 'Greece': 9, 'South': 10, 'China': 11,
                   'Cuba': 12, 'Iran': 13, 'Honduras': 14, 'Philippines': 15, 'Italy': 16, 'Poland': 17, 'Jamaica': 18,
                   'Vietnam': 19, 'Mexico': 20, 'Portugal': 21, 'Ireland': 22, 'France': 23, 'Dominican-Republic': 24,
                   'Laos': 25, 'Ecuador': 26, 'Taiwan': 27, 'Haiti': 28, 'Columbia': 29, 'Hungary': 30, 'Guatemala': 31,
                   'Nicaragua': 32, 'Scotland': 33, 'Thailand': 34, 'Yugoslavia': 35, 'El-Salvador': 36,
                   'Trinadad&Tobago': 37, 'Peru': 38, 'Hong': 39, 'Holand-Netherlands': 40}

# Input widgets for predictor variables
age = st.sidebar.slider('Age', min_value=1, max_value=100, step=1)
fnlwgt = st.sidebar.number_input('Fnlwgt', min_value=0)
capital_gain = st.sidebar.number_input('Capital Gain', min_value=0)
capital_loss = st.sidebar.number_input('Capital Loss', min_value=0)
hours_per_week = st.sidebar.number_input('Hours per Week', min_value=0)

workclass = st.sidebar.selectbox('Workclass', list(workclass_mapping.keys()))
education = st.sidebar.selectbox('Education', list(education_mapping.keys()))
marital_status = st.sidebar.selectbox('Marital Status', list(marital_status_mapping.keys()))
occupation = st.sidebar.selectbox('Occupation', list(occupation_mapping.keys()))
relationship = st.sidebar.selectbox('Relationship', list(relationship_mapping.keys()))
race = st.sidebar.selectbox('Race', list(race_mapping.keys()))
sex = st.sidebar.selectbox('Sex', list(sex_mapping.keys()))
country = st.sidebar.selectbox('Country', list(country_mapping.keys()))

# Create a dictionary of predictor values
predictors_dict = {
    'age': age,
    'fnlwgt': fnlwgt,
    'capital_gain': capital_gain,
    'capital_loss': capital_loss,
    'hours_per_week': hours_per_week,
    'workclass': workclass_mapping[workclass],
    'education': education_mapping[education],
    'marital_status': marital_status_mapping[marital_status],
    'occupation': occupation_mapping[occupation],
    'relationship': relationship_mapping[relationship],
    'race': race_mapping[race],
    'sex': sex_mapping[sex],
    'country': country_mapping[country]
}

# Prediction button
if st.button('Predict'):
    try:
        logger.info("Preparing data for prediction...")
        
        # Prepare data for prediction
        # Prepare data for prediction
        payload = json.dumps({'dataframe_records': [predictors_dict]})

        
        logger.info(f"Payload: {payload}")  # Logging payload before sending

        logger.info("Sending request to the model API...")

        # Send request to the model API
        response = requests.post(
            url=f"http://138.197.72.58:5001/invocations",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        logger.info("Received response from the model API.")

        # Check if the request was successful
        if response.status_code == 200:
            # Get prediction from response
            prediction = response.json().get('predictions')
            st.success(f"Prediction: {prediction}")
        else:
            st.error(f"Failed to get prediction. Please try again later.  {response.status_code} {response.content}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

