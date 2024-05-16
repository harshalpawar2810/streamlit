import streamlit as st
import json
import requests
import logging

# Setting up logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

st.title('Income Classification System - Group 37')


# Sidebar
st.subheader('Predictors')

# Numeric values associated with options for other predictors
# workclass_mapping = {'Private': 0, 'Self-emp-not-inc': 1, 'Self-emp-inc': 2, 'Federal-gov': 3, 'Local-gov': 4,
#                      'State-gov': 5, 'Without-pay': 6, 'Never-worked': 7}
# education_mapping = {'Bachelors': 0, 'Some-college': 1, '11th': 2, 'HS-grad': 3, 'Prof-school': 4, 'Assoc-acdm': 5,
#                      'Assoc-voc': 6, '9th': 7, '7th-8th': 8, '12th': 9, 'Masters': 10, '1st-4th': 11, '10th': 12,
#                      'Doctorate': 13, '5th-6th': 14, 'Preschool': 15}
# marital_status_mapping = {'Married-civ-spouse': 0, 'Divorced': 1, 'Never-married': 2, 'Separated': 3, 'Widowed': 4,
#                           'Married-spouse-absent': 5, 'Married-AF-spouse': 6}
# occupation_mapping = {'Tech-support': 0, 'Craft-repair': 1, 'Other-service': 2, 'Sales': 3, 'Exec-managerial': 4,
#                       'Prof-specialty': 5, 'Handlers-cleaners': 6, 'Machine-op-inspct': 7, 'Adm-clerical': 8,
#                       'Farming-fishing': 9, 'Transport-moving': 10, 'Priv-house-serv': 11, 'Protective-serv': 12,
#                       'Armed-Forces': 13}
# relationship_mapping = {'Wife': 0, 'Own-child': 1, 'Husband': 2, 'Not-in-family': 3, 'Other-relative': 4,
#                         'Unmarried': 5}
# race_mapping = {'White': 0, 'Asian-Pac-Islander': 1, 'Amer-Indian-Eskimo': 2, 'Other': 3, 'Black': 4}
# sex_mapping = {'Female': 0, 'Male': 1}
# country_mapping = {'United-States': 0, 'Cambodia': 1, 'England': 2, 'Puerto-Rico': 3, 'Canada': 4, 'Germany': 5,
#                    'Outlying-US(Guam-USVI-etc)': 6, 'India': 7, 'Japan': 8, 'Greece': 9, 'South': 10, 'China': 11,
#                    'Cuba': 12, 'Iran': 13, 'Honduras': 14, 'Philippines': 15, 'Italy': 16, 'Poland': 17, 'Jamaica': 18,
#                    'Vietnam': 19, 'Mexico': 20, 'Portugal': 21, 'Ireland': 22, 'France': 23, 'Dominican-Republic': 24,
#                    'Laos': 25, 'Ecuador': 26, 'Taiwan': 27, 'Haiti': 28, 'Columbia': 29, 'Hungary': 30, 'Guatemala': 31,
#                    'Nicaragua': 32, 'Scotland': 33, 'Thailand': 34, 'Yugoslavia': 35, 'El-Salvador': 36,
#                    'Trinadad&Tobago': 37, 'Peru': 38, 'Hong': 39, 'Holand-Netherlands': 40}
workclass_mapping = {
    'Private': -0.05688482,
    'State-gov': 1.38143928,
    'Federal-gov': -3.61993761,
    'Self-emp-not-inc': 0.98524132,
    'Self-emp-inc': 0.51663858,
    'Local-gov': -1.83841122,
    'Without-pay': 1.72464158,
    'Never-worked': -0.79628508
}
education_scaled_values = {
    'HS-grad': 0.29559551,
    'Some-college': 0.78174432,
    '7th-8th': -0.87574156,
    '10th': -3.90360396,
    'Doctorate': 0.14855654,
    'Prof-school': 0.67268182,
    'Bachelors': -0.01250644,
    'Masters': 0.43085826,
    '11th': -2.73226689,
    'Assoc-acdm': -0.38959275,
    'Assoc-voc': -0.1905533,
    '1st-4th': -1.56092982,
    '5th-6th': -1.18384351,
    '12th': -2.04707863,
    '9th': -0.61524507,
    'Preschool': 0.556092
}
marital_status_scaled_values = {
    'Widowed': 1.46149159,
    'Divorced': -2.19140579,
    'Separated': 1.17211717,
    'Never-married': 0.82985988,
    'Married-civ-spouse': -0.12907113,
    'Married-spouse-absent': 0.41097081,
    'Married-AF-spouse': -0.89021749
}
occupation_scaled_values = {
    'Exec-managerial': -0.45011676,
    'Machine-op-inspct': 0.24021984,
    'Prof-specialty': 0.68021058,
    'Other-service': 0.40494287,
    'Adm-clerical': -2.16023601,
    'Craft-repair': -0.80499857,
    'Transport-moving': 1.09527946,
    'Handlers-cleaners': 0.05006106,
    'Sales': 0.90512068,
    'Farming-fishing': -0.17484904,
    'Tech-support': 1.00386059,
    'Protective-serv': 0.79778429,
    'Armed-Forces': -1.30517638,
    'Priv-house-serv': 0.55023887
}
relationship_scaled_values = {
    'Not-in-family': 0.01674572,
    'Unmarried': 1.42849409,
    'Own-child': 1.08469213,
    'Other-relative': 0.64145433,
    'Husband': -1.05120069,
    'Wife': 1.70940074
}
race_scaled_values = {
    'White': 0.36650348,
    'Black': -1.59436474,
    'Asian-Pac-Islander': -3.15079341,
    'Other': -0.49006103,
    'Amer-Indian-Eskimo': -5.81152579
}
sex_scaled_values = {
    'Female': -1.42233076,
    'Male': 0.70307135
}
country_scaled_values = {
    'United-States': 0.21504802,
    'Mexico': -0.95865955,
    'Greece': -3.19682706,
    'Vietnam': 0.28833596,
    'China': -7.20975976,
    'Taiwan': -0.01665313,
    'India': -1.86661007,
    'Philippines': -0.54442281,
    'Trinadad&Tobago': 0.13985629,
    'Canada': -8.38346733,
    'South': -0.0981999,
    'Holand-Netherlands': -2.55088917,
    'Puerto-Rico': -0.26852662,
    'Poland': -0.44950549,
    'Iran': -1.71813039,
    'England': -4.02958583,
    'Germany': -3.44870055,
    'Italy': -1.4422342,
    'Japan': -1.1903607,
    'Hong': -2.18857703,
    'Honduras': -2.36406828,
    'Cuba': -5.73106309,
    'Ireland': -1.57689657,
    'Cambodia': -10.38993368,
    'Peru': -0.64255828,
    'Nicaragua': -0.84941191,
    'Dominican-Republic': -5.20329341,
    'Haiti': -2.75060415,
    'El-Salvador': -4.37053463,
    'Hungary': -2.02311948,
    'Columbia': -6.37700098,
    'Guatemala': -2.9651259,
    'Jamaica': -1.31355883,
    'Ecuador': -4.7570705,
    'France': -3.72459674,
    'Yugoslavia': 0.35981414,
    'Scotland': -0.18211068,
    'Portugal': -0.35760193,
    'Laos': -1.0721925,
    'Thailand': 0.0626592,
    'Outlying-US(Guam-USVI-etc)': -0.74413779
}





# Input widgets for predictor variables
age = st.sidebar.slider('Age', min_value=1, max_value=100, step=1)
fnlwgt = st.sidebar.number_input('Fnlwgt', min_value=0)
capital_gain = st.sidebar.number_input('Capital Gain', min_value=0)
capital_loss = st.sidebar.number_input('Capital Loss', min_value=0)
hours_per_week = st.sidebar.number_input('Hours per Week', min_value=0)

workclass = st.sidebar.selectbox('Workclass', list(workclass_mapping.keys()))
education = st.sidebar.selectbox('Education', list(education_scaled_values.keys()))
marital_status = st.sidebar.selectbox('Marital Status', list(marital_status_scaled_values.keys()))
occupation = st.sidebar.selectbox('Occupation', list(occupation_scaled_values.keys()))
relationship = st.sidebar.selectbox('Relationship', list(relationship_scaled_values.keys()))
race = st.sidebar.selectbox('Race', list(race_scaled_values.keys()))
sex = st.sidebar.selectbox('Sex', list(sex_scaled_values.keys()))
country = st.sidebar.selectbox('Country', list(country_scaled_values.keys()))

# Create a dictionary of predictor values
predictors_dict = {
    'age': age,
    'fnlwgt': fnlwgt,
    'capital_gain': capital_gain,
    'capital_loss': capital_loss,
    'hours_per_week': hours_per_week,
    'workclass': workclass_mapping[workclass],
    'education': education_scaled_values[education],
    'marital_status': marital_status_scaled_values[marital_status],
    'occupation': occupation_scaled_values[occupation],
    'relationship': relationship_scaled_values[relationship],
    'race': race_scaled_values[race],
    'sex': sex_scaled_values[sex],
    'country': country_scaled_values[country]
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

