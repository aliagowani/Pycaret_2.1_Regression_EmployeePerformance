# Initiate local directory
# import os
# os.getcwd()
# os.chdir('/Users/aliagowani/Documents/Home Documents/School Papers/Northwestern University - NU/Other Documents/Pycaret - Wine')

# import pandas as pd
# import numpy as np

# wine_df = pd.read_csv('winequality-red.csv', sep=';')

# wine_df.head()

# wine_df.quality = np.where(wine_df.quality >= 6,'Good', 'Bad')

# wine_df.head()

# from pycaret.classification import *
# #exp_clf01 = setup(data = wine_df, target = 'quality', session_id = 123)

# exp_clf01 = setup(data = wine_df, target = 'quality', session_id = 123, normalize = True, transformation = True, silent=True)

# best = compare_models()

# et_model = create_model('et')

# evaluate_model(et_model)

# predict_model(et_model)

# save_model(et_model, model_name = 'extra_tree_model')

from numpy.lib.shape_base import column_stack
from pandas.core.frame import DataFrame
from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np


def predict_quality(model, df):
    predictions_data = predict_model(estimator = model, data = df)
    return predictions_data['Label'][0]
    
model = load_model('cat_boost_class_model')

st.title('Wine Quality Classifier Web App')
st.write('This is a web app to classify the quality of your wine based on\
         several features that you can see in the sidebar. Please adjust the\
         value of each feature. After that, click on the Predict button at the bottom to\
         see the prediction of the classifier.')

friday = st.sidebar.slider(label = 'Friday', min_value = 0.0,
                          max_value = 100.0 ,
                          value = 90.0,
                          step = 0.1)

sunday = st.sidebar.slider(label = 'Sunday', min_value = 0.0,
                          max_value = 100.0 ,
                          value = 90.0,
                          step = 0.1)
                          
thursday = st.sidebar.slider(label = 'Thursday', min_value = 0.0,
                          max_value = 100.0 ,
                          value = 90.0,
                          step = 0.1)                    

tuesday = st.sidebar.slider(label = 'Tuesday', min_value = 0.0,
                          max_value = 100.0 ,
                          value = 90.0,
                          step = 0.1)

wednesday = st.sidebar.slider(label = 'Wednesday', min_value = 0.0,
                          max_value = 100.0 ,
                          value = 90.0,
                          step = 0.1)
                          
saturday = st.sidebar.slider(label = 'Saturday', min_value = 0.0,
                          max_value = 100.0 ,
                          value = 90.0,
                          step = 0.1) 

ncoachingid = st.sidebar.slider(label = 'ncoachingid', min_value = 0.0,
                          max_value = 100.0 ,
                          value = 90.0,
                          step = 0.1) 

total_coaching_improved = st.sidebar.slider(label = 'total coaching improved', min_value = 0.0,
                          max_value = 100.0 ,
                          value = 90.0,
                          step = 0.1) 

monday = st.sidebar.slider(label = 'Monday', min_value = 0.0,
                          max_value = 100.0 ,
                          value = 90.0,
                          step = 0.1) 

weekbefore = st.sidebar.slider(label = 'weekbefore', min_value = 0.0,
                          max_value = 100.0 ,
                          value = 90.0,
                          step = 0.1)                          

csat = st.sidebar.slider(label = 'csat', min_value = 0.0,
                          max_value = 100.0 ,
                          value = 90.0,
                          step = 0.1) 

#actualvalue_class = st.sidebar.selectbox('Actual_Class', ['1', '0']) 

Wednesday_multiply_Tuesday = wednesday * tuesday
st.sidebar.write('Alcohol Multiplier: ', Wednesday_multiply_Tuesday)

Sunday_multiply_Friday = sunday * friday
st.sidebar.write('Alcohol Multiplier: ', Sunday_multiply_Friday)



features = {'Friday': friday, 'Sunday': sunday,
            'Thursday': thursday, 'Tuesday': tuesday,
            'Wednesday': wednesday, 'Saturday': saturday,
            'ncoachingid': ncoachingid, 
            'total coaching improved': total_coaching_improved,
            'Monday': monday,
            'weekbefore': weekbefore,
            'csat': csat
            }
 

features_df  = pd.DataFrame([features])

st.table(features_df.transpose())  

prediction = predict_quality(model, features_df)
st.write(' Based on feature values, your wine quality is '+ str(prediction))

if st.button('Predict'):
    prediction = predict_quality(model, features_df)
    st.write(' Based on feature values, your wine quality is '+ str(prediction))

#streamlit run pycaret_wine.py

# data = pd.read_csv('pred_final_blender.csv')

# st.line_chart(data['pH'])

# import altair as alt

# #source = data.cars()

# brush = alt.selection(type='interval')

# points = alt.Chart(data).mark_point().encode(
#     x='pH',
#     y='quality'
#     )


# points