import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(
    page_title="School Enrollment Predictor",
    page_icon="🎓",
    layout="centered",
)

@st.cache_data
def load_and_preprocess_data():
    """
    Loads the dataset and prepares it for encoding.
    """
    try:
        df = pd.read_csv('ind_sch_updated.csv')
    except FileNotFoundError:
        st.error("Error: 'ind_sch_updated.csv' not found. Please place the file in the same directory as app.py.")
        return None, None, None

    df_processed = df.drop(columns=['unit', 'note'])
    fiscal_years = df_processed['fiscal_year'].unique()
    school_types = df_processed['school_type'].unique()
    
    return df_processed, fiscal_years, school_types

@st.cache_resource
def train_model(df_processed):
    """
    Performs one-hot encoding and trains both models.
    Returns the trained Random Forest model, encoder, feature names, and performance metrics.
    """
    categorical_features = ['fiscal_year', 'school_type']
    target = 'value'
    
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_data = encoder.fit_transform(df_processed[categorical_features])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_features))
    final_df = pd.concat([df_processed[[target]].reset_index(drop=True), encoded_df], axis=1)
    
    X = final_df.drop(target, axis=1)
    y = final_df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    lr_model = LinearRegression().fit(X_train, y_train)
    lr_y_pred = lr_model.predict(X_test)
    lr_r2 = r2_score(y_test, lr_y_pred)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_y_pred))

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
    rf_y_pred = rf_model.predict(X_test)
    rf_r2 = r2_score(y_test, rf_y_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_y_pred))
    
    metrics = {
        "Linear Regression": {"R²": lr_r2, "RMSE": lr_rmse},
        "Random Forest": {"R²": rf_r2, "RMSE": rf_rmse}
    }
    
    return rf_model, encoder, X.columns.tolist(), metrics


st.title("🎓 School Enrollment Predictor")
st.markdown("Select a fiscal year and school type to predict the number of student enrollments in India.")

df_processed, fiscal_years, school_types = load_and_preprocess_data()

if df_processed is not None:
    model, encoder, feature_names, metrics = train_model(df_processed)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        selected_year = st.selectbox("Select Fiscal Year", options=fiscal_years)
        selected_school = st.selectbox("Select School Type", options=school_types)
        predict_button = st.button("Predict Enrollment", type="primary", use_container_width=True)

    st.markdown("---")

    if predict_button:
        input_data = pd.DataFrame([[selected_year, selected_school]], columns=['fiscal_year', 'school_type'])
        
        encoded_input = encoder.transform(input_data)
        input_df = pd.DataFrame(encoded_input, columns=encoder.get_feature_names_out(['fiscal_year', 'school_type']))
        
        input_df_aligned = input_df.reindex(columns=feature_names, fill_value=0)
        
        prediction = model.predict(input_df_aligned)[0]
        
        st.subheader("📈 Prediction Result")
        st.success(f"**The predicted enrollment is: {int(prediction):,}**")
        
        st.subheader("📊 Model Performance Comparison")
        st.markdown("The prediction was made using a **Random Forest Regressor**, which performed better than a standard Linear Regression model during testing.")
        
        metrics_df = pd.DataFrame(metrics).T
        metrics_df['R²'] = metrics_df['R²'].map('{:.4f}'.format)
        metrics_df['RMSE'] = metrics_df['RMSE'].map('{:,.2f}'.format)
        
        st.table(metrics_df)
        st.info("The **Random Forest** model was chosen for its higher R-squared ($R^2$) and lower Root Mean Squared Error (RMSE).")