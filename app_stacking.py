import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# Define the StackedModel class so it can be loaded from the joblib file
class StackedModel:
    def __init__(self, stacking_model, scaler, failure_type_model):
        self.stacking_model = stacking_model
        self.scaler = scaler
        self.failure_type_model = failure_type_model
        self.feature_names = []
        
    def predict(self, X):
        # Scale the input
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        return self.stacking_model.predict(X_scaled)
    
    def predict_proba(self, X):
        # Scale the input
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        return self.stacking_model.predict_proba(X_scaled)
    
    def predict_failure_type(self, X):
        # Scale the input
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        return self.failure_type_model.predict(X_scaled)

# Load the models and preprocessing objects
@st.cache_resource
def load_models():
    stacked_model = joblib.load('stacked_model.joblib')
    le = joblib.load('label_encoder.joblib')
    return stacked_model, le

# Set page config
st.set_page_config(page_title="Machine Failure Prediction", layout="wide")

# Title
st.title("Machine Predictive Maintenance System")
st.write("Enter the machine parameters to predict failure probability and type")

# Initialize session state for showing results
if 'show_results' not in st.session_state:
    st.session_state.show_results = False

# Load models
try:
    model, le = load_models()
    
    # Create input form
    with st.form("prediction_form"):
        st.subheader("Machine Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            machine_type = st.selectbox("Machine Type", ['L', 'M', 'H'])
            
            tool_wear = st.slider(
                "Tool Wear [min]", 
                min_value=0, 
                max_value=250, 
                value=0,
                help="Tool wear in minutes"
            )
            
        with col2:
            air_temp = st.slider(
                "Air Temperature [°C]", 
                min_value=20.0, 
                max_value=35.0, 
                value=25.0, 
                step=0.1,
                help="Air temperature in degrees Celsius"
            )
            
            process_temp = st.slider(
                "Process Temperature [°C]", 
                min_value=30.0, 
                max_value=45.0, 
                value=35.0, 
                step=0.1,
                help="Process temperature in degrees Celsius"
            )
            
        with col3:
            rot_speed = st.slider(
                "Rotational Speed [rpm]", 
                min_value=1000, 
                max_value=3000, 
                value=1500,
                help="Rotational speed in revolutions per minute"
            )
            
            torque = st.slider(
                "Torque [Nm]", 
                min_value=3.0, 
                max_value=77.0, 
                value=40.0, 
                step=0.1,
                help="Torque in Newton meters"
            )
            
        submit_button = st.form_submit_button("Predict")
        
        if submit_button:
            st.session_state.show_results = True
    
    # If form submitted, make prediction
    if st.session_state.show_results:
        # Map type to encoded value
        machine_type_mapped = {'L': 0, 'M': 1, 'H': 2}.get(machine_type, 0)
        
        # Calculate temperature difference
        temp_diff = process_temp - air_temp
        
        # Prepare input data
        input_data = pd.DataFrame({
            'Type': [machine_type_mapped],
            'Rotational_speed__rpm_': [rot_speed],
            'Torque__Nm_': [torque],
            'Tool_wear__min_': [tool_wear],
            'Air_temperature__C_': [air_temp],
            'Process_temperature__C_': [process_temp],
            'Temperature_difference__C_': [temp_diff]
        })
        
        # Make predictions
        failure_prediction = model.predict(input_data)[0]
        failure_prob = model.predict_proba(input_data)[0]
        
        # Only predict failure type if failure is predicted
        if failure_prediction == 1:
            failure_type_prediction = model.predict_failure_type(input_data)[0]
            failure_types = ["No Failure", "Heat Dissipation Failure", "Power Failure", 
                             "Overstrain Failure", "Tool Wear Failure", "Random Failures"]
            failure_type_name = failure_types[failure_type_prediction]
        
        # Display results section
        st.write("---")
        st.subheader("Prediction Results")
        
        # Display in columns with adjusted widths
        col1, col2 = st.columns([1, 1.2])  # Giving more width to the recommendation column
        
        with col1:
            # Probability gauge chart with adjusted size
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=failure_prob[1] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Failure Probability (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "red" if failure_prediction == 1 else "green"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            
            # Set chart height and width
            fig.update_layout(
                height=350,
                width=450,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            st.plotly_chart(fig)
            
        with col2:
            # Add some padding
            st.write("")
            
            # Display prediction results
            if failure_prediction == 1:
                st.error("⚠️ MACHINE FAILURE PREDICTED!")
                st.markdown(f"""
                * **Failure Probability**: {failure_prob[1]:.2%}
                * **Predicted Failure Type**: {failure_type_name}
                """)
                
                # Show recommendations
                st.subheader("Recommended Actions")
                
                if failure_type_prediction == 1:  # Heat Dissipation
                    st.markdown("""
                    * Check cooling system components
                    * Inspect ventilation system
                    * Reduce operational temperature if possible
                    * Schedule maintenance for heat management systems
                    """)
                elif failure_type_prediction == 2:  # Power Failure
                    st.markdown("""
                    * Inspect electrical connections and components
                    * Check power supply stability
                    * Verify circuit protection devices
                    * Consider backup power solutions
                    """)
                elif failure_type_prediction == 3:  # Overstrain
                    st.markdown("""
                    * Reduce operational load immediately
                    * Inspect mechanical components for wear
                    * Check alignment of moving parts
                    * Review operational parameters
                    """)
                elif failure_type_prediction == 4:  # Tool Wear
                    st.markdown("""
                    * Replace tool components
                    * Inspect cutting edges and surfaces
                    * Check lubrication systems
                    * Review material quality being processed
                    """)
                else:  # Random Failures
                    st.markdown("""
                    * Perform comprehensive system diagnostics
                    * Check for intermittent issues
                    * Review maintenance history
                    * Inspect all critical components
                    """)
            else:
                st.success("✅ NO FAILURE PREDICTED")
                st.markdown(f"""
                * **Failure Probability**: {failure_prob[1]:.2%}
                * **Status**: Machine is operating within normal parameters
                """)
                
                # Show recommendations for healthy machine
                st.subheader("Maintenance Recommendations")
                st.markdown("""
                * Continue regular maintenance schedule
                * Monitor critical parameters periodically
                * Keep records of performance trends
                * Perform preventive inspections as scheduled
                """)
        
        # Add space after prediction results
        st.write("")
        st.write("")
        
        # Show input parameters for reference
        with st.expander("View Input Parameters"):
            st.dataframe(input_data)
            
        # Option to download the prediction
        csv = input_data.to_csv(index=False)
        st.download_button(
            label="Download Prediction Data",
            data=csv,
            file_name="machine_prediction.csv",
            mime="text/csv"
        )

except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.warning("Please ensure you have trained the models first by running train_model_stacking.py")
    
# Footer
st.markdown("---")
st.markdown("#### About")
st.markdown("""
This predictive maintenance application uses a stacked ensemble of machine learning models 
(XGBoost and Random Forest) to predict potential machine failures before they occur.
The model analyzes various sensor readings and operational parameters to identify patterns
that indicate a high risk of failure.
""")

# Sidebar with information
with st.sidebar:
    st.title("Information")
    st.info("""
    **Machine Types**:
    - L: Low capacity
    - M: Medium capacity
    - H: High capacity
    
    **Parameter Ranges**:
    - Air Temperature: 20-35°C
    - Process Temperature: 30-45°C
    - Rotational Speed: 1000-3000 rpm
    - Torque: 3-77 Nm
    - Tool Wear: 0-250 min
    """)
    
    st.write("---")
    st.subheader("Model Performance")
    st.write("Stacked model accuracy: 99.55%")
    
    # Show sample images of model evaluation
    try:
        st.image("confusion_matrix_stacking_ensemble.png", caption="Confusion Matrix")
        st.image("feature_importance.png", caption="Feature Importance")
    except:
        st.write("Model evaluation visualizations not available. Run the training script first.") 