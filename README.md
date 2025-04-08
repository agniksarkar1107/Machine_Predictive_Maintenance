# Machine Predictive Maintenance System

A machine learning solution for predicting equipment failures and providing maintenance recommendations using a stacked ensemble of XGBoost and Random Forest models.

## Features

- **Advanced Machine Learning Models**:
  - XGBoost Classifier
  - Random Forest Classifier
  - Stacking Ensemble with Logistic Regression meta-learner
  - Failure Type Classification

- **Interactive Web Interface**:
  - Real-time predictions
  - Visual probability gauge
  - Detailed maintenance recommendations
  - Parameter input validation
  - Downloadable prediction reports

- **Comprehensive Analysis**:
  - Failure probability estimation
  - Specific failure type prediction
  - Feature importance visualization
  - Model performance metrics

## Dataset

The system uses a predictive maintenance dataset with the following features:

- Machine type (L, M, H)
- Air temperature
- Process temperature
- Rotational speed (rpm)
- Torque (Nm)
- Tool wear (min)
- Target (failure/no failure)
- Failure type

## Model Performance

The stacked ensemble achieves:
- Training Accuracy: 99.61%
- Test Accuracy: 98.60%
- High precision and recall for both failure and non-failure cases

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/machine-predictive-maintenance.git
cd machine-predictive-maintenance
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Train the models:
```bash
python train_model_stacking.py
```

2. Start the web application:
```bash
streamlit run app_stacking.py
```

3. Access the web interface at `http://localhost:8501`

## Project Structure

```
machine-predictive-maintenance/
├── README.md
├── requirements.txt
├── train_model_stacking.py    # Model training script
├── app_stacking.py           # Streamlit web application
├── predictive_maintenance.csv # Dataset
├── stacked_model.joblib      # Trained stacked model
├── failure_type_model.joblib # Failure type classifier
└── label_encoder.joblib      # Label encoder for categorical variables
```

## Model Details

### Feature Importance
The top 5 most important features for prediction:
1. Torque
2. Rotational Speed
3. Tool Wear
4. Temperature Difference
5. Air Temperature

### Maintenance Recommendations

The system provides specific maintenance recommendations based on:
- Failure probability
- Predicted failure type
- Current machine parameters

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset source: [Machine Predictive Maintenance Classification Dataset](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification)
- Built with Streamlit, XGBoost, and scikit-learn 