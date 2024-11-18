import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def load_data(uploaded_file):
    """Loads data from an uploaded file."""
    try:
        if uploaded_file.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload an Excel or CSV file.")
            return None
        return data
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def identify_numeric_columns(data):
    """Identifies relevant numeric columns for analysis."""
    numeric_columns = {}
    
    for column in data.columns:
        # Convert column name to string and then check
        col_name = str(column).lower()
        
        # Skip ID and date-like columns
        skip_keywords = {'id', 'number', 'num', 'date', 'time', 'timestamp'}
        if any(keyword in col_name for keyword in skip_keywords):
            continue
            
        # Check if column is numeric
        try:
            if pd.api.types.is_numeric_dtype(data[column]):
                # Check if column has meaningful variance and is not all zeros
                if data[column].std() > 0 and not (data[column] == 0).all():
                    numeric_columns[column] = {
                        'mean': float(data[column].mean()),
                        'std': float(data[column].std()),
                        'missing': int(data[column].isnull().sum())
                    }
        except Exception as e:
            continue
    
    return numeric_columns

def preprocess_data(data, columns_to_use):
    """Preprocesses numeric data for anomaly detection."""
    processed_data = data.copy()
    
    for column in columns_to_use:
        try:
            # Convert to numeric, coerce errors to NaN
            processed_data[column] = pd.to_numeric(processed_data[column], errors='coerce')
            
            # Fill missing values with median
            median_value = processed_data[column].median()
            processed_data[column] = processed_data[column].fillna(median_value)
            
            # Remove infinite values
            processed_data[column] = processed_data[column].replace([np.inf, -np.inf], median_value)
        except Exception as e:
            st.warning(f"Error processing column {column}: {str(e)}")
            continue
    
    return processed_data

# Streamlit UI
st.title("Financial Data Anomaly Detection")
st.write("This tool analyzes numeric features to detect anomalies in financial data.")

# File upload
uploaded_file = st.file_uploader("Upload your financial data file", type=["xlsx", "csv"])

if uploaded_file is not None:
    # Load the data
    financial_data = load_data(uploaded_file)
    
    if financial_data is not None:
        st.write("### Data Preview")
        st.dataframe(financial_data.head())
        
        # Identify numeric columns
        numeric_columns = identify_numeric_columns(financial_data)
        
        if not numeric_columns:
            st.error("No suitable numeric columns found in the dataset.")
        else:
            # Feature selection
            st.sidebar.header("Feature Selection")
            
            # Show column statistics
            st.sidebar.subheader("Available Numeric Features")
            selected_features = []
            for col in numeric_columns.keys():
                if st.sidebar.checkbox(f"{col}", value=True):
                    selected_features.append(col)
            
            # Show column statistics
            if st.sidebar.checkbox("Show Feature Statistics"):
                stats_df = pd.DataFrame.from_dict(numeric_columns, orient='index')
                st.sidebar.dataframe(stats_df)
            
            if st.button("Detect Anomalies"):
                if len(selected_features) < 2:
                    st.error("Please select at least 2 features for analysis.")
                else:
                    try:
                        # Preprocess the data
                        processed_data = preprocess_data(financial_data, selected_features)
                        
                        # Scale the features
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(processed_data[selected_features])
                        
                        # Run Isolation Forest
                        iso_forest = IsolationForest(
                            n_estimators=150,
                            contamination=0.05,
                            max_samples='auto',
                            bootstrap=True,
                            n_jobs=-1,
                            random_state=42
                        )
                        
                        processed_data['anomaly'] = iso_forest.fit_predict(X_scaled)
                        processed_data['anomaly'] = processed_data['anomaly'].map({1: 0, -1: 1})
                        
                        # Display results
                        st.write("### Analysis Results")
                        total_samples = len(processed_data)
                        anomaly_count = processed_data['anomaly'].sum()
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Samples", total_samples)
                        with col2:
                            st.metric("Anomalies Detected", anomaly_count)
                        with col3:
                            st.metric("Anomaly Percentage", f"{(anomaly_count/total_samples)*100:.2f}%")
                        
                        # Visualizations
                        st.write("### Feature Correlations and Anomalies")
                        
                        # Create correlation matrix visualization
                        corr_matrix = processed_data[selected_features].corr()
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
                        plt.title('Feature Correlation Matrix')
                        st.pyplot(fig)
                        
                        # Create scatter plots for pairs of features
                        st.write("### Feature Pair Analysis")
                        for i in range(0, len(selected_features), 2):
                            if i + 1 < len(selected_features):
                                x_feature = selected_features[i]
                                y_feature = selected_features[i + 1]
                                
                                fig, ax = plt.subplots(figsize=(10, 6))
                                sns.scatterplot(
                                    data=processed_data,
                                    x=x_feature,
                                    y=y_feature,
                                    hue='anomaly',
                                    palette={0: 'blue', 1: 'red'},
                                    alpha=0.6
                                )
                                plt.title(f'Anomaly Detection: {x_feature} vs {y_feature}')
                                plt.xlabel(x_feature)
                                plt.ylabel(y_feature)
                                st.pyplot(fig)
                        
                        # Feature importance
                        feature_importance = pd.DataFrame({
                            'Feature': selected_features,
                            'Importance': np.abs(np.mean(X_scaled, axis=0))
                        }).sort_values('Importance', ascending=False)
                        
                        st.write("### Feature Importance")
                        fig, ax = plt.subplots(figsize=(10, 4))
                        sns.barplot(data=feature_importance, x='Importance', y='Feature')
                        plt.title('Feature Importance in Anomaly Detection')
                        st.pyplot(fig)
                        
                        # Download options
                        st.write("### Download Results")
                        anomalies = processed_data[processed_data['anomaly'] == 1]
                        
                        # Prepare download data with original values
                        download_cols = selected_features + ['anomaly']
                        csv_anomalies = anomalies[download_cols].to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "Download Anomalies Report",
                            data=csv_anomalies,
                            file_name='anomalies_report.csv',
                            mime='text/csv'
                        )
                        
                        # Statistical summary
                        st.write("### Statistical Summary of Anomalies")
                        st.write("Statistics for Detected Anomalies:")
                        st.dataframe(anomalies[selected_features].describe())
                        
                    except Exception as e:
                        st.error(f"An error occurred during analysis: {str(e)}")
                        st.error("Please check your data and selected features.")

else:
    st.info("Please upload a financial data file to begin analysis.")
