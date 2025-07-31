# %%
"""
London Energy Consumption Prediction
===================================

This script implements a comprehensive machine learning solution for predicting
daily electrical consumption in London boroughs using historical weather and
energy consumption data.

Project Overview:
- Data Integration: Weather (1979-2021) + Energy (2011-2014) datasets
- Advanced Imputation: Deep learning for missing cloud cover data
- Predictive Modeling: Regression model for energy consumption forecasting
- Business Insights: Actionable recommendations for energy companies

Author: [Your Name]
Date: [Current Date]

Cursor Workaround: Using # %% cell markers for Jupyter-style workflow
Reference: https://www.trymito.io/blog/using-cursor-with-jupyter-notebooks-a-workaround-for-data-scientists
"""

# %%
# =============================================================================
# IMPORTS AND CONFIGURATION
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    mean_squared_error, 
    mean_absolute_percentage_error
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.inspection import permutation_importance
import folium
from folium.plugins import MarkerCluster
import matplotlib.colors as colors
import warnings
import os
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# %%
# =============================================================================
# CONFIGURATION AND SETTINGS
# =============================================================================

class Config:
    """Configuration settings for the project."""
    
    # File paths (update these with your actual file paths)
    WEATHER_DATA_PATH = "london_weather_data.csv"
    ENERGY_DATA_PATH = "london_energy_data.csv"
    
    # Model parameters
    RANDOM_SEED = 42
    TRAIN_TEST_SPLIT = 0.8
    BATCH_SIZE = 64
    EPOCHS = 30
    LEARNING_RATE = 0.01
    
    # Visualization settings
    FIGURE_SIZE = (12, 8)
    DPI = 300
    
    # Output directories
    OUTPUT_DIR = "outputs"
    PLOTS_DIR = "plots"
    MODELS_DIR = "models"

# Create output directories
for directory in [Config.OUTPUT_DIR, Config.PLOTS_DIR, Config.MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

print("Configuration loaded and output directories created")

# %%
# =============================================================================
# DATA LOADING AND VALIDATION
# =============================================================================

def load_weather_data(file_path):
    """
    Load and validate weather dataset.
    
    Args:
        file_path (str): Path to weather data CSV file
        
    Returns:
        pd.DataFrame: Loaded and validated weather data
    """
    print("Loading weather data...")
    
    try:
        df = pd.read_csv(file_path)
        print(f"Weather data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"Error: Weather data file not found at {file_path}")
        print("Please download the dataset from: https://www.kaggle.com/datasets/emmanuelfwerr/london-weather-data")
        return None
    except Exception as e:
        print(f"Error loading weather data: {e}")
        return None

def load_energy_data(file_path):
    """
    Load and validate energy dataset.
    
    Args:
        file_path (str): Path to energy data CSV file
        
    Returns:
        pd.DataFrame: Loaded and validated energy data
    """
    print("Loading energy data...")
    
    try:
        df = pd.read_csv(file_path)
        print(f"Energy data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"Error: Energy data file not found at {file_path}")
        print("Please download the dataset from: https://www.kaggle.com/datasets/emmanuelfwerr/london-homes-energy-data")
        return None
    except Exception as e:
        print(f"Error loading energy data: {e}")
        return None

# %%
# Load the datasets
weather_df = load_weather_data(Config.WEATHER_DATA_PATH)
energy_df = load_energy_data(Config.ENERGY_DATA_PATH)

# Display basic information about the datasets
if weather_df is not None:
    print("\nWeather dataset info:")
    print(weather_df.info())
    print("\nWeather dataset head:")
    print(weather_df.head())

if energy_df is not None:
    print("\nEnergy dataset info:")
    print(energy_df.info())
    print("\nEnergy dataset head:")
    print(energy_df.head())

# %%
# =============================================================================
# DATA VALIDATION
# =============================================================================

def validate_weather_data(df):
    """
    Validate weather dataset for required columns and data quality.
    
    Args:
        df (pd.DataFrame): Weather dataset to validate
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    print("Validating weather data...")
    
    required_columns = ['date', 'mean_temp', 'max_temp', 'min_temp', 'precipitation']
    
    # Check for required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        return False
    
    # Check for reasonable temperature ranges (London climate)
    if 'mean_temp' in df.columns:
        temp_range = df['mean_temp'].describe()
        if temp_range['min'] < -20 or temp_range['max'] > 40:
            print("Warning: Temperature values seem outside normal London range")
    
    # Check for missing values
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        print("Missing values detected:")
        print(missing_counts[missing_counts > 0])
    
    print("Weather data validation completed")
    return True

def validate_energy_data(df):
    """
    Validate energy dataset for required columns and data quality.
    
    Args:
        df (pd.DataFrame): Energy dataset to validate
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    print("Validating energy data...")
    
    required_columns = ['Date', 'MWH', 'Borough']
    
    # Check for required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        return False
    
    # Check for reasonable energy consumption values
    if 'MWH' in df.columns:
        energy_range = df['MWH'].describe()
        if energy_range['min'] < 0:
            print("Warning: Negative energy consumption values detected")
    
    print("Energy data validation completed")
    return True

# %%
# Validate the datasets
if weather_df is not None:
    weather_valid = validate_weather_data(weather_df)
else:
    weather_valid = False

if energy_df is not None:
    energy_valid = validate_energy_data(energy_df)
else:
    energy_valid = False

print(f"Weather data valid: {weather_valid}")
print(f"Energy data valid: {energy_valid}")

# %%
# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def engineer_weather_features(df):
    """
    Engineer features from weather dataset.
    
    Args:
        df (pd.DataFrame): Raw weather data
        
    Returns:
        pd.DataFrame: Weather data with engineered features
    """
    print("Engineering weather features...")
    
    # Create a copy to avoid modifying original data
    df_engineered = df.copy()
    
    # Extract date components
    df_engineered['year'] = df_engineered['date'].astype(str).str[:4].astype(int)
    df_engineered['month'] = df_engineered['date'].astype(str).str[4:6].astype(int)
    df_engineered['day'] = df_engineered['date'].astype(str).str[6:].astype(int)
    
    # Create seasonal features
    df_engineered['season'] = pd.cut(df_engineered['month'], 
                                   bins=[0, 3, 6, 9, 12], 
                                   labels=['Winter', 'Spring', 'Summer', 'Autumn'])
    
    # Create temperature difference features
    df_engineered['temp_range'] = df_engineered['max_temp'] - df_engineered['min_temp']
    
    # Create lag features for time series analysis
    for lag in [1, 7, 30]:
        df_engineered[f'mean_temp_lag_{lag}'] = df_engineered['mean_temp'].shift(lag)
        df_engineered[f'precipitation_lag_{lag}'] = df_engineered['precipitation'].shift(lag)
    
    # Create rolling averages
    for window in [7, 30]:
        df_engineered[f'mean_temp_rolling_{window}'] = df_engineered['mean_temp'].rolling(window=window).mean()
        df_engineered[f'precipitation_rolling_{window}'] = df_engineered['precipitation'].rolling(window=window).mean()
    
    print("Weather feature engineering completed")
    return df_engineered

def engineer_energy_features(df):
    """
    Engineer features from energy dataset.
    
    Args:
        df (pd.DataFrame): Raw energy data
        
    Returns:
        pd.DataFrame: Energy data with engineered features
    """
    print("Engineering energy features...")
    
    # Create a copy to avoid modifying original data
    df_engineered = df.copy()
    
    # Convert Date to datetime
    df_engineered['Date'] = pd.to_datetime(df_engineered['Date'])
    
    # Extract temporal features
    df_engineered['year'] = df_engineered['Date'].dt.year
    df_engineered['month'] = df_engineered['Date'].dt.month
    df_engineered['day'] = df_engineered['Date'].dt.day
    df_engineered['day_of_week'] = df_engineered['Date'].dt.dayofweek
    df_engineered['is_weekend'] = df_engineered['day_of_week'].isin([5, 6]).astype(int)
    
    # Create seasonal features
    df_engineered['season'] = pd.cut(df_engineered['month'], 
                                   bins=[0, 3, 6, 9, 12], 
                                   labels=['Winter', 'Spring', 'Summer', 'Autumn'])
    
    # Create lag features
    for lag in [1, 7, 30]:
        df_engineered[f'MWH_lag_{lag}'] = df_engineered['MWH'].shift(lag)
    
    # Create rolling averages
    for window in [7, 30]:
        df_engineered[f'MWH_rolling_{window}'] = df_engineered['MWH'].rolling(window=window).mean()
    
    print("Energy feature engineering completed")
    return df_engineered

# %%
# Apply feature engineering
if weather_valid:
    weather_df = engineer_weather_features(weather_df)
    print("Weather features engineered successfully")

if energy_valid:
    energy_df = engineer_energy_features(energy_df)
    print("Energy features engineered successfully")

# %%
# =============================================================================
# EXPLORATORY DATA ANALYSIS
# =============================================================================

def analyze_weather_data(df):
    """
    Perform comprehensive analysis of weather data.
    
    Args:
        df (pd.DataFrame): Weather dataset
    """
    print("Analyzing weather data...")
    
    # Basic statistics
    print("\n=== Weather Data Statistics ===")
    print(df.describe())
    
    # Missing values analysis
    print("\n=== Missing Values Analysis ===")
    missing_counts = df.isnull().sum()
    print(missing_counts[missing_counts > 0])
    
    # Correlation analysis
    print("\n=== Correlation Analysis ===")
    correlation_matrix = df.select_dtypes(include=[np.number]).corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=Config.FIGURE_SIZE)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Weather Data Correlation Matrix')
    plt.tight_layout()
    plt.savefig(f"{Config.PLOTS_DIR}/weather_correlation.png", dpi=Config.DPI, bbox_inches='tight')
    plt.show()
    
    # Seasonal analysis
    print("\n=== Seasonal Analysis ===")
    seasonal_stats = df.groupby('month')[['mean_temp', 'precipitation', 'sunshine']].mean()
    print(seasonal_stats)
    
    # Plot seasonal patterns
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Monthly temperature
    monthly_temp = df.groupby('month')['mean_temp'].mean()
    axes[0,0].plot(monthly_temp.index, monthly_temp.values, 'b-', linewidth=2, marker='o')
    axes[0,0].set_title('Average Monthly Temperature')
    axes[0,0].set_xlabel('Month')
    axes[0,0].set_ylabel('Temperature (°C)')
    axes[0,0].grid(True)
    
    # Monthly precipitation
    monthly_precip = df.groupby('month')['precipitation'].mean()
    axes[0,1].plot(monthly_precip.index, monthly_precip.values, 'g-', linewidth=2, marker='o')
    axes[0,1].set_title('Average Monthly Precipitation')
    axes[0,1].set_xlabel('Month')
    axes[0,1].set_ylabel('Precipitation (mm)')
    axes[0,1].grid(True)
    
    # Temperature distribution
    axes[1,0].hist(df['mean_temp'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1,0].set_title('Temperature Distribution')
    axes[1,0].set_xlabel('Temperature (°C)')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].grid(True)
    
    # Precipitation distribution
    axes[1,1].hist(df['precipitation'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1,1].set_title('Precipitation Distribution')
    axes[1,1].set_xlabel('Precipitation (mm)')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{Config.PLOTS_DIR}/weather_seasonal_analysis.png", dpi=Config.DPI, bbox_inches='tight')
    plt.show()

# %%
# Analyze weather data
if weather_valid:
    analyze_weather_data(weather_df)

# %%
def analyze_energy_data(df):
    """
    Perform comprehensive analysis of energy data.
    
    Args:
        df (pd.DataFrame): Energy dataset
    """
    print("Analyzing energy data...")
    
    # Basic statistics
    print("\n=== Energy Data Statistics ===")
    print(df.describe())
    
    # Borough analysis
    print("\n=== Borough Analysis ===")
    borough_stats = df.groupby('Borough')['MWH'].agg(['count', 'mean', 'std', 'min', 'max'])
    print(borough_stats)
    
    # Temporal analysis
    print("\n=== Temporal Analysis ===")
    df['Date'] = pd.to_datetime(df['Date'])
    df['month'] = df['Date'].dt.month
    monthly_energy = df.groupby('month')['MWH'].mean()
    print(monthly_energy)
    
    # Plot energy consumption patterns
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Monthly energy consumption
    axes[0,0].plot(monthly_energy.index, monthly_energy.values, 'r-', linewidth=2, marker='o')
    axes[0,0].set_title('Average Monthly Energy Consumption')
    axes[0,0].set_xlabel('Month')
    axes[0,0].set_ylabel('Energy Consumption (MWH)')
    axes[0,0].grid(True)
    
    # Energy consumption over time
    df_sorted = df.sort_values('Date')
    axes[0,1].plot(df_sorted['Date'], df_sorted['MWH'], alpha=0.6)
    axes[0,1].set_title('Energy Consumption Over Time')
    axes[0,1].set_xlabel('Date')
    axes[0,1].set_ylabel('Energy Consumption (MWH)')
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].grid(True)
    
    # Energy distribution
    axes[1,0].hist(df['MWH'], bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[1,0].set_title('Energy Consumption Distribution')
    axes[1,0].set_xlabel('Energy Consumption (MWH)')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].grid(True)
    
    # Borough comparison
    borough_means = df.groupby('Borough')['MWH'].mean().sort_values(ascending=False)
    axes[1,1].bar(range(len(borough_means)), borough_means.values)
    axes[1,1].set_title('Average Energy Consumption by Borough')
    axes[1,1].set_xlabel('Borough')
    axes[1,1].set_ylabel('Average Energy Consumption (MWH)')
    axes[1,1].set_xticks(range(len(borough_means)))
    axes[1,1].set_xticklabels(borough_means.index, rotation=45)
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{Config.PLOTS_DIR}/energy_analysis.png", dpi=Config.DPI, bbox_inches='tight')
    plt.show()

# %%
# Analyze energy data
if energy_valid:
    analyze_energy_data(energy_df)

# %%
# =============================================================================
# MISSING DATA IMPUTATION
# =============================================================================

def create_cloud_cover_model(df_clean, df_missing):
    """
    Create and train deep learning model for cloud cover imputation.
    
    Args:
        df_clean (pd.DataFrame): Dataset with complete cloud cover data
        df_missing (pd.DataFrame): Dataset with missing cloud cover data
        
    Returns:
        keras.Model: Trained classification model
    """
    print("Creating cloud cover imputation model...")
    
    # Prepare features and labels
    feature_columns = [col for col in df_clean.columns if col != 'cloud_cover']
    X = df_clean[feature_columns].to_numpy(dtype='float32')
    y = df_clean['cloud_cover'].to_numpy(dtype='int')
    
    # Split data
    boundary = int(X.shape[0] * Config.TRAIN_TEST_SPLIT)
    X_train, X_test = X[:boundary], X[boundary:]
    y_train, y_test = y[:boundary], y[boundary:]
    
    # Standardize features
    means = X_train.mean(axis=0)
    stds = X_train.std(axis=0)
    X_train = (X_train - means) / stds
    X_test = (X_test - means) / stds
    
    # Calculate class weights for imbalanced data
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    
    # Create model architecture
    inputs = keras.layers.Input(shape=(X_train.shape[1],))
    z = inputs
    
    # Projection layer
    z = keras.layers.Dense(256)(z)
    
    # Residual blocks
    for i in range(4):
        s = z  # Shortcut connection
        z = keras.layers.LayerNormalization()(z)
        z = keras.layers.Dense(6*256)(z)
        z = keras.activations.gelu(z)
        z = keras.layers.Dense(256)(z)
        z = keras.layers.Add()([s, z])  # Residual connection
    
    # Output layer
    z = keras.layers.LayerNormalization()(z)
    z = keras.layers.Dense(10)(z)  # 10 classes for cloud cover (0-9)
    outputs = keras.activations.softmax(z)
    
    # Create and compile model
    model = keras.Model(inputs=[inputs], outputs=[outputs])
    
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
        metrics=["accuracy"]
    )
    
    # Early stopping callback
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        restore_best_weights=True,
        verbose=1
    )
    
    # Train model
    print("Training cloud cover imputation model...")
    history = model.fit(
        X_train, y_train,
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        validation_split=0.2,
        class_weight=class_weight_dict,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Save model
    model.save(f"{Config.MODELS_DIR}/cloud_cover_model.h5")
    
    return model, means, stds

# %%
# Check for missing cloud cover data and handle imputation
if weather_valid and weather_df['cloud_cover'].isnull().sum() > 0:
    print("Missing cloud cover data detected. Creating imputation model...")
    
    # Split data for imputation
    df_clean = weather_df[weather_df['cloud_cover'].notna()].copy()
    df_missing = weather_df[weather_df['cloud_cover'].isna()].copy()
    
    print(f"Clean data: {df_clean.shape[0]} rows")
    print(f"Missing data: {df_missing.shape[0]} rows")
    
    # Create and train imputation model
    cloud_model, means, stds = create_cloud_cover_model(df_clean, df_missing)
    
    # Impute missing values
    feature_columns = [col for col in df_clean.columns if col != 'cloud_cover']
    X_pred = df_missing[feature_columns].to_numpy(dtype='float32')
    X_pred_std = (X_pred - means) / stds
    predictions = cloud_model.predict(X_pred_std)
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Fill missing values
    weather_df.loc[weather_df['cloud_cover'].isna(), 'cloud_cover'] = predicted_labels
    print("Cloud cover imputation completed")
else:
    print("No missing cloud cover data found")

# %%
# =============================================================================
# DATA INTEGRATION
# =============================================================================

def integrate_datasets(weather_df, energy_df):
    """
    Integrate weather and energy datasets.
    
    Args:
        weather_df (pd.DataFrame): Weather dataset
        energy_df (pd.DataFrame): Energy dataset
        
    Returns:
        pd.DataFrame: Integrated dataset
    """
    print("Integrating weather and energy datasets...")
    
    # Ensure date columns are in the same format
    weather_df = weather_df.rename(columns={'date': 'Date'})
    weather_df['Date'] = pd.to_datetime(weather_df['Date'])
    energy_df['Date'] = pd.to_datetime(energy_df['Date'])
    
    # Merge datasets
    integrated_df = energy_df.merge(weather_df, on='Date', how='inner')
    
    print(f"Integrated dataset shape: {integrated_df.shape}")
    print(f"Date range: {integrated_df['Date'].min()} to {integrated_df['Date'].max()}")
    
    return integrated_df

# %%
# Integrate datasets
if weather_valid and energy_valid:
    integrated_df = integrate_datasets(weather_df, energy_df)
    print("Dataset integration completed")
    
    # Display integrated dataset info
    print("\nIntegrated dataset info:")
    print(integrated_df.info())
    print("\nIntegrated dataset head:")
    print(integrated_df.head())
else:
    print("Cannot integrate datasets - one or both datasets are invalid")
    integrated_df = None

# %%
# =============================================================================
# PREPARE FINAL DATASET FOR MODELING
# =============================================================================

def prepare_final_dataset(df):
    """
    Prepare final dataset for modeling.
    
    Args:
        df (pd.DataFrame): Integrated dataset
        
    Returns:
        tuple: Features and target arrays, feature names
    """
    print("Preparing final dataset for modeling...")
    
    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df, columns=['Borough', 'season'])
    
    # Convert boolean columns to int
    bool_columns = df_encoded.select_dtypes(include='bool').columns
    for col in bool_columns:
        df_encoded[col] = df_encoded[col].astype(int)
    
    # Remove date columns and other non-numeric columns
    exclude_columns = ['Date', 'MWH']
    feature_columns = [col for col in df_encoded.columns if col not in exclude_columns]
    
    # Prepare features and target
    X = df_encoded[feature_columns].to_numpy(dtype='float32')
    y = df_encoded['MWH'].to_numpy(dtype='float32')
    
    print(f"Final dataset shape: X={X.shape}, y={y.shape}")
    print(f"Number of features: {len(feature_columns)}")
    
    return X, y, feature_columns

# %%
# Prepare final dataset
if integrated_df is not None:
    X, y, feature_names = prepare_final_dataset(integrated_df)
    
    # Split data
    boundary = int(X.shape[0] * Config.TRAIN_TEST_SPLIT)
    X_train, X_test = X[:boundary], X[boundary:]
    y_train, y_test = y[:boundary], y[boundary:]
    
    # Standardize features
    means = X_train.mean(axis=0)
    stds = X_train.std(axis=0)
    X_train = (X_train - means) / stds
    X_test = (X_test - means) / stds
    
    print("Final dataset prepared successfully")
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
else:
    print("Cannot prepare final dataset - integrated dataset is None")
    X_train, X_test, y_train, y_test, feature_names = None, None, None, None, None

# %%
# =============================================================================
# ENERGY CONSUMPTION PREDICTION MODEL
# =============================================================================

def create_energy_prediction_model(X_train, y_train, feature_count):
    """
    Create and train deep learning model for energy consumption prediction.
    
    Args:
        X_train (np.array): Training features
        y_train (np.array): Training targets
        feature_count (int): Number of input features
        
    Returns:
        keras.Model: Trained regression model
    """
    print("Creating energy consumption prediction model...")
    
    # Create model architecture
    inputs = keras.layers.Input(shape=(feature_count,))
    z = inputs
    
    # Projection layer
    z = keras.layers.Dense(192)(z)
    
    # Residual blocks
    for i in range(3):
        s = z  # Shortcut connection
        z = keras.layers.LayerNormalization()(z)
        z = keras.layers.Dense(4*192)(z)
        z = keras.activations.gelu(z)
        z = keras.layers.Dense(192)(z)
        z = keras.layers.Add()([s, z])  # Residual connection
    
    # Output layer
    z = keras.layers.Dense(1)(z)
    outputs = keras.activations.gelu(z)
    
    # Create and compile model
    model = keras.Model(inputs=[inputs], outputs=[outputs])
    
    model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
        metrics=["mse", "mae"]
    )
    
    # Early stopping callback
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        restore_best_weights=True,
        verbose=1
    )
    
    # Train model
    print("Training energy consumption prediction model...")
    history = model.fit(
        X_train, y_train,
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Save model
    model.save(f"{Config.MODELS_DIR}/energy_prediction_model.h5")
    
    return model, history

# %%
# Create and train energy prediction model
if X_train is not None:
    energy_model, history = create_energy_prediction_model(X_train, y_train, X_train.shape[1])
    print("Energy prediction model trained successfully")
else:
    print("Cannot train model - training data is None")
    energy_model, history = None, None

# %%
# =============================================================================
# MODEL EVALUATION
# =============================================================================

def evaluate_model(model, X_train, y_train, X_test, y_test, feature_names):
    """
    Comprehensive model evaluation.
    
    Args:
        model (keras.Model): Trained model
        X_train (np.array): Training features
        y_train (np.array): Training targets
        X_test (np.array): Test features
        y_test (np.array): Test targets
        feature_names (list): List of feature names
    """
    print("Evaluating model performance...")
    
    # Make predictions
    y_train_pred = model.predict(X_train).flatten()
    y_test_pred = model.predict(X_test).flatten()
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mape = mean_absolute_percentage_error(y_train, y_train_pred) * 100
    test_mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100
    
    # Print results
    print("\n=== Model Performance Metrics ===")
    print(f"Training RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Training MAPE: {train_mape:.2f}%")
    print(f"Test MAPE: {test_mape:.2f}%")
    
    # Plot predictions vs actual
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Training set
    axes[0].scatter(y_train, y_train_pred, alpha=0.5)
    axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual Energy Consumption (MWH)')
    axes[0].set_ylabel('Predicted Energy Consumption (MWH)')
    axes[0].set_title(f'Training Set (RMSE: {train_rmse:.4f})')
    axes[0].grid(True)
    
    # Test set
    axes[1].scatter(y_test, y_test_pred, alpha=0.5)
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[1].set_xlabel('Actual Energy Consumption (MWH)')
    axes[1].set_ylabel('Predicted Energy Consumption (MWH)')
    axes[1].set_title(f'Test Set (RMSE: {test_rmse:.4f})')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{Config.PLOTS_DIR}/model_predictions.png", dpi=Config.DPI, bbox_inches='tight')
    plt.show()
    
    return {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mape': train_mape,
        'test_mape': test_mape
    }

# %%
# Evaluate the model
if energy_model is not None:
    results = evaluate_model(energy_model, X_train, y_train, X_test, y_test, feature_names)
    print("Model evaluation completed")
else:
    print("Cannot evaluate model - model is None")
    results = None

# %%
# =============================================================================
# FEATURE IMPORTANCE ANALYSIS
# =============================================================================

def analyze_feature_importance(model, feature_names, X_test, y_test):
    """
    Analyze feature importance using permutation importance.
    
    Args:
        model (keras.Model): Trained model
        feature_names (list): List of feature names
        X_test (np.array): Test features
        y_test (np.array): Test targets
    """
    print("Analyzing feature importance...")
    
    # Calculate permutation importance
    result = permutation_importance(
        model, X_test, y_test, 
        n_repeats=10, 
        random_state=Config.RANDOM_SEED
    )
    
    # Create feature importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(20)
    plt.barh(range(len(top_features)), top_features['importance_mean'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top 20 Most Important Features for Energy Consumption Prediction')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{Config.PLOTS_DIR}/feature_importance.png", dpi=Config.DPI, bbox_inches='tight')
    plt.show()
    
    # Print top features
    print("\n=== Top 10 Most Important Features ===")
    print(importance_df.head(10))
    
    return importance_df

# %%
# Analyze feature importance
if energy_model is not None:
    importance_df = analyze_feature_importance(energy_model, feature_names, X_test, y_test)
    print("Feature importance analysis completed")
else:
    print("Cannot analyze feature importance - model is None")
    importance_df = None

# %%
# =============================================================================
# VISUALIZATION AND INSIGHTS
# =============================================================================

def create_geographic_visualization(df):
    """
    Create interactive map showing energy consumption by location.
    
    Args:
        df (pd.DataFrame): Integrated dataset with location information
    """
    print("Creating geographic visualization...")
    
    # Check if location data is available
    if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
        print("Location data not available for geographic visualization")
        return
    
    # Create map
    m = folium.Map(location=[51.5074, -0.1278], zoom_start=10, tiles='OpenStreetMap')
    marker_cluster = MarkerCluster().add_to(m)
    
    # Normalize MWH values for color mapping
    norm = colors.Normalize(vmin=df['MWH'].min(), vmax=df['MWH'].max())
    
    # Add markers
    for index, row in df.iterrows():
        color = plt.cm.jet(norm(row['MWH']))
        color_hex = colors.rgb2hex(color)
        
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            color=color_hex,
            fill_color=color_hex,
            fill_opacity=0.5,
            radius=5,
            popup=f"Borough: {row['Borough']}<br>MWH: {row['MWH']:.2f}"
        ).add_to(marker_cluster)
    
    # Save map
    m.save(f"{Config.OUTPUT_DIR}/energy_consumption_map.html")
    print(f"Map saved to {Config.OUTPUT_DIR}/energy_consumption_map.html")

# %%
# Create geographic visualization
if integrated_df is not None:
    create_geographic_visualization(integrated_df)

# %%
def generate_business_insights(results, importance_df):
    """
    Generate business insights from model results.
    
    Args:
        results (dict): Model performance results
        importance_df (pd.DataFrame): Feature importance analysis
    """
    print("Generating business insights...")
    
    insights = []
    
    # Model performance insights
    if results['test_mape'] < 10:
        insights.append("Excellent model performance with less than 10% prediction error")
    elif results['test_mape'] < 20:
        insights.append("Good model performance with reasonable prediction accuracy")
    else:
        insights.append("Model performance needs improvement - consider feature engineering or data quality")
    
    # Feature importance insights
    top_features = importance_df.head(5)['feature'].tolist()
    insights.append(f"Top predictive factors: {', '.join(top_features)}")
    
    # Seasonal insights
    if 'month' in importance_df['feature'].values:
        insights.append("Seasonal patterns significantly influence energy consumption")
    
    if 'mean_temp' in importance_df['feature'].values:
        insights.append("Temperature is a key driver of energy consumption")
    
    # Print insights
    print("\n=== Business Insights ===")
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight}")
    
    # Save insights to file
    with open(f"{Config.OUTPUT_DIR}/business_insights.txt", 'w') as f:
        f.write("London Energy Consumption Prediction - Business Insights\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Model Performance:\n")
        f.write(f"- Test RMSE: {results['test_rmse']:.4f}\n")
        f.write(f"- Test MAPE: {results['test_mape']:.2f}%\n\n")
        f.write("Key Insights:\n")
        for i, insight in enumerate(insights, 1):
            f.write(f"{i}. {insight}\n")
    
    print(f"Insights saved to {Config.OUTPUT_DIR}/business_insights.txt")

# %%
# Generate business insights
if results is not None and importance_df is not None:
    generate_business_insights(results, importance_df)
    print("Business insights generated successfully")
else:
    print("Cannot generate business insights - results or importance data is None")

# %%
# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("=" * 60)
print("LONDON ENERGY CONSUMPTION PREDICTION - COMPLETED")
print("=" * 60)

if results is not None:
    print(f"\nFinal Model Performance:")
    print(f"- Test RMSE: {results['test_rmse']:.4f}")
    print(f"- Test MAPE: {results['test_mape']:.2f}%")

print(f"\nOutputs saved to:")
print(f"- Plots: {Config.PLOTS_DIR}/")
print(f"- Models: {Config.MODELS_DIR}/")
print(f"- Results: {Config.OUTPUT_DIR}/")

print("\nAnalysis completed successfully!")
print("=" * 60)