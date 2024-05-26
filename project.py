# Technology used:
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import dtale
from scipy.stats import shapiro, kstest, norm, probplot
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor


#Data collection
df1 = pd.read_csv("ResaleFlatPricesBasedonRegistrationDateFromJan2015toDec2016.csv")
df2 = pd.read_csv("ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv")
df3 = pd.read_csv("ResaleFlatPricesBasedonRegistrationDateFromMar2012toDec2014.csv")

# Concatenate data frames vertically
df = pd.concat([df1, df2, df3], axis=0)

# Data cleaning:
df['month'] = pd.to_datetime(df['month'], format='%Y-%m')

# Extract the month and year information
df['year'] = df['month'].dt.year
df['month'] = df['month'].dt.month
df = df.dropna()

def extract_years(string):
    if isinstance(string, str):
        if '-' in string:
            parts = string.split('-')
            if len(parts) >= 1:
                years = int(parts[0])
                return years
        else:
            # Use regular expression to extract numeric part from the string
            numeric_part = re.search(r'\d+', string)
            if numeric_part:
                years = int(numeric_part.group())
                return years
    elif isinstance(string, int):
        return string
    return None

# Apply the custom function to the column
df['remaining_lease'] = df['remaining_lease'].apply(extract_years)

# Data visualization:
def plot(df, column):
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Box plot
    sns.boxplot(data=df, x=column, ax=axes[0])
    axes[0].set_title(f'Box Plot for {column}')

    # Distribution plot
    sns.histplot(data=df, x=column, kde=True, bins=50, ax=axes[1])
    axes[1].set_title(f'Distribution Plot for {column}')

    # Violin plot
    sns.violinplot(data=df, x=column, ax=axes[2])
    axes[2].set_title(f'Violin Plot for {column}')

    # Display the figure
    st.pyplot(fig)
    
column_names = df.columns.tolist()
    
df_cat = df.select_dtypes(object)
    
df_encoded = df.copy()

def EDA(df):
    # Launch d-tale for the DataFrame
    d = dtale.show(df)

    # To view the d-tale instance in a Jupyter notebook, you can use:
    d.open_browser()
    
def encode_cat(df, column_name):
    from sklearn.preprocessing import LabelEncoder

    # Instantiate the LabelEncoder object
    le = LabelEncoder()

    # Fit and transform the specified column
    encoded_column = le.fit_transform(df[column_name])

    # Replace the original column with the encoded values
    df[column_name] = encoded_column

    # Return the modified DataFrame
    return df
    
column_Cat_names = df_cat.columns.tolist()
for column_name in column_Cat_names:
    df_encoded = encode_cat(df_encoded, column_name)

def check_normality(data, column):
    print(f"\nNormality check for {column}:")
    
    # Shapiro-Wilk Test
    stat, p_value = shapiro(data[column].dropna())
    print(f"Shapiro-Wilk Test: Statistic={stat}, p-value={p_value}")
    
    # Kolmogorov-Smirnov Test
    stat, p_value = kstest(data[column].dropna(), 'norm', args=(data[column].mean(), data[column].std()))
    print(f"Kolmogorov-Smirnov Test: Statistic={stat}, p-value={p_value}")
    
    # Plot histogram
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(data[column].dropna(), kde=True)
    plt.title(f"Histogram of {column}")
    
    # Q-Q Plot
    plt.subplot(1, 2, 2)
    probplot(data[column].dropna(), dist="norm", plot=plt)
    plt.title(f"Q-Q Plot of {column}")
    
    plt.tight_layout()
    st.pyplot()
num_cols = df_encoded.select_dtypes(include=np.number).columns    

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded)

# Apply dimensionality reduction (PCA) to reduce the number of features
pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
X_pca = pca.fit_transform(X_scaled)

# Train the KMeans model
kmeans = KMeans(n_clusters=5)  # Assuming 5 clusters
kmeans.fit(X_scaled)

# Assign cluster labels using K-means clustering
labels = kmeans.labels_

df_skewed = df_encoded.copy()
column_skewness = df_skewed.columns.tolist()

# skewness data:
epsilon = 1e-9
def Skewness(df_skewed,column_skewness):
    df_skewed[column_skewness] = np.log(df_skewed[column_skewness] + epsilon)
for i in column_skewness:
    Skewness(df_skewed,i)  
    
df_outiler = df_skewed.copy()
column_outlier = df_outiler.columns.tolist()

def outlier(df, column):
    iqr = df[column].quantile(0.75) - df[column].quantile(0.25)
    upper_threshold = df[column].quantile(0.75) + (1.5*iqr)
    lower_threshold = df[column].quantile(0.25) - (1.5*iqr)
    df[column] = df[column].clip(lower_threshold, upper_threshold)

for i in column_outlier:  
    outlier(df_outiler, i)
    
features = df_outiler[["flat_type","floor_area_sqm","lease_commence_date","remaining_lease","storey_range","year","flat_model"]]

target = df['resale_price']

X = features
y = target

def machine_learning_resale(X, y, algorithm):
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

    # Standardize the features
    SS = StandardScaler()
    X = SS.fit_transform(X)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)
    
    # Initialize the model with specified algorithm
    model = algorithm()
    
    # Fit the model on the training data
    model.fit(X_train, y_train)
    
    # Calculate accuracy for training and testing sets
    train_mse = mean_squared_error(y_train, model.predict(X_train))
    test_mse = mean_squared_error(y_test, model.predict(X_test))
        
    # Calculate R2 score for training and testing sets
    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2 = r2_score(y_test, model.predict(X_test))

    return train_mse, test_mse, train_r2, test_r2

machine_learning_resale(X, y, GradientBoostingRegressor)

def confusion(X, y, algorithm):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

    # Initialize the model with specified algorithm
    model = algorithm()

    # Fit the model on the training data
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(7, 5))
    sns.heatmap(conf_matrix, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.title('Confusion Matrix')
    
    # Display the plot using Streamlit
    st.pyplot()

# streamlit:
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor

# Suppress deprecation warning
st.set_option('deprecation.showPyplotGlobalUse', False)

# Header
st.header('Singapore Resale Flat Prices Prediction')
# Buttons for EDA and Data view
if st.button("EDA"):
    EDA(df_encoded)  # Assuming EDA is a defined function
if st.button("Data view"):   
    for i in column_names:
        plot(df, i)  # Assuming plot is a defined function
    for col in num_cols:
        check_normality(df_encoded, col) 

# Create tabs
tab1 = st.tabs(["Data prediction"])
# Data prediction tab
with tab1[0]:
    st.write("Please enter the following feature values:")
    st.write("Flate_type - 2, floor_area_sqm - 60.0, lease_commence_date - 1986, remaining_lease- 70, storey_range - 2, year - 2015, flat_model- 5")
    input_features = []

    for feature_name in ["flat_type", "floor_area_sqm", "lease_commence_date", "remaining_lease", "storey_range", "year", "flat_model"]:
        input_value = st.number_input(f"Enter {feature_name}:")
        input_features.append(input_value)

    if len(input_features) == 7:
        selected_model = GradientBoostingRegressor()
    else:
        st.error("Please enter all feature values.")          

    if st.button("Predict"):
        # Assuming df_outlier is your DataFrame and it has already been preprocessed
        X = df_outiler[["flat_type", "floor_area_sqm", "lease_commence_date", "remaining_lease", "storey_range", "year", "flat_model"]]
        y = df['resale_price']

        selected_model.fit(X, y)
        predicted_score = selected_model.predict([input_features])[0]
        st.write("Predicted resale price:", predicted_score)
