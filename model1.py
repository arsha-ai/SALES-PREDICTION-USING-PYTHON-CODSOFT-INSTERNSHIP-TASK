import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Step 1: Load and prepare the dataset
dataset_path = r"C:\Users\siddh\OneDrive\Desktop\codsoft intern\SALES PRIDECTION\advertising (1).csv"
df = pd.read_csv(dataset_path)

X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Step 2: Split the data and scale features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Create and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Step 4: Evaluate the model
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# Step 5: Function to get user input and make predictions
def predict_sales():
    print("\nEnter advertising budget for each medium:")
    tv = float(input("TV advertising budget ($): "))
    radio = float(input("Radio advertising budget ($): "))
    newspaper = float(input("Newspaper advertising budget ($): "))
    
    # Scale the input
    user_input_scaled = scaler.transform([[tv, radio, newspaper]])
    
    # Make prediction
    predicted_sales = model.predict(user_input_scaled)[0]
    
    print(f"\nPredicted Sales: ${predicted_sales:.2f}")
    
    # Display feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.bar(feature_importance['feature'], feature_importance['importance'])
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.title("Feature Importance in Sales Prediction")
    plt.tight_layout()
    plt.show()

# Step 6: Main loop for user interaction
while True:
    predict_sales()
    again = input("\nWould you like to make another prediction? (yes/no): ")
    if again.lower() != 'yes':
        break

print("Thank you for using the Sales Prediction model!")


