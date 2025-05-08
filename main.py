from logreg import *

#Function to predict the class of a new sample
def predict_vehicle_type(input_data, model):
    """
    Predict vehicle type (bus, car, van) from input features.
    
    Parameters:
        input_data (list or np.array): 18 numerical features in order:
            [COMPACTNESS, CIRCULARITY, DISTANCE_CIRCULARITY, RADIUS_RATIO,
             PR_AXL_ASPECT_RATIO, MAX_LENGTH_ASYMETRY, SCATTER_RATIO,
             ELONGATEDNESS, PR_AXL_RECTANGULARITY, MAXIMAL_INDENTATION_DEPTH,
             SCALED_VARIANCE_MAJOR, SCALED_VARIANCE_MINOR, SCALED_RADIUS_OF_GYRATION,
             SKEWNESS_ABOUT_MAJOR, SKEWNESS_ABOUT_MINOR, KURTOSIS_ABOUT_MAJOR,
             KURTOSIS_ABOUT_MINOR, HOLLOWS_RATIO]
    
    Returns:
        str: Predicted vehicle type ('bus', 'car', 'van')
    """
    # Scale input data
    input_scaled = scaler.transform([input_data])
    
    # Predict and decode
    prediction = model.predict(input_scaled)
    return le.inverse_transform(prediction)[0]

# Example input (replace with actual values)
sample_input = [
    [89.8, 98.7, 93.2, 0.89,
     0.12, 0.45, 0.67, 4.32,
     0.81, 0.23, 2.11, 1.98,
     3.05, 0.76, 1.23, 0.54,
     0.89, 0.34],
    [107,30, 106, 172,
     50, 6, 255, 26,
     28, 169, 280, 1.957,
     264, 85, 5, 9,
     181, 183]
]

for i in range(2):
    predicted_class = predict_vehicle_type(sample_input[i], best_model)
    print(f"Predicted Vehicle {i+1} Type is {predicted_class} \n")

