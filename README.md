# Vehicle Silhouettes Classification

This repository contains a machine learning project that classifies vehicle types (bus, car, van) based on silhouette features using Logistic Regression. 
The project demonstrates data preprocessing, model training, hyperparameter tuning with GridSearchCV, and making predictions on new data.

## Repository Structure

```
 Vehicle Silhouettes/
    ├── vehicle.csv             # Dataset containing vehicle silhouette features and target class
    ├── logreg.py               # Script for training Logistic Regression, hyperparameter tuning, and model evaluation
    └── main.py                 # Main script for loading the trained model and making predictions on new vehicle data
```

## Overview

- **Data Preprocessing:**  
  - Reads the dataset from `vehicle.csv`.
  - Encodes categorical labels using `LabelEncoder`.
  - Handles missing values by dropping rows with any NaNs.
  - Splits the data into training and test sets.
  - Scales the features using `StandardScaler`.

- **Model Training & Hyperparameter Tuning:**  
  - The primary model is Logistic Regression, which is trained on the scaled data.
  - Hyperparameter tuning is performed using `GridSearchCV` to search over parameters like regularization strength (`C`), solver type, and maximum iterations.

- **Prediction Functionality:**  
  - A helper function in `main.py` demonstrates how to predict the vehicle type for new input data.
  - The predicted class is then decoded back to its original label (bus, car, van).

## Requirements

- Python 3.x
- Pandas
- NumPy
- scikit-learn
- seaborn
- matplotlib

You can install the required packages using pip:

```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```

## How to Run

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/your-repository-name.git
   ```

2. **Navigate to the Project Folder:**

   ```bash
   cd your-repository-name/Applied ML/Vehicle Silhouettes
   ```

3. **Train the Model:**

   The training and hyperparameter tuning are implemented in `logreg.py`.

   ```bash
   python logreg.py
   ```

   This script will:
   - Read and preprocess the data from `vehicle.csv`
   - Train a Logistic Regression model on the training data
   - Tune the hyperparameters using GridSearchCV
   - Output the best hyperparameters and prepare the best model for prediction

4. **Make Predictions:**

   The `main.py` script uses the trained model to predict the vehicle type for new inputs.

   ```bash
   python main.py
   ```

   This will:
   - Scale new input data
   - Use the best trained model to predict the vehicle type
   - Print the predicted class for each sample

## Additional Notes

- **Dataset Location:**  
  Ensure that `vehicle.csv` is in the correct folder relative to the scripts. Adjust file paths in the code if necessary.

- **Customization:**  
  You can adjust the hyperparameter grid in `logreg.py` to suit different modeling needs. Feel free to experiment with different solvers and parameters.

- **Troubleshooting:**  
  In case of issues with scaling or encoding, double-check that the dataset matches the expected format as used in the scripts.

## Contact

For questions or contributions, please open an issue or submit a pull request.

Happy coding!
