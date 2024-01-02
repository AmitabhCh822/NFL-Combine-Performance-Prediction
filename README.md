# NFL Combine Performance Prediction

This repository houses a data analysis project which applies the K-Nearest Neighbors (KNN) algorithm to predict NFL Combine performance grades. The analysis uses a dataset from the 2014 NFL Combine, featuring player measurements and drill results.

## Project Structure

- `nfl_combine_2014.dat.txt` - The original dataset containing NFL combine data in a fixed-width format.
- `nfl_combine_analysis.py` - Python script for preprocessing data, training the KNN model, and evaluating its performance.
- `nfl.csv` - Generated CSV file after preprocessing the original dataset for ease of use.

## Getting Started

To run the analysis on your local machine:

1. Clone this repository.
2. Ensure you have a Python environment with necessary libraries installed (`numpy`, `pandas`, `sklearn`).
3. Run the `nfl_combine_analysis.py` script to perform the data preprocessing, model training, and evaluation.

## Analysis Workflow

The script `nfl_combine_analysis.py` follows these steps:

1. Reads and processes the original `.dat` file.
2. Converts the processed data into a CSV file for further analysis.
3. Preprocesses the data by labeling columns, handling missing values, and normalizing.
4. Transforms the 'Grade' attribute into a binary classification target.
5. Splits the data into training and testing sets.
6. Trains and tests a KNN classifier.
7. Outputs the model's performance metrics.

## Results

The script will output:

- A confusion matrix that shows the classifier's performance in detail.
- A classification report with precision, recall, f1-score, and accuracy.
- The overall accuracy score of the model.

These results help in understanding the model's effectiveness at predicting performance grades based on combine data.

## Contributor
Amitabh Chakravorty




Feel free to fork this repository and submit pull requests. For substantial changes, please open an issue first to discuss what you would like to change.
