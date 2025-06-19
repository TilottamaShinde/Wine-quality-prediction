#  Wine Quality Prediction

This project uses machine learning to predict the quality of red wine based on physicochemical features.  
It uses the [UCI Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality).

##  Files

- `preprocess.py`: Load and scale the data.
- `train.py`: Train a Random Forest Regressor.
- `predict.py`: Make predictions using saved model.
- `utils.py`: Extra tools like correlation matrix plot.
- `data/winequality-red.csv`: Dataset file.

##  Libraries Used

- pandas
- scikit-learn
- joblib
- seaborn & matplotlib (optional for visualization)

##  How to Run

1. Clone this repo.
2. Download the [dataset CSV](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv) into the `data/` folder.
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Train the model:
    ```bash
    python train.py
    ```
5. Predict sample quality:
    ```bash
    python predict.py
    ```

##  Concepts Covered

- Regression
- Feature Scaling
- Model Evaluation (MSE, R²)
- Joblib for model saving
