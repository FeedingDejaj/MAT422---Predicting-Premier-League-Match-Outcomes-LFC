import os
import logging
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s [predict_outcome]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def improved_predict_match_outcomes(
    csv_file_path,
    train_size=33,
    test_size=5,
    output_dir="../modsData"
):
    """
    Train models (Logistic Regression and Random Forest) on the first `train_size` games of a season,
    then predict the final `test_size` games' outcomes and evaluate performance.

    Args:
        csv_file_path (str): Path to the processed CSV file.
        train_size (int): Number of games to use for training. Default is 33.
        test_size (int): Number of games to use for testing. Default is 5.
        output_dir (str): Directory where figures and logs can be saved.

    Returns:
        dict: A dictionary containing evaluation metrics and results.
    """
    logger.info(f"Loading processed data from {csv_file_path}")
    if not os.path.exists(csv_file_path):
        logger.error(f"CSV file {csv_file_path} does not exist.")
        raise FileNotFoundError(f"CSV file {csv_file_path} not found.")

    df = pd.read_csv(csv_file_path)
    required_columns = ['Date', 'Opponent', 'Result_Code']
    missing_req_cols = [col for col in required_columns if col not in df.columns]
    if missing_req_cols:
        logger.error(f"Missing required columns: {missing_req_cols}")
        raise ValueError(f"Required columns missing: {missing_req_cols}")

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    total_matches = len(df)
    if total_matches < train_size + test_size:
        logger.error(f"Not enough matches ({total_matches}) for train_size={train_size} and test_size={test_size}")
        raise ValueError("Insufficient data to split into the requested training and testing sets.")

    target = 'Result_Code'
    features = [
        'Home Venue', 'Opponent_Code',
        'Goals For_Rolling_5', 'Goals Against_Rolling_5',
        'Goal Difference_Rolling_5', 'Possession_Rolling_5',
        'Expected Goals (xG)_Rolling_5', 'Expected Goals Against (xGA)_Rolling_5',
        'Interceptions_Rolling_5', 'Clearances_Rolling_5', 'Errors_Rolling_5',
        'Days Since Last Match'
    ]

    missing_features = [feat for feat in features if feat not in df.columns]
    if missing_features:
        logger.warning(f"Warning: The following features are missing and will be ignored: {missing_features}")
        features = [feat for feat in features if feat in df.columns]

    X = df[features]
    y = df[target]

    # Split the data
    X_train = X.iloc[:train_size].reset_index(drop=True)
    y_train = y.iloc[:train_size].reset_index(drop=True)
    X_test = X.iloc[train_size:train_size+test_size].reset_index(drop=True)
    y_test = y.iloc[train_size:train_size+test_size].reset_index(drop=True)

    dates_test = df['Date'].iloc[train_size:train_size+test_size].reset_index(drop=True)
    opponents_test = df['Opponent'].iloc[train_size:train_size+test_size].reset_index(drop=True)

    # Handle class imbalance via oversampling
    train_data = pd.concat([X_train, y_train], axis=1)
    win = train_data[train_data[target] == 1]
    draw = train_data[train_data[target] == 0]
    loss = train_data[train_data[target] == -1]

    max_count = max(len(win), len(draw), len(loss))
    if max_count == 0:
        logger.error("No training samples found. Cannot proceed with model training.")
        raise ValueError("No training data available.")

    def oversample_class(df_class, class_name):
        if df_class.empty:
            logger.warning(f"No samples for class {class_name}. The model cannot learn about this class.")
            return df_class
        return resample(df_class, replace=True, n_samples=max_count, random_state=42)

    win_oversampled = oversample_class(win, 'Win')
    draw_oversampled = oversample_class(draw, 'Draw')
    loss_oversampled = oversample_class(loss, 'Loss')

    train_balanced = pd.concat([win_oversampled, draw_oversampled, loss_oversampled])
    train_balanced = train_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    X_train_balanced = train_balanced[features]
    y_train_balanced = train_balanced[target]

    scaler = StandardScaler()
    X_train_balanced = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)

    # Logistic Regression
    logreg_model = LogisticRegression(max_iter=1000)
    logreg_model.fit(X_train_balanced, y_train_balanced)
    y_pred_logreg = logreg_model.predict(X_test_scaled)

    accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
    logger.info(f"Logistic Regression Model Accuracy after balancing: {accuracy_logreg:.2f}")

    logreg_report = classification_report(y_test, y_pred_logreg, target_names=['Loss', 'Draw', 'Win'], zero_division=0)
    logger.info("Logistic Regression Classification Report after balancing:\n" + logreg_report)

    cm_logreg = confusion_matrix(y_test, y_pred_logreg, labels=[-1, 0, 1])
    cm_df_logreg = pd.DataFrame(cm_logreg, index=['Actual Loss', 'Actual Draw', 'Actual Win'],
                                columns=['Predicted Loss', 'Predicted Draw', 'Predicted Win'])

    # Save Logistic Regression Confusion Matrix to file instead of showing
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_df_logreg, annot=True, fmt='d', cmap='Blues')
    plt.title('Logistic Regression Confusion Matrix')
    plt.ylabel('Actual Result')
    plt.xlabel('Predicted Result')
    plt.tight_layout()

    # Derive a filename from the input CSV (e.g., season year)
    base_name = os.path.splitext(os.path.basename(csv_file_path))[0]
    lr_cm_path = os.path.join(output_dir, f"{base_name}_logreg_confusion_matrix.png")
    plt.savefig(lr_cm_path)
    plt.close()

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_balanced, y_train_balanced)
    y_pred_rf = rf_model.predict(X_test_scaled)

    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    logger.info(f"Random Forest Model Accuracy after balancing: {accuracy_rf:.2f}")

    rf_report = classification_report(y_test, y_pred_rf, target_names=['Loss', 'Draw', 'Win'], zero_division=0)
    logger.info("Random Forest Classification Report after balancing:\n" + rf_report)

    cm_rf = confusion_matrix(y_test, y_pred_rf, labels=[-1, 0, 1])
    cm_df_rf = pd.DataFrame(cm_rf, index=['Actual Loss', 'Actual Draw', 'Actual Win'],
                            columns=['Predicted Loss', 'Predicted Draw', 'Predicted Win'])

    # Save Random Forest Confusion Matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_df_rf, annot=True, fmt='d', cmap='Greens')
    plt.title('Random Forest Confusion Matrix')
    plt.ylabel('Actual Result')
    plt.xlabel('Predicted Result')
    plt.tight_layout()

    rf_cm_path = os.path.join(output_dir, f"{base_name}_rf_confusion_matrix.png")
    plt.savefig(rf_cm_path)
    plt.close()

    # Compare predictions with actual outcomes
    results = pd.DataFrame()
    results['Date'] = dates_test
    results['Opponent'] = opponents_test
    results['Actual Result'] = y_test.map({1: 'Win', 0: 'Draw', -1: 'Loss'})
    results['LogReg Prediction'] = pd.Series(y_pred_logreg).map({1: 'Win', 0: 'Draw', -1: 'Loss'})
    results['RandomForest Prediction'] = pd.Series(y_pred_rf).map({1: 'Win', 0: 'Draw', -1: 'Loss'})

    logger.info("Predictions vs Actual Results:\n" + results[['Date', 'Opponent', 'Actual Result', 'LogReg Prediction', 'RandomForest Prediction']].to_string(index=False))

    return {
        'logreg_accuracy': accuracy_logreg,
        'rf_accuracy': accuracy_rf,
        'logreg_report': logreg_report,
        'rf_report': rf_report,
        'results_dataframe': results
    }

