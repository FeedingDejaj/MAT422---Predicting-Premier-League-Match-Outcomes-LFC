import os
import logging
import pandas as pd


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s [feature_engineering]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def data_exploration_cleaning_feature_engineering(
    input_csv_path,
    output_csv_path,
    rolling_window=5
):
    """
    Perform data exploration, cleaning, and feature engineering on the scraped data.

    Args:
        input_csv_path (str): Path to the CSV file produced by soccer_data.py.
        output_csv_path (str): Path where the processed CSV file will be saved.
        rolling_window (int): Number of previous games to consider for rolling features (default=5).

    Returns:
        pd.DataFrame: The processed DataFrame with engineered features.
    """
    logger.info(f"Loading data from {input_csv_path}")
    if not os.path.exists(input_csv_path):
        logger.error(f"Input file {input_csv_path} does not exist.")
        raise FileNotFoundError(f"Input file {input_csv_path} not found.")

    df = pd.read_csv(input_csv_path)

    # Ensure required columns are present
    required_columns = ['Date', 'Goals For', 'Goals Against', 'Opponent']
    missing_required = [col for col in required_columns if col not in df.columns]
    if missing_required:
        logger.error(f"Missing required columns: {missing_required}")
        raise ValueError(f"Missing required columns in the data: {missing_required}")

    # Convert 'Date' to datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Date'], inplace=True)

    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)

    # Feature Engineering
    # Create Goal Difference
    if 'Goals For' in df.columns and 'Goals Against' in df.columns:
        df['Goal Difference'] = df['Goals For'] - df['Goals Against']

    # Create a binary target variable for Win/Loss/Draw
    if 'Goals For' in df.columns and 'Goals Against' in df.columns:
        df['Result'] = df.apply(
            lambda row: 'Win' if row['Goals For'] > row['Goals Against']
            else ('Draw' if row['Goals For'] == row['Goals Against'] else 'Loss'),
            axis=1
        )
        result_mapping = {'Win': 1, 'Draw': 0, 'Loss': -1}
        df['Result_Code'] = df['Result'].map(result_mapping)

    # Create Opponent_Code using label encoding
    if 'Opponent' in df.columns:
        df['Opponent_Code'] = df['Opponent'].astype('category').cat.codes

    # Features for form over last N games
    rolling_features = [
        'Goals For', 'Goals Against', 'Goal Difference', 'Possession',
        'Expected Goals (xG)', 'Expected Goals Against (xGA)',
        'Interceptions', 'Clearances', 'Errors'
    ]

    for feature in rolling_features:
        if feature in df.columns:
            df[f'{feature}_Rolling_{rolling_window}'] = df[feature].rolling(window=rolling_window, min_periods=1).mean()
        else:
            logger.warning(f"Column '{feature}' not found for rolling calculation.")

    # Days since last match
    if 'Date' in df.columns:
        df['Days Since Last Match'] = df['Date'].diff().dt.days.fillna(0)
    else:
        logger.warning("Column 'Date' not found - cannot compute 'Days Since Last Match'.")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_csv_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Save processed data
    df.to_csv(output_csv_path, index=False)
    logger.info(f"Processed data saved to {output_csv_path}")

    logger.info(f"Processed Data Summary:\n"
                f"Total matches: {len(df)}\n"
                f"Date range: {df['Date'].min()} to {df['Date'].max()}\n"
                f"Columns: {', '.join(df.columns)}")

    return df
