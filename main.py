import os
import logging
import pandas as pd


from soccer_data_pipeline import scrape_fbref_data
from feature_engineering_pipeline import data_exploration_cleaning_feature_engineering
from predict_outcome_pipeline import improved_predict_match_outcomes


logger = logging.getLogger()
logger.setLevel(logging.INFO)


log_file = 'pipeline_log.txt'
file_handler = logging.FileHandler(log_file, mode='w')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s %(levelname)s [main]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)


team = "Liverpool"

season_mappings = {
    "2018-2019": 2018,
    "2019-2020": 2019,
    "2020-2021": 2020,
    "2021-2022": 2021,
    "2022-2023": 2022,
    "2023-2024": 2023
}


data_dir = "../modsData"
if not os.path.exists(data_dir):
    os.makedirs(data_dir, exist_ok=True)


TRAIN_SIZE = 33
TEST_SIZE = 5

ROLLING_WINDOW = 5

# A list to store summary results for each season
results_summary = []



def main():
    logger.info("Starting the pipeline for multiple seasons...")
    for season_name, end_year in season_mappings.items():
        logger.info(f"\nProcessing season: {season_name}")

        # Define file paths
        raw_output_csv = os.path.join(data_dir, f"lfc_data_{end_year}.csv")
        processed_csv = os.path.join(data_dir, f"lfc_data_{end_year}_processed.csv")


        try:
            scrape_fbref_data(team, [end_year], raw_output_csv)
        except Exception as e:
            logger.error(f"Failed during scrape_fbref_data for {season_name}: {e}")
            continue

        # Step 2: Feature Engineering
        try:
            data_exploration_cleaning_feature_engineering(
                input_csv_path=raw_output_csv,
                output_csv_path=processed_csv,
                rolling_window=ROLLING_WINDOW
            )
        except Exception as e:
            logger.error(f"Failed during feature engineering for {season_name}: {e}")
            continue

        # Step 3: Predict Outcomes
        try:
            results = improved_predict_match_outcomes(
                csv_file_path=processed_csv,
                train_size=TRAIN_SIZE,
                test_size=TEST_SIZE
            )
            # Extract summary metrics
            logreg_acc = results.get('logreg_accuracy', None)
            rf_acc = results.get('rf_accuracy', None)

            # Add to summary
            results_summary.append({
                'Season': season_name,
                'LogisticRegressionAccuracy': logreg_acc,
                'RandomForestAccuracy': rf_acc
            })
        except Exception as e:
            logger.error(f"Failed during prediction for {season_name}: {e}")
            continue

    # After processing all seasons, save a summary of results
    summary_df = pd.DataFrame(results_summary)
    summary_path = os.path.join(data_dir, "season_results_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"All seasons processed. Summary saved to {summary_path}")

    logger.info("Pipeline complete.")

if __name__ == "__main__":
    main()
