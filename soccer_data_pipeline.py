import os
import logging
import warnings
import pandas as pd
from soccerdata import FBref


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s [soccer_data]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def scrape_fbref_data(team, seasons, output_csv_path,
                      leagues=['ENG-Premier League'],
                      stat_types=["schedule", "shooting", "passing", "defense", "keeper", "misc"],
                      force_cache=True):
    """
    Scrape football statistics from FBref for a specific team and given seasons,
    then process and save the cleaned data to a CSV file.

    Args:
        team (str): Team name (e.g., "Liverpool")
        seasons (list): List of seasons to scrape (e.g., [2023] for 2023-2024)
        output_csv_path (str): Path where the CSV file will be saved
        leagues (list): List of leagues to consider. Default is ['ENG-Premier League'].
        stat_types (list): List of stat types to fetch. Default includes all main categories.
        force_cache (bool): Whether to force caching from the soccerdata library.
    """
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_csv_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    fbref = FBref(leagues=leagues, seasons=seasons)

    try:
        match_logs = {}

        # Fetch all stat types
        for stat_type in stat_types:
            logger.info(f"Fetching {stat_type} data...")
            try:
                data = fbref.read_team_match_stats(
                    stat_type=stat_type,
                    team=team,
                    force_cache=force_cache
                )

                # Handle multi-level columns
                if isinstance(data.columns, pd.MultiIndex):
                    new_cols = []
                    for col in data.columns:
                        if isinstance(col, tuple):
                            parts = [str(part) for part in col if part and str(part) != 'nan']
                            new_cols.append('_'.join(parts))
                        else:
                            new_cols.append(str(col))
                    data.columns = new_cols

                data = data.reset_index(drop=True)
                data.columns = data.columns.astype(str)

                logger.info(f"Columns in {stat_type}: {data.columns.tolist()}")
                match_logs[stat_type] = data

            except Exception as e:
                logger.warning(f"Error fetching {stat_type} data: {e}")
                match_logs[stat_type] = pd.DataFrame()

        # Start with schedule data
        if "schedule" not in match_logs or match_logs["schedule"].empty:
            raise ValueError("Failed to fetch schedule data or schedule data is empty.")

        match_data = match_logs["schedule"]
        logger.info(f"Schedule columns: {match_data.columns.tolist()}")

        # Define base columns and standardize naming
        base_columns = ['date', 'comp', 'venue', 'opponent']
        for col in match_data.columns:
            if col.lower() in [bc.lower() for bc in base_columns]:
                match_data.rename(columns={col: col.lower()}, inplace=True)

        # Merge additional stats
        for stat_type in stat_types[1:]:
            if stat_type in match_logs and not match_logs[stat_type].empty:
                try:
                    merge_df = match_logs[stat_type].copy()
                    for col in merge_df.columns:
                        if col.lower() in [bc.lower() for bc in base_columns]:
                            merge_df.rename(columns={col: col.lower()}, inplace=True)

                    available_merge_cols = [c for c in base_columns if c in merge_df.columns and c in match_data.columns]
                    if available_merge_cols:
                        match_data = match_data.merge(
                            merge_df,
                            on=available_merge_cols,
                            how='left',
                            suffixes=('', f'_{stat_type}')
                        )
                    else:
                        logger.warning(f"No common columns found for merging {stat_type} data")
                except Exception as e:
                    logger.warning(f"Could not merge {stat_type} data: {e}")

        logger.info(f"Final columns after merging: {match_data.columns.tolist()}")


        relevant_columns = {
            'date': 'Date',
            'venue': 'Venue',
            'opponent': 'Opponent',
            'gf': 'Goals For',
            'ga': 'Goals Against',
            'xg': 'Expected Goals (xG)',
            'xga': 'Expected Goals Against (xGA)',
            'poss': 'Possession',
            'sh': 'Shots Total',
            'sot': 'Shots on Target',
            'dist': 'Average Shot Distance',
            'fk': 'Free Kicks',
            'pk': 'Penalties',
            'pkatt': 'Penalties Attempted',
            'cmp': 'Passes Completed',
            'att': 'Passes Attempted',
            'cmp_pct': 'Pass Completion Percentage',
            'totdist': 'Total Pass Distance',
            'prgdist': 'Progressive Pass Distance',
            'kp': 'Key Passes',
            'final_third': 'Passes into Final Third',
            'ppa': 'Passes into Penalty Area',
            'crspa': 'Crosses into Penalty Area',
            'tkl': 'Tackles',
            'tklw': 'Tackles Won',
            'int': 'Interceptions',
            'blocks': 'Blocks',
            'clr': 'Clearances',
            'err': 'Errors',
            'touches': 'Total Touches',
            'succ': 'Successful Take-Ons',
            'att_takes': 'Take-Ons Attempted',
            'rec': 'Passes Received',
            'save_pct': 'Save Percentage',
            'cs': 'Clean Sheets Keeper',
            'pksv': 'Penalties Saved',
            'pkatt_against': 'Penalties Faced',
            'recov': 'Ball Recoveries',
            'aer_won': 'Aerials Won',
            'aer_lost': 'Aerials Lost'
        }

        # Case-insensitive mapping of available columns
        available_columns = {col.lower(): col for col in match_data.columns}
        available_relevant_columns = {
            available_columns[k]: v
            for k, v in relevant_columns.items()
            if k in available_columns
        }

        if not available_relevant_columns:
            raise ValueError("No relevant columns found in the data after merging.")

        # Select and rename columns
        lfc_cleaned_data = match_data[list(available_relevant_columns.keys())].rename(columns=available_relevant_columns)

        # Convert Date to datetime
        lfc_cleaned_data['Date'] = pd.to_datetime(lfc_cleaned_data['Date'], errors='coerce')
        lfc_cleaned_data.dropna(subset=['Date'], inplace=True)

        # Add Home Venue indicator
        if 'Venue' in lfc_cleaned_data.columns:
            lfc_cleaned_data['Home Venue'] = lfc_cleaned_data['Venue'].str.lower().apply(
                lambda x: 1 if 'home' in x else 0)

        # Calculate Clean Sheets
        if 'Goals Against' in lfc_cleaned_data.columns:
            lfc_cleaned_data['Clean Sheets'] = lfc_cleaned_data['Goals Against'].apply(
                lambda x: 1 if pd.notna(x) and x == 0 else 0)

        # Sort by date
        lfc_cleaned_data = lfc_cleaned_data.sort_values('Date')

        # Save to CSV
        lfc_cleaned_data.to_csv(output_csv_path, index=False)
        logger.info(f"Data scraped successfully and saved to {output_csv_path}")

        # Print summary
        logger.info(f"Summary of scraped data:\n"
                    f"Total matches: {len(lfc_cleaned_data)}\n"
                    f"Date range: {lfc_cleaned_data['Date'].min()} to {lfc_cleaned_data['Date'].max()}\n"
                    f"Columns retrieved: {', '.join(lfc_cleaned_data.columns)}")

        return lfc_cleaned_data

    except Exception as e:
        logger.error(f"Error processing data: {e}", exc_info=True)
        raise
