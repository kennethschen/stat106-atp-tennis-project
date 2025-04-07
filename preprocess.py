import os
import logging
import pandas as pd
import glob

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def read_csv_from_local(file_path):
    """Read a CSV file from the local filesystem.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        pandas.DataFrame: The CSV data as a DataFrame
    """
    logging.info(f"Reading file: {file_path}")
    return pd.read_csv(file_path)

def main():
    logging.info("Starting preprocessing script")

    # Get all match CSV files from the data folder
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    csv_files = glob.glob(os.path.join(data_dir, 'atp_matches_*.csv'))
    logging.info(f"Found {len(csv_files)} CSV files")
    
    # Read and concatenate all CSV files
    logging.info("Reading and concatenating CSV files")
    df = pd.concat([read_csv_from_local(file) for file in csv_files])
    logging.info(f"Combined data shape: {df.shape}")

    # Handle 'tourney_date' conversion with error checking
    def safe_date_parse(date_str):
        try:
            return pd.to_datetime(date_str, format="%Y%m%d")
        except ValueError:
            return pd.NaT

    df["tourney_date"] = df["tourney_date"].apply(safe_date_parse)
    df = df.dropna(subset=["tourney_date"])
    logging.info(f"Data shape after date parsing: {df.shape}")

    # Features that cannot be null
    for col in df.columns:
        if col.startswith("h_") or col.startswith("l_"):
            df = df.dropna(subset=[col])

    # For consistency, convert following to features, which are
    # preceeded by winner_ or loser_ but need to be represented as w_ or l_
    cols_to_convert = ["rank", "ht", "age"]
    for col in cols_to_convert:
        df[f"w_{col}"] = pd.to_numeric(df[f"winner_{col}"], errors="coerce")
        df[f"l_{col}"] = pd.to_numeric(df[f"loser_{col}"], errors="coerce")
        df = df.dropna(subset=[f"w_{col}", f"l_{col}"])
        df = df.drop([f"winner_{col}", f"loser_{col}"], axis=1)

    logging.info(f"Final data shape: {df.shape}")

    # Write the combined data to a new CSV in root
    output_file = "combined_atp_matches.csv"
    logging.info(f"Writing combined data to {output_file}")
    df.to_csv(output_file, index=False)
    logging.info(f"Combined data successfully written to {output_file}")
    logging.info("Preprocessing completed")


if __name__ == "__main__":
    main()