import pandas as pd
import sqlite3
import os

def csv_to_db(csv_files, db_name):
    """
    Convert multiple CSV files to a SQLite database with separate tables.
    
    Parameters:
    csv_files (list): List of paths to CSV files
    db_name (str): Name of the output SQLite database file
    """
    # Create or connect to the SQLite database
    conn = sqlite3.connect(db_name)
    
    try:
        for csv_file in csv_files:
            # Read CSV file
            df = pd.read_csv(csv_file)
            
            # Get table name from CSV filename (without extension)
            table_name = os.path.splitext(os.path.basename(csv_file))[0]
            
            # Write DataFrame to SQLite table
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            
            print(f"Successfully converted {csv_file} to table {table_name}")
            
    except Exception as e:
        print(f"Error occurred: {e}")
        
    finally:
        # Close the connection
        conn.close()
        print(f"Database {db_name} created successfully")

if __name__ == "__main__":
    # List of CSV files to convert
    csv_files = [
        "regressor_training_history.csv",
        "vae_training_history.csv",
        "thermoelectric_materials.csv"
    ]
    
    # Name of the output database
    db_name = "thermoelectric_data.db"
    
    # Convert CSV files to database
    csv_to_db(csv_files, db_name)