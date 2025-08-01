import streamlit as st
import sqlite3
import pandas as pd
from pathlib import Path

st.title("SQLite Database Inspector")

# File uploader for .db file
uploaded_file = st.file_uploader("Upload SQLite .db file", type=["db", "sqlite", "sqlite3"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with open("temp.db", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    try:
        # Connect to SQLite database
        conn = sqlite3.connect("temp.db")
        cursor = conn.cursor()

        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]

        if not tables:
            st.warning("No tables found in the database.")
        else:
            st.header("Database Overview")
            st.write(f"Number of tables: {len(tables)}")
            
            # Display table selection
            selected_table = st.selectbox("Select a table to inspect", tables)

            if selected_table:
                # Get table schema
                cursor.execute(f"PRAGMA table_info('{selected_table}');")
                schema = cursor.fetchall()
                
                st.header(f"Table: {selected_table}")
                
                # Display schema
                st.subheader("Schema")
                schema_df = pd.DataFrame(
                    schema, 
                    columns=['CID', 'Name', 'Type', 'Not Null', 'Default Value', 'Primary Key']
                )
                st.dataframe(schema_df)

                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM '{selected_table}';")
                row_count = cursor.fetchone()[0]
                st.write(f"Total rows: {row_count}")

                # Display sample data
                st.subheader("Sample Data (First 5 rows)")
                query = f"SELECT * FROM '{selected_table}' LIMIT 5;"
                sample_df = pd.read_sql_query(query, conn)
                st.dataframe(sample_df)

                # Basic statistics
                st.subheader("Basic Statistics")
                df = pd.read_sql_query(f"SELECT * FROM '{selected_table}'", conn)
                
                # Numerical columns statistics
                numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
                if len(numerical_cols) > 0:
                    st.write("Numerical Columns Statistics:")
                    stats_df = df[numerical_cols].describe()
                    st.dataframe(stats_df)
                else:
                    st.write("No numerical columns found for statistical analysis.")

                # Unique values in categorical columns
                categorical_cols = df.select_dtypes(include=['object']).columns
                if len(categorical_cols) > 0:
                    st.write("Categorical Columns Unique Values:")
                    for col in categorical_cols:
                        unique_count = df[col].nunique()
                        st.write(f"{col}: {unique_count} unique values")

        # Clean up
        conn.close()
        if Path("temp.db").exists():
            Path("temp.db").unlink()

    except sqlite3.Error as e:
        st.error(f"Database error: {str(e)}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    st.info("Please upload an SQLite database file to begin.")
