import streamlit as st
import pandas as pd
import io
import csv
from pathlib import Path

st.title("CSV Empty Value Handler with Fixed Columns")

# Define the expected 76 column names
column_names = [
    '', 'Formula', 'modformula', 'In', 'Tl', 'La', 'Sr', 'Mn', 'Ni', 'Ru', 'Pd', 'Hf', 'Cs', 'Sc', 'Co', 
    'Si', 'Fe', 'Li', 'Cl', 'Yb', 'Te', 'N', 'Ti', 'Cd', 'Zr', 'Y', 'Ga', 'Cr', 'Pr', 'Tm', 'Br', 'Ca', 
    'Mg', 'Rb', 'Au', 'Nd', 'Ce', 'Ho', 'I', 'Ba', 'Se', 'Pb', 'Ge', 'Gd', 'Tb', 'Dy', 'Cu', 'Na', 'Sb', 
    'Bi', 'P', 'As', 'Sm', 'Zn', 'Al', 'Sn', 'Ag', 'Nb', 'Mo', 'V', 'S', 'K', 'Lu', 'O', 'Eu', 'Ta', 'B', 
    'Er', 'temperature(K)', 'seebeck_coefficient(μV/K)', 'electrical_conductivity(S/m)', 
    'thermal_conductivity(W/mK)', 'power_factor(W/mK2)', 'ZT', 'reference', 'sum_elements'
]

# File uploader for CSV file
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# Options for handling empty values
fill_option = st.selectbox(
    "How to handle empty values?",
    options=["NaN", "0", "Custom Value"],
    index=0
)

# Input for custom value if selected
custom_value = None
if fill_option == "Custom Value":
    custom_value = st.text_input("Enter custom value for empty fields", value="")

# Process the CSV file
if uploaded_file is not None:
    try:
        # Save uploaded file temporarily
        temp_csv = "temp.csv"
        with open(temp_csv, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Preprocess CSV to ensure 76 columns
        processed_lines = []
        with open(temp_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for row in reader:
                # Truncate to 76 fields or pad with empty strings
                if len(row) > len(column_names):
                    processed_lines.append(row[:len(column_names)])
                else:
                    processed_lines.append(row + [""] * (len(column_names) - len(row)))

        # Convert to DataFrame with predefined column names
        df = pd.DataFrame(processed_lines[1:], columns=column_names)
        df = df.replace("", None)  # Treat empty strings as None for filling

        # Handle empty values based on user selection
        if fill_option == "NaN":
            pass  # Pandas keeps None as NaN
        elif fill_option == "0":
            df = df.fillna(0)
        elif fill_option == "Custom Value" and custom_value:
            df = df.fillna(custom_value)
        else:
            st.warning("Please provide a custom value for empty fields.")
            st.stop()

        # Display processed data preview
        st.header("Processed Data Preview")
        st.write(f"Number of rows: {len(df)}")
        st.write(f"Number of columns: {len(df.columns)}")
        st.dataframe(df.head())

        # Provide download link for processed CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False, sep=",", quoting=csv.QUOTE_NONNUMERIC)
        st.download_button(
            label="Download Processed CSV",
            data=csv_buffer.getvalue(),
            file_name="processed_data.csv",
            mime="text/csv"
        )

        # Clean up temporary file
        if Path(temp_csv).exists():
            Path(temp_csv).unlink()

    except pd.errors.ParserError as e:
        st.error(f"Error parsing CSV file: {str(e)}. Ensure the CSV uses commas as delimiters and fields with commas are quoted.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    st.info("Please upload a CSV file to begin.")
