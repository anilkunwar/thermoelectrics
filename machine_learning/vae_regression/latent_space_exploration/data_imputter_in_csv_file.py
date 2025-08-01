import streamlit as st
import pandas as pd
import io

st.title("CSV Empty Value Handler")

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
        # Read CSV with pandas, treating empty strings as NA
        df = pd.read_csv(uploaded_file, na_values=["", " ", None], keep_default_na=False)

        # Handle empty values based on user selection
        if fill_option == "NaN":
            pass  # Pandas already converts empty values to NaN
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
        df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download Processed CSV",
            data=csv_buffer.getvalue(),
            file_name="processed_data.csv",
            mime="text/csv"
        )

    except pd.errors.ParserError as e:
        st.error(f"Error parsing CSV file: {str(e)}. Please ensure the CSV is well-formed.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    st.info("Please upload a CSV file to begin.")
