import streamlit as st
import subprocess

# Streamlit web application code
st.title("Run Ubuntu Commands")

# Get user input command
command = st.text_input("Enter the command", "/usr/bin/paraview")

# Run the command
if st.button("Run"):
    try:
        # Execute the command and capture the output
        output = subprocess.check_output(command.split())
        st.code(output.decode("utf-8"))
    except subprocess.CalledProcessError as e:
        st.error(f"Command execution failed with error code {e.returncode}")
        st.error(e.output.decode("utf-8"))
