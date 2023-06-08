import streamlit as st
import subprocess

def run_elmer_solver(sif_file_path):
    # Command to run ElmerSolver
    command = f"ElmerSolver {sif_file_path}"

    # Run the command using subprocess and capture the output
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True
    )

    # Display the output in the web interface
    with st.empty():
        for line in process.stdout:
            st.text(line.strip())

def main():
    st.title("ElmerSolver Web Interface")
    st.write("Please provide the path to the SIF file.")

    # File input
    sif_file_path = st.text_input("Path to SIF file")

    if sif_file_path:
        # Run ElmerSolver
        run_elmer_solver(sif_file_path)

if __name__ == "__main__":
    main()
