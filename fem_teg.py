import streamlit as st
import subprocess

def run_program(program_path):
    try:
        process = subprocess.Popen(program_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        
        # Read and display the log messages in real-time
        for line in process.stdout:
            st.text(line.strip())
            
        # Wait for the process to finish
        process.wait()
        
        st.text(f"Program exited with return code: {process.returncode}")
        
    except FileNotFoundError:
        st.error(f"Error: Program '{program_path}' not found.")

def main():
    st.title("Program Runner")

    program_path = st.text_input("Enter the program path to run")
    if st.button("Run"):
        run_program(program_path)

if __name__ == '__main__':
    main()
