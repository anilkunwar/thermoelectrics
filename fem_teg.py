import streamlit as st
import subprocess

def run_program(program_path):
    try:
        process = subprocess.Popen(program_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, shell=True)
        log_frame = st.empty()

        for line in process.stdout:
            log_frame.text(line.strip())

        process.wait()
    except FileNotFoundError:
        st.error(f"Error: Program '{program_path}' not found.")

def main():
    st.title("Program Runner")

    program_path = st.text_input("Enter the program path to run")
    if st.button("Run"):
        run_program(program_path)

if __name__ == '__main__':
    main()
