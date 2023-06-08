import streamlit as st
import subprocess

def run_program(program_path):
    try:
        result = subprocess.run(program_path, capture_output=True, text=True, shell=True)
        output = result.stdout.strip()
        st.code(output, language='text')
    except FileNotFoundError:
        st.error(f"Error: Program '{program_path}' not found.")

def main():
    st.title("Program Runner")

    program_path = st.text_input("Enter the program path to run")
    if st.button("Run"):
        run_program(program_path)

if __name__ == '__main__':
    main()
