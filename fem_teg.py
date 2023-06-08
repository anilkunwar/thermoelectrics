import streamlit as st
import subprocess

def run_program(program_name):
    try:
        result = subprocess.run(program_name, capture_output=True, text=True)
        output = result.stdout.strip()
        st.code(output, language='text')
    except FileNotFoundError:
        st.error(f"Error: Program '{program_name}' not found.")

def main():
    st.title("Program Runner")

    program_name = st.text_input("Enter the program name to run")
    if st.button("Run"):
        run_program(program_name)

if __name__ == '__main__':
    main()
