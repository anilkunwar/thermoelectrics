import streamlit as st
import subprocess

def run_program(program_path):
    try:
        result = subprocess.run(program_path, capture_output=True, text=True, shell=True)
        output = result.stdout.strip()
        with st.beta_expander("Show Output"):
            st.code(output, language='text')
        st.markdown(get_download_link(output), unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"Error: Program '{program_path}' not found.")

def get_download_link(output):
    output_encoded = output.encode()
    b64 = base64.b64encode(output_encoded).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="output.txt">Download Output</a>'
    return href

def main():
    st.title("Program Runner")

    program_path = st.text_input("Enter the program path to run")
    if st.button("Run"):
        run_program(program_path)

if __name__ == '__main__':
    main()
