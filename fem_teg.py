import subprocess
import streamlit as st

def run_command(command):
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    return output.decode()

def main():
    st.title("Remote Ubuntu Terminal Command")
    
    command = st.text_input("Enter the command to run:")
    
    if st.button("Run"):
        output = run_command(command)
        st.code(output)

if __name__ == "__main__":
    main()
