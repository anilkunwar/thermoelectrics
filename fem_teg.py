import subprocess
import streamlit as st
import os

def run_command(command):
    print("Executing command:", command)  # Print the command for debugging purposes
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    return output.decode(), error.decode()

def main():
    st.title("Remote Ubuntu Terminal Command")
    
    command = st.text_input("Enter the command to run:")
    
    if st.button("Run"):
        os.environ["PATH"] += ":/usr/bin/ElmerSolver"  # Replace "/path/to/directory" with the actual directory containing the executable
        output, error = run_command(command)
        
        st.subheader("Output:")
        st.code(output)
        
        if error:
            st.subheader("Error:")
            st.code(error)

if __name__ == "__main__":
    main()
