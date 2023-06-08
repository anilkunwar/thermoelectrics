import subprocess
import streamlit as st

def run_command(command):
    process = subprocess.Popen(
        command.split(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # Read the output line by line and display it in the Streamlit interface
    for line in iter(process.stdout.readline, ''):
        st.text(line.strip())
    
    process.wait()  # Wait for the process to finish
    return process.returncode

def main():
    st.title("Remote ElmerSolver Execution")
    
    command = "/usr/bin/ElmerSolver_mpi"  # Full path to ElmerSolver_mpi executable
    
    if st.button("Run ElmerSolver"):
        st.text("Executing ElmerSolver...")
        st.text("----------------------------------")
        
        returncode = run_command(command)
        
        st.text("----------------------------------")
        if returncode == 0:
            st.text("ElmerSolver execution completed successfully.")
        else:
            st.text("ElmerSolver execution failed.")
            

if __name__ == "__main__":
    main()
