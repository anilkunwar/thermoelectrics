import os
import streamlit as st
import subprocess

def run_elmer_solver(sif_file_path, mesh_file_path):
    # Command to run ElmerSolver
    command = f"ElmerSolver {sif_file_path}"

    # Change directory to the mesh file location
    mesh_directory = os.path.dirname(mesh_file_path)
    command = f"cd {mesh_directory} && {command}"

    # Run the command using subprocess
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()

    # Print the output and error messages
    st.text("Output:")
    st.text(output.decode())

    st.text("Error:")
    st.text(error.decode())

def main():
    st.title("ElmerSolver Web Interface")
    st.write("Please provide the path to the SIF file and the mesh file.")

    # File input
    sif_file_path = st.text_input("~/femstlit/teg.sif")
    mesh_file_path = st.text_input("~/femstlit/Mesh_2-t1h0a0t0")

    if sif_file_path and mesh_file_path:
        # Run ElmerSolver
        run_elmer_solver(sif_file_path, mesh_file_path)

if __name__ == "__main__":
    main()



