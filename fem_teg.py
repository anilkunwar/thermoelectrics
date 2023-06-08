import streamlit as st
import subprocess

def run_elmer_solver(sif_file_path, mesh_file_path):
    # Command to run ElmerSolver
    command = f"ElmerSolver {sif_file_path}"

    # Change directory to the mesh file location
    mesh_directory = "/".join(mesh_file_path.split("/")[:-1])
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

    # File upload
    sif_file = st.file_uploader("Upload SIF file", type=["sif"])
    mesh_file = st.file_uploader("Upload Mesh file", type=["mesh"])

    if sif_file and mesh_file:
        # Save the uploaded files
        sif_file_path = "uploaded_files/case.sif"
        mesh_file_path = "uploaded_files/mesh.mesh"
        with open(sif_file_path, "wb") as sif_out:
            sif_out.write(sif_file.read())
        with open(mesh_file_path, "wb") as mesh_out:
            mesh_out.write(mesh_file.read())

        # Run ElmerSolver
        run_elmer_solver(sif_file_path, mesh_file_path)

if __name__ == "__main__":
    main()







