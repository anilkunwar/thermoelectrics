#import os
import subprocess
import streamlit as st

# Create Streamlit app
st.title("FEM Simulation for Thermoelectric Generator")

#For running the function in the local computer
#def convert_unv_to_msh(file_path, folder_name):
    ## Run ElmerGrid command to convert UNV to MSH
    #os.system(f"ElmerGrid 8 2 {file_path} -autoclean -out {folder_name}")
# For running the function globally
def convert_unv_to_msh(file_path, folder_name):
    # Run ElmerGrid command to convert UNV to MSH
    command = ["ElmerGrid", "8", "2", file_path, "-autoclean", "-out", folder_name]
    subprocess.run(command)

def generate_sif_file(folder_name, bottom_temperature, top_temperature):
    # Create Elmer sif file content with user-defined temperatures
    sif_content = f'''
    Header
  !Mesh DB "." "Mesh_1coarse"
  !Mesh DB "." "Mesh_2fine"
  Mesh DB "." "Mesh_2-t1h0a0t0"
  Include Path ""
  Results Directory "."
End

Simulation
  Coordinate System = Cartesian
  Simulation Type = Steady
  Output Intervals(1) = 1
  Steady State Max Iterations = 1

  Post File = "result.vtu"
  Use Mesh Names = True
  Coordinate Scaling = 1.0e-6
End

Body 1
  Target Bodies(2) = 3 5
  Name = "Solid_3ceramicsBot Solid_5ceramicsTop" !Ceramics materials
  Equation = 1
  Material = 1
  !Body Force = 1
End

Body 2
  Target Bodies(3) = 4 1 7
  Name = "Solid_4cuConnect Solid_1cuNeg Solid_7cuPos" !Cu Conductor
  Equation = 1
  Material = 2
  !Body Force = 1
End

Body 3
  Target Bodies(1) = 2
  Name = "Solid_2legNeg" !Negative leg
  Equation = 1
  Material = 3
  !Body Force = 2
End

Body 4
  Target Bodies(1) = 6
  Name = "Solid_6legPos" !Positive leg
  Equation = 1
  Material = 4
  !Body Force = 3
End

Equation 1 :: Active Solvers(1) = 1
!Body Force 1 :: Current Source = Real 100
!Body Force 2 :: Current Source = Real 100
!Body Force 3 :: Current Source = Real 100

Material 1
  Electric Conductivity = 1.0E-14 !Seetawan2012-PE
  Density = 1000 
  Heat Capacity = 770
  Heat Conductivity = 20

  Seebeck Coefficient = Real 5.0E-06
End

Material 2
  Electric Conductivity = 5.92E+07 !Seetawan2012-PE
  Density = 1000 
  Heat Capacity = 390
  Heat Conductivity = 400

  Seebeck Coefficient = Real 6.50E-06
End

Material 3
  Electric Conductivity = 8.0E+03 
  Density = 1000 
  Heat Capacity = 200
  Heat Conductivity = 9.79

  Seebeck Coefficient = Real -4.8023E-04
End

Material 4
  Electric Conductivity = 1.022E+04
  Density = 1000 
  Heat Capacity = 200
  Heat Conductivity = 8.55

  Seebeck Coefficient = Real 1.47E-04
End

Solver 1
  Equation = "ThermoElectric"
  Variable = POT[Temperature:1 Potential:1]
  Procedure = "ThermoElectricSolver" "ThermoElectricSolver"
  Element = "p:1"

  Nonlinear System Convergence Tolerance=1e-6
  Nonlinear System Max Iterations=100
  Nonlinear System Newton After Iterations=1
  Nonlinear System Newton After Tolerance=1e-9

  Linear System Solver = "Iterative"
  Linear System Iterative Method = BicgstabL
  Bicgstabl Polynomial Degree = 2
  Linear System Max Iterations = 200
  Linear System Residual Output = 40
  Linear System Preconditioning = Ilu
  Linear System Convergence Tolerance = 1e-8

  Steady State Convergence Tolerance = 1e-6
End

Boundary Condition 1
  Target Boundaries = 19
  Name = Face_19cerBotBot
  !Potential   = 0
  Temperature = {bottom_temperature}
End

Boundary Condition 2
  Target Boundaries = 24
  Name = Face_24cerTopTop
  !Potential   = 0
  Temperature = {top_temperature}
End

Boundary Condition 3
  Target Boundaries = 35
  Name = Face_35cuPosBot
  Potential   = 0
  !Temperature = {bottom_temperature}
End

#Solver 1 :: Reference Norm = Real 221.28738
#RUN
    '''

    # Save the SIF file
    sif_file = os.path.join(folder_name, "case.sif")
    with open(sif_file, "w") as f:
        f.write(sif_content)

    # Run ElmerSolver command
    #os.system(f"ElmerSolver {sif_file}") # For the app run locally
    # Run ElmerSolver command
    command = ["ElmerSolver", sif_file]
    subprocess.run(command)
    
# User input for boundary temperatures
bottom_temperature = st.number_input("Bottom Boundary Temperature", value=496.0)
top_temperature = st.number_input("Top Boundary Temperature", value=500.0)
# the unv file is accesed
if st.button("Run Elmer Simulation"):
    # Specify the file path to the UNV file
    file_path = "~/femstlit/Mesh_2-t1h0a0t0.unv"

    # Create a folder with the same name as the UNV file
    folder_name = os.path.splitext(os.path.basename(file_path))[0]
    os.makedirs(folder_name, exist_ok=True)

    # Convert UNV to MSH
    convert_unv_to_msh(file_path, folder_name)

    # Generate and run ElmerSolver
    generate_sif_file(folder_name, bottom_temperature, top_temperature)

    st.success("Elmer simulation completed.")





# File upload section for computing in the local browser
#uploaded_file = st.file_uploader("Upload UNV File", type="unv")

#if uploaded_file is not None:
    ## Get the uploaded file name
    #file_name = uploaded_file.name

    ## Save the uploaded file with the same name
    #with open(file_name, "wb") as f:
        #f.write(uploaded_file.getbuffer())

    ## Create a folder with the same name as the uploaded file
    #folder_name = os.path.splitext(file_name)[0]
    #os.makedirs(folder_name, exist_ok=True)

    ## Convert UNV to MSH
    #convert_unv_to_msh(file_name, folder_name)

    ## User input for boundary temperatures
    #bottom_temperature = st.number_input("Bottom Boundary Temperature", value=496.0)
    #top_temperature = st.number_input("Top Boundary Temperature", value=500.0)

    #if st.button("Run Elmer Simulation"):
        ## Generate and run ElmerSolver
        #generate_sif_file(folder_name, bottom_temperature, top_temperature)

        #st.success("Elmer simulation completed.")

#else:
    #st.write("Upload a UNV file of the mesh of Thermoelectric Generator")

