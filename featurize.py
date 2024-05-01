import streamlit as st
import csv
import pandas as pd
from pymatgen.core.composition import Composition
import re

def parse_formula(formula):
    # Regular expression pattern to match elements and their stoichiometric ratios
    pattern = r'([A-Z][a-z]*)(\d*\.?\d*)?'

    # Extract elements from the formula
    elements = re.findall(pattern, formula)
    
    # Return a list of unique elements
    return list(set([element[0] for element in elements]))

def extract_multiplier_and_replace(input_formula):
    # Define a regular expression pattern to find a number after a closing parenthesis
    pattern = r'\)(\d*\.?\d*)'

    # Search for the pattern in the input formula
    match = re.search(pattern, input_formula)

    if match:
        # If a match is found, extract the multiplier
        multiplier = match.group(1)
        multiplier = float(multiplier) if multiplier else 1.0  # Set default multiplier to 1 if empty string

        # Split the input formula based on the pattern
        parts = re.split(pattern, input_formula)

        # Extract the part without the multiplier
        formula_without_multiplier = parts[0]

        # Remove the content before the opening parenthesis
        content_within_parentheses = formula_without_multiplier.split('(')[-1]

        # Find the elements and their stoichiometry within the parentheses
        elements_within_parentheses = re.findall(r'([A-Za-z]+)(\d*\.?\d*)', content_within_parentheses)

        # Multiply the stoichiometry of each element by the multiplier
        modified_elements = [(element, str(float(stoichiometry) * multiplier) if stoichiometry else '0.0') for element, stoichiometry in elements_within_parentheses]

        # Construct the modified formula using the terms before the opening parenthesis and the modified element compositions
        modified_formula = formula_without_multiplier.split('(')[0]  # Get the terms before the opening parenthesis
        modified_formula += ''.join(element + stoichiometry for element, stoichiometry in modified_elements)  # Add the modified element compositions
        
        return modified_formula
    else:
        # If no match is found, return the original formula
        return input_formula


def count_elements(csv_file):
    elements = set()
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            formula = row['Formula']
            elements.update(parse_formula(formula))
    return list(elements)

def featurize_materials(csv_file, available_elements):
    features = []
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            modified_formula = extract_multiplier_and_replace(row['Formula'])
            composition = Composition(modified_formula)
            composition_dict = composition.fractional_composition.as_dict()
            feature_vector = {element: composition_dict.get(element, 0) for element in available_elements}
            features.append(feature_vector)
    return features

# Read the CSV file and count the available elements
csv_file = 'thermoelectric_materials.csv'
available_elements = count_elements(csv_file)

# Display the number of elements and their list
st.write("Number of elements present in the list:", len(available_elements))
st.write("Available elements:", available_elements)

# Featurize the materials
features = featurize_materials(csv_file, available_elements)

# Create a DataFrame to store the featurized values
df = pd.DataFrame(features)

# Load the original dataset
original_df = pd.read_csv(csv_file)

# Insert the featurized values before the 'temperature(K)' column
temp_col_index = original_df.columns.get_loc("temperature(K)")
modified_formulas = original_df['Formula'].apply(extract_multiplier_and_replace)
original_df.insert(temp_col_index, 'modformula', modified_formulas)

# Combine the featurized values with the original dataframe
#df_combined = pd.concat([original_df.iloc[:, :temp_col_index], df, original_df.iloc[:, temp_col_index:]], axis=1)
# Combine the featurized values with the original dataframe
df_combined = pd.concat([original_df.iloc[:, :temp_col_index+1], df, original_df.iloc[:, temp_col_index+1:]], axis=1)
# Calculate the sum of the values of all elemental columns in each row
# Select only the elemental columns
elemental_columns = ["Mg", "Cs", "Co", "Zr", "Se", "Dy", "Pb", "Ga", "O", "Sn", "Yb", "B", "La", "Si", "V", "Fe", "S", "Sc", "Tl", "Zn", "Cl", "Ce", "Er", "Nd", "Pd", "Y", "P", "Ta", "In", "Te", "Ru", "Rb", "Tm", "Tb", "Sb", "Al", "Lu", "Bi", "Pr", "Eu", "Sm", "Ba", "Cr", "Sr", "Ni", "Ca", "As", "Mn", "Mo", "Cd", "Ti", "Nb", "Hf", "Gd", "Ag", "Ge", "Li", "Br", "Au", "I", "N", "Na", "Cu", "Ho", "K"]
elemental_df = df_combined[elemental_columns]

# Calculate the sum of the values of all elemental columns in each row
df_combined['sum_elements'] = elemental_df.sum(axis=1)
# Drop any columns with "Unnamed" in the name
df_combined = df_combined.loc[:, ~df_combined.columns.str.contains('^Unnamed')]

# Allow users to download the combined dataframe in a CSV file
st.write("Download combined data")
st.write(df_combined)
st.download_button(
    label="Download CSV",
    data=df_combined.to_csv().encode('utf-8'),
    file_name='combined_data.csv',
    mime='text/csv'
)


## Display the combined dataframe
#st.write("Download combined data")
#st.write(df_combined)
#st.download_button(
#    label="Download CSV",
#    data=df_combined.to_csv().encode('utf-8'),
#    file_name='combined_data.csv',
#    mime='text/csv'
#)

