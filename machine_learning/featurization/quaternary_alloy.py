import re

def extract_multiplier_and_replace(input_formula):
    # Define a regular expression pattern to find a number after a closing parenthesis
    pattern = r'\)(\d*\.?\d*)'

    print("Input Formula:", input_formula)

    # Search for the pattern in the input formula
    match = re.search(pattern, input_formula)

    if match:
        # If a match is found, extract the multiplier
        multiplier = match.group(1)
        print("Multiplier:", multiplier)

        # Split the input formula based on the pattern
        parts = re.split(pattern, input_formula)

        # Extract the part without the multiplier
        formula_without_multiplier = parts[0]
        print("Formula without multiplier:", formula_without_multiplier)

        # Remove the content before the opening parenthesis
        content_within_parentheses = formula_without_multiplier.split('(')[-1]
        print("Content within parentheses:", content_within_parentheses)

        # Find the elements and their stoichiometry within the parentheses
        elements_within_parentheses = re.findall(r'([A-Za-z]+)(\d*\.?\d*)', content_within_parentheses)
        print("Elements within parentheses:", elements_within_parentheses)

        # Multiply the stoichiometry of each element by the multiplier
        modified_elements = [(element, str(float(stoichiometry) * float(multiplier))) for element, stoichiometry in elements_within_parentheses]
        print("Modified elements:", modified_elements)

        # Construct the modified formula using the terms before the opening parenthesis and the modified element compositions
        modified_formula = formula_without_multiplier.split('(')[0]  # Get the terms before the opening parenthesis
        modified_formula += ''.join(element + stoichiometry for element, stoichiometry in modified_elements)  # Add the modified element compositions
        print("Modified Formula:", modified_formula)
    else:
        # If no match is found, print that no multiplier is present
        print("No multiplier found in the input formula.")

# Test the function
input_formula = "BiSb(Br0.02Se0.98)3"
extract_multiplier_and_replace(input_formula)

