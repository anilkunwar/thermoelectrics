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

        # Extract the last part of the formula (to be replaced)
        last_part_match = re.search(r'([A-Za-z]+)(\d*\.?\d*)$', formula_without_multiplier)
        element = last_part_match.group(1)
        stoichiometry = last_part_match.group(2)
        print("Last part element:", element)
        print("Last part stoichiometry:", stoichiometry)

        # Multiply the last part with the multiplier
        multiplied_stoichiometry = str(float(stoichiometry) * float(multiplier))
        print("Multiplied last part:", multiplied_stoichiometry)

        # Replace the last part of the formula with the multiplied result
        modified_formula = formula_without_multiplier[:last_part_match.start(2)] + multiplied_stoichiometry
        print("Modified Formula:", modified_formula)
    else:
        # If no match is found, print that no multiplier is present
        print("No multiplier found in the input formula.")

# Test the function
input_formula = "Bi2(Te1.5)2"
extract_multiplier_and_replace(input_formula)

