def format_path(list_of_tuples):
    # Extract the first tuple and prepare the initial part of the string
    initial_part = "->".join(list_of_tuples[0])
    
    # Use a list comprehension to generate the rest of the parts excluding the first element of each tuple after the first one
    rest_parts = [f"{r}->{t}" for (_, r, t) in list_of_tuples[1:]]
    
    # Combine the initial part with the rest of the parts
    result = f"{initial_part}->{'->'.join(rest_parts)}"
    
    return result