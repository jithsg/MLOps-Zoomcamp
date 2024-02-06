# Path to the file containing the dependencies
file_path = 'dependencies.txt'

# Processed lines will be stored here
cleaned_lines = []

with open(file_path, 'r') as file:
    for line in file:
        # Split on the first "=" and keep the part before it
        cleaned_line = line.split('=', 1)[0]
        cleaned_lines.append(cleaned_line)

# Assuming you want to overwrite the original file with cleaned lines
# If you want to keep the original file and save this to a new file, change 'file_path' to a new file name
with open(file_path, 'w') as file:
    for line in cleaned_lines:
        file.write(line + '\n')

print(f"Processed {len(cleaned_lines)} lines.")
