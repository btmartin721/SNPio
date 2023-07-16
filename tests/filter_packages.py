import re

with open("all_packages.txt", "r") as file:
    lines = file.readlines()

# Regular expression pattern to match the package names
pattern = r"^\S+"

# Filter out the dependencies and keep only the top-level packages
top_level_packages = [
    re.match(pattern, line).group()
    for line in lines
    if not line.startswith("#")
]

# Write the top-level packages to a file
with open("top_level_packages.txt", "w") as file:
    file.write("\n".join(top_level_packages))
