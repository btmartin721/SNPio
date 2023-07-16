import modulefinder


def get_package_imports(package_name):
    # Create a ModuleFinder object
    finder = modulefinder.ModuleFinder()

    # Run the ModuleFinder to analyze the package
    finder.run_script("path/to/your/package/__init__.py")

    # Get the imported modules
    imported_modules = list(finder.modules.keys())

    # Filter out built-in modules and packages
    package_imports = [
        module_name
        for module_name in imported_modules
        if module_name.startswith(package_name)
    ]

    return package_imports


# Provide your package name
package_name = "snpio"

# Get the list of imports for your package
imports = get_package_imports(package_name)

with open("test/modules.txt", "w") as fout:
    # Print the imported modules
    for module in imports:
        print(module)
        fout.write(f"{module}\n")
