import pkg_resources


def get_installed_packages():
    installed_packages = [
        package.key
        for package in pkg_resources.working_set
        if package.location.endswith(
            ".dist-info"
        )  # Filter out individual modules
    ]
    return installed_packages


# Get the list of installed packages
packages = get_installed_packages()

# Print the installed packages
for package in packages:
    print(package)
