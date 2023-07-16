import sys
import os

sys.path.append(os.path.normpath(os.getcwd()))

import inspect
from snpio.read_input import genotype_data

docstrings = {}

for name, obj in inspect.getmembers(genotype_data):
    if (
        inspect.isclass(obj)
        or inspect.isfunction(obj)
        or inspect.ismethod(obj)
    ):
        docstrings[name] = inspect.getdoc(obj)

with open("docs/GenotypeData_Docstrings.txt", "w") as fout:
    for k, v in docstrings.items():
        fout.write(f"Function Name: {k}\nDocstring: {v}\n\n")
