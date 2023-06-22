import sys

sys.path.insert(0, "C:/Users/evobi/Desktop/SNPio")

import inspect
from snpio.read_input import genotype_data

for name, obj in inspect.getmembers(genotype_data):
    if (
        inspect.isclass(obj)
        or inspect.isfunction(obj)
        or inspect.ismethod(obj)
    ):
        print(f"Docstring for {name}:")
        print(inspect.getdoc(obj))
        print()
