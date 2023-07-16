Example script
===============

The `run_snpio.py` script provides a template you can use to get started.

Just type:

.. code-block:: shell

   python3 run_snpio.py

and it will run the example data. There is also a get_args function that will let you provide command-line arguments to set the thresholds, filenames, etc. Just provide get_args to main() like the following:

.. code-block:: python3

    def main(args):
        gd = GenotypeData(args.filename, args.filetype, ...)

    if __name__ == "__main__":
        main(get_args())

