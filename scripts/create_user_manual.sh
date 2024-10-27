#!/bin/bash

pandoc --listings --toc -H snpio/docs/listings-setup.tex README.md --metadata-file=snpio/docs/HEADER.yaml --template=snpio/docs/template.tex -o UserManual.pdf
