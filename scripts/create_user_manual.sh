#!/bin/zsh

pandoc --listings --toc -H docs/listings-setup.tex README.md --metadata-file=docs/HEADER.yaml --template=docs/template.tex -o UserManual.pdf
