#!/bin/zsh

pandoc --listings --toc -H snpio/docs/listings-setup.tex README.md --metadata-file=snpio/docs/HEADER.yaml --template=snpio/docs/template.tex --pdf-engine=xelatex -o UserManual.pdf --include-in-header=snpio/docs/admonitions.tex --lua-filter=snpio/docs/admonitions-filter.lua
