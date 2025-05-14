#!/bin/sh
echo
if [ -e .commit ]
    then
    rm .commit
    ./scripts/create_user_manual.sh
    git add pyproject.toml recipe/meta.yaml snpio/docs/source/conf.py snpio/docs/HEADER.yaml template.tex UserManual.pdf
    git commit --amend -C HEAD --no-verify
fi
exit