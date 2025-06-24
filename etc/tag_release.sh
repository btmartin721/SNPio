#!/bin/bash

git add -A ./
git commit -m "Tag release"
git push origin master
git tag v1.3.0
git push origin v1.3.0


