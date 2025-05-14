#!/bin/bash

git add -A ./
git commit -m "Tag release"
git push origin master
git tag v1.2.37
git push origin v1.2.37


