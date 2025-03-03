#!/bin/sh

find . -name ".DS_Store" -exec rm -rf {} \;
find . -name "__pycache__" -exec rm -rf {} \;
