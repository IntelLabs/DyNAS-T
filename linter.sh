#!/bin/bash

ROOT_PATH="."

print_usage() {
cat << EOF
Usage: ./linter.sh [-p <path>][-h]

-p <path> : directory or file path to run linting on. Defaults to current working directory.
-h        : display this message
EOF
}

while getopts p:h flag
do
    case "${flag}" in
        p) ROOT_PATH=${OPTARG};;
        h) print_usage
           exit 1 ;;
        *) print_usage
           exit 1 ;;
    esac
done

echo "Path set to ${ROOT_PATH}"

echo 'Running `isort`'
python -m isort ${ROOT_PATH}

echo 'Running `black`'
# `-S` - `black` will not replace single quotes with double quotes.
python -m black -S ${ROOT_PATH}

echo 'Running `mypy`'
python -m mypy ${ROOT_PATH}
