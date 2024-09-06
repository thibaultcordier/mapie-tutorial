#!/bin/bash

function create_venv() {
    # try to create a .venv with python3
    (
        python3 -m venv .venv &&
        . ./.venv/bin/activate &&
        pip install --upgrade pip &&
        pip install -r requirements.txt
    ) || \
    # if it fails: warn the user (clean the .venv if it was partially created)
    (
        rm -rf .venv &&
        echo "ERROR: failed to create the .venv : do it yourself!" &&
        exit 1
    );
}

create_venv && (
    echo ""
    echo "***************************************************************"
    echo "Successfully created the virtual environment! it is located at:"
    echo "$(pwd)/.venv"
    echo "***************************************************************"
    echo ""
)
