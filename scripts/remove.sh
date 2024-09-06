#!/bin/bash

function remove_venv() {
    rm -rf .venv;
}

remove_venv && (
    echo ""
    echo "*****************************************************************************"
    echo "Successfully cleaned the virtual environment; you are now using this python:"
    echo "$(which python)"
    echo "*****************************************************************************"
    echo ""
)
