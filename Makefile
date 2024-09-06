#PYTHONPATH=.:$$PYTHONPATH
SHELL=bash
PROJECT_NAME=2024-copa-tutorial-mapie

create_env:
	source ./scripts/create.sh $(PROJECT_NAME);

remove_env:
	source ./scripts/remove.sh $(PROJECT_NAME);

clean:
	rm -rf .ipynb_checkpoints
