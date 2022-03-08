DATA_OUT = data_out
DATA_IN = data
CONFIG = config.yaml
VERBOSE = ON

export VERBOSE

COLOR = \033[0;34m

# Clean only DATA_OUT folder
clean_data:
	@echo "$(COLOR)Clean data...\033[0m"
	rm -rf $(DATA_OUT)/*

# Clean complete project
clean: clean_data
	@echo "$(COLOR)Clean...\033[0m"
	$(MAKE) -C backend/ clean
	$(MAKE) -C cpp_backend/ clean
	rm -rf .vscode
	rm -rf .pytest_cache
	rm -rf .idea

# Compile c++ backend
compile: clean
	@echo "$(COLOR)Compile...\033[0m"
	$(MAKE) -C cpp_backend/ compile

# Generate mdp from dataset
generate:
	@echo "$(COLOR)Generate...\033[0m"
	python3 -W ignore gen_mdp.py -d_in $(DATA_IN) -d_out $(DATA_OUT) -c $(CONFIG) --plot

# Evaluate and test policy
evaluate:
	@echo "$(COLOR)Evaluate...\033[0m"
	python3 -W ignore eval_mdp.py -d_out $(DATA_OUT) -c $(CONFIG) --plot
