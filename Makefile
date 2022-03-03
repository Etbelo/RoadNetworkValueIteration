# Clean complete project
clean:
	$(MAKE) -C backend/ clean
	$(MAKE) -C cpp_backend/ clean
	rm -rf .vscode
	rm -rf .pytest_cache
	rm -rf .idea
	rm -rf data_out/*.npy
	rm -rf data_out/*.png
	rm -rf data_out/*.csv
	rm -rf data_out/*.p

# Compile c++ backend
compile:
	$(MAKE) -C cpp_backend/ compile

# Generate and evaluate mdp from dataset
generate:
	python3 -W ignore gen_mdp.py --plot

evaluate:
	python3 -W ignore eval_mdp.py --plot