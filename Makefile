.DEFAULT_GOAL := help

help:  ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

test:  ## Run the deterministic test suite
	python3 -m unittest discover -s tests

.PHONY: help test
