.PHONY: lint
lint:
	black .
	isort .
	pylint odc
	pylint tests

.PHONY: test
test:
	bash run_tests.sh

.PHONY: build
build:
	bash build_package.sh

.PHONY: clean-tt-cache
clean-tt-cache:
	ls ~/.triton/cache/ | grep -v '\.' | xargs -I {} rm -r ~/.triton/cache/{}
