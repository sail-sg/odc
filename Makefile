.PHONY: lint
lint:
	black .
	isort .
	pylint odc
	pylint tests
	pylint examples

.PHONY: test
test:
	bash run_tests.sh

build_nvshmem:
	bash build_nvshmem_wrapper.sh

.PHONY: build
build:
	bash build_package.sh

.PHONY: clean-tt-cache
clean-tt-cache:
	ls ~/.triton/cache/ | grep -v '\.' | xargs -I {} rm -r ~/.triton/cache/{}
