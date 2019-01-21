DIST_DIR ?= ${PWD}/dist
BUILD_DIR ?= ${PWD}/build

.PHONY: container
container:
	docker build -t audio-analysis .

.PHONY: run
run:
	docker run \
		-t \
		-e DISPLAY=${DISPLAY} \
		--device /dev/snd \
		audio-analysis
