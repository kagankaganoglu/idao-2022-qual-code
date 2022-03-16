all: build

build:
	@echo 'starting....'
run:
	unzip -qo data.zip
	bash run.sh
train:
	unzip -qo data.zip
	bash train.sh
