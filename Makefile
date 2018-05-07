.FORCE:
proto: .FORCE
	python -m grpc_tools.protoc -I $$(pwd)/protos --mypy_out=./mincall --python_out=./mincall $$(pwd)/protos/*

USERNAME=""
TAG="latest-py3"

ifeq ($(strip $(USERNAME)), "")
	BASE_TAG=""
else
	BASE_TAG=$(USERNAME)/
endif

docker-build:
	docker build -t $(BASE_TAG)mincall:$(TAG)-gpu -f Dockerfile.gpu .
	docker build -t $(BASE_TAG)mincall:$(TAG) -f Dockerfile .

.PHONY: docker-build

docker-push: docker-build
	docker push $(BASE_TAG)mincall:$(TAG)-gpu
	docker push $(BASE_TAG)mincall:$(TAG)

.PHONY: docker-push

