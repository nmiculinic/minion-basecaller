.FORCE:
proto: .FORCE
	python -m grpc_tools.protoc -I $$(pwd)/protos --mypy_out=./mincall --python_out=./mincall $$(pwd)/protos/*

