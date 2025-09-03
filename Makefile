port?=8000

repo=test
platform=linux/amd64#,linux/arm/v7,linux/arm64
docker_img_tag=torch_model

.PHONY: build run test
docker-build:
	docker buildx build \
		--platform  $(platform) \
		-t movingju/$(repo):$(docker_img_tag) \
		.

docker-test:
	docker build \
	-t movingju/$(repo):$(docker_img_tag) \
	.

docker-run:
	docker rm $(docker_img_tag)
	docker run \
	--name $(docker_img_tag) \
	-d \
	--shm-size=4g \
	--gpus all \
	-v $$(pwd)/modules/Facial_keypoints:/app/modules/Facial_keypoints \
	-p $(port):8000 \
	movingju/$(repo):$(docker_img_tag)

docker-logs:
	docker logs -f $(docker_img_tag)

docker-push:
	docker push movingju/$(repo):$(docker_img_tag)

build:
	cmake . -B build
	cmake --build build

run: 
	uv run main.py

clear:
	rm -r .venv 