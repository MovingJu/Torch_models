port?=8000

repo=test
platform=linux/amd64#,linux/arm/v7,linux/arm64
docker_img_tag=torch_model

.PHONY: build run test docker-run
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
	./save_exit.sh > result.txt &
	docker rm $(docker_img_tag)
	docker run \
	--name $(docker_img_tag) \
	-d \
	--shm-size=4g \
	--gpus all \
	-v $$(pwd)/models:/app/models \
	-p $(port):8000 \
	movingju/$(repo):$(docker_img_tag)

docker-logs:
	docker logs -f $(docker_img_tag)

docker-stop:
	docker stop $(docker_img_tag)

docker-push:
	docker push movingju/$(repo):$(docker_img_tag)

run: 
	uv run main.py

clear:
	rm -r .venv save_exit.log

logs:
	tail -f result.txt