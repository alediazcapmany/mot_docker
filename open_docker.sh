#!/bin/bash

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
xhost +local:docker > /dev/null 2>&1

echo "Abriendo el contenedor de Docker con Ubuntu 22.04 y OpenCV 4.11.0..."

docker run -it --rm \
    --net=host \
    --env="DISPLAY" \
    --volume="$HOME/.Xauthority:/root/.Xauthority:rw" \
    -v "$PROJECT_DIR:/app" \
    yolo_rust_docker /bin/bash
