# Usamos Ubuntu 24.04 como base limpia
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# 1. Dependencias del sistema, herramientas y OpenCV
RUN apt-get update && apt-get install -y \
    build-essential cmake git wget unzip curl \
    pkg-config clang libclang-dev \
    libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Descargar y compilar OpenCV 4.11.0 (Solo módulos necesarios)
WORKDIR /opt
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/4.11.0.zip \
    && unzip opencv.zip \
    && mkdir -p build && cd build \
    && cmake ../opencv-4.11.0 \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBUILD_LIST=core,dnn,highgui,imgproc,videoio,objdetect \
        -DWITH_GTK=ON \
        -DBUILD_TESTS=OFF \
        -DBUILD_PERF_TESTS=OFF \
    && make -j$(nproc) \
    && make install \
    && ldconfig \
    && rm -rf /opt/opencv* /opt/build

# 3. Instalar Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# 4. Variables de entorno apuntando a OpenCV y Clang
ENV OPENCV_INCLUDE_PATHS="/usr/local/include/opencv4"
ENV LIBCLANG_PATH="/usr/lib/llvm-18/lib/"
ENV PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:/usr/local/lib64/pkgconfig"

# 4.Instalar el paquete de locales
RUN apt-get update && apt-get install -y locales

# Generar el locale deseado (ejemplo: Inglés US)
RUN locale-gen en_US.UTF-8

# Configurar las variables de entorno
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

WORKDIR /app