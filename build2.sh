#!/bin/bash
set -x

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

LLAMA_CPP_DIR="${SCRIPT_DIR}/build/llamacpp"
AO_LLAMA_DIR="${SCRIPT_DIR}/build/ao-llama"
PROCESS_DIR="${SCRIPT_DIR}/aos/process"
LIBS_DIR="${PROCESS_DIR}/libs"

AO_IMAGE="p3rmaw3b/ao:0.1.4"

EMXX_FLAGS="-msimd128 -O3 -flto -mbulk-memory \
            -s MEMORY64=1 \
            -s SUPPORT_LONGJMP=1"

# Cleanup
rm -rf aos/process/libs
rm -rf libs

# Clone llama.cpp if it doesn't exist
if [ ! -d "${LLAMA_CPP_DIR}" ]; then
    git clone https://github.com/ggerganov/llama.cpp.git ${LLAMA_CPP_DIR}
    cd ${LLAMA_CPP_DIR}
    git checkout tags/b4154 -b b4154
    cd ..
fi

# Build llama.cpp with emscripten
sudo docker run --platform=linux/amd64 -v ${LLAMA_CPP_DIR}:/llamacpp ${AO_IMAGE} sh -c \
    "cd /llamacpp && emcmake cmake \
    -DCMAKE_CXX_FLAGS='${EMXX_FLAGS} -DGGML_NO_OPENMP' \
    -DCMAKE_C_FLAGS='${EMXX_FLAGS} -DGGML_NO_OPENMP' \
    -DCMAKE_EXE_LINKER_FLAGS='${EMXX_FLAGS}' \
    -DCMAKE_SHARED_LINKER_FLAGS='${EMXX_FLAGS}' \
    -S . -B . \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_SERVER=OFF \
    -DGGML_USE_CPU=ON \
    -DLLAMA_NATIVE=OFF \
    -DGGML_METAL=OFF \
    -DGGML_CUDA=OFF \
    -DGGML_VULKAN=OFF \
    -DGGML_OPENBLAS=OFF \
    -DGGML_CUBLAS=OFF \
    -DGGML_HIP=OFF \
    -DGGML_KOMPUTE=OFF \
    -DGGML_USE_OPENMP=OFF \
    -DLLAMA_OPENMP=OFF \
    -DBUILD_SHARED_LIBS=OFF"

# Build the libraries
sudo docker run --platform=linux/amd64 -v ${LLAMA_CPP_DIR}:/llamacpp ${AO_IMAGE} sh -c \
    "cd /llamacpp && CFLAGS='${EMXX_FLAGS} -DGGML_NO_OPENMP -DGGML_USE_OPENMP=OFF' LDFLAGS='${EMXX_FLAGS}' emmake make -j4 llama common ggml-cpu"

# Build ao-llama
sudo docker run --platform=linux/amd64 -v ${LLAMA_CPP_DIR}:/llamacpp -v ${AO_LLAMA_DIR}:/ao-llama ${AO_IMAGE} sh -c \
    "cd /ao-llama && ./build.sh"

# Fix permissions
sudo chmod -R 777 ${LLAMA_CPP_DIR}
sudo chmod -R 777 ${AO_LLAMA_DIR}

# Create directory structure
mkdir -p $LIBS_DIR/llamacpp/common
mkdir -p $LIBS_DIR/llamacpp/ggml/src
mkdir -p $LIBS_DIR/llamacpp/ggml/ggml-cpu

# Copy llama.cpp libraries
cp ${LLAMA_CPP_DIR}/src/libllama.a $LIBS_DIR/llamacpp/libllama.a
cp ${LLAMA_CPP_DIR}/common/libcommon.a $LIBS_DIR/llamacpp/common/libcommon.a
cp ${LLAMA_CPP_DIR}/ggml/src/libggml.a $LIBS_DIR/llamacpp/ggml/src/libggml.a
cp ${LLAMA_CPP_DIR}/ggml/src/ggml-cpu/libggml-cpu.a $LIBS_DIR/llamacpp/ggml/ggml-cpu/libggml-cpu.a
cp ${LLAMA_CPP_DIR}/ggml/src/libggml-base.a $LIBS_DIR/llamacpp/ggml/src/libggml-base.a

# Copy ao-llama files
mkdir -p $LIBS_DIR/ao-llama
cp ${AO_LLAMA_DIR}/libaollama.so $LIBS_DIR/ao-llama/libaollama.so
cp ${AO_LLAMA_DIR}/libaostream.so $LIBS_DIR/ao-llama/libaostream.so
cp ${AO_LLAMA_DIR}/Llama.lua ${PROCESS_DIR}/Llama.lua

# Cleanup .so files
rm -rf ${AO_LLAMA_DIR}/*.so

# Copy config
cp ${SCRIPT_DIR}/config.yml ${PROCESS_DIR}/config.yml

# Build the process module
cd ${PROCESS_DIR} 
docker run -e DEBUG=1 --platform=linux/amd64 -v ./:/src ${AO_IMAGE} ao-build-module

# Copy the process module to test directories
cp ${PROCESS_DIR}/process.wasm ${SCRIPT_DIR}/tests/process.wasm
cp ${PROCESS_DIR}/process.js ${SCRIPT_DIR}/tests/process.js
cp ${PROCESS_DIR}/process.js ${SCRIPT_DIR}/test-llm/process.js
cp ${PROCESS_DIR}/process.wasm ${SCRIPT_DIR}/test-llm/process.wasm

echo "Build completed successfully!"