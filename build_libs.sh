#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

LLAMA_CPP_DIR="${SCRIPT_DIR}/build/llamacpp"
AO_LLAMA_DIR="${SCRIPT_DIR}/build/ao-llama"
LIBS_DIR="${SCRIPT_DIR}/process/libs"

AO_IMAGE="aoc:latest" # TODO: Change to remote image when ready

EMXX_CFLAGS="-s MEMORY64=1 -O3 -msimd128 -fno-rtti -DNDEBUG \
	-flto=full -s BUILD_AS_WORKER=1 -s EXPORT_ALL=1 \
	-s EXPORT_ES6=1 -s MODULARIZE=1 -s INITIAL_MEMORY=800MB \
	-s MAXIMUM_MEMORY=4GB -s ALLOW_MEMORY_GROWTH -s FORCE_FILESYSTEM=1 \
	-s EXPORTED_FUNCTIONS=_main -s EXPORTED_RUNTIME_METHODS=callMain -s \
	NO_EXIT_RUNTIME=1 -Wno-unused-command-line-argument -Wno-experimental"

# Clone llama.cpp if it doesn't exist
rm -rf ${LLAMA_CPP_DIR}
rm -rf libs
if [ ! -d "${LLAMA_CPP_DIR}" ]; then \
	git clone https://github.com/ggerganov/llama.cpp.git ${LLAMA_CPP_DIR}; \
	cd ${LLAMA_CPP_DIR}; git checkout tags/b3233 -b b3233; \
fi
cd ..

# Patch llama.cpp to remove alignment asserts
sed -i.bak 's/#define ggml_assert_aligned.*/#define ggml_assert_aligned\(ptr\)/g' ${LLAMA_CPP_DIR}/ggml.c
sed -i.bak '/.*GGML_ASSERT.*GGML_MEM_ALIGN == 0.*/d' ${LLAMA_CPP_DIR}/ggml.c

# Build llama.cpp into a static library with emscripten
sudo docker run -v ${LLAMA_CPP_DIR}:/llamacpp ${AO_IMAGE} sh -c \
		"cd /llamacpp && emcmake cmake -DCMAKE_CXX_FLAGS='${EMXX_CFLAGS}' -S . -B . -DLLAMA_BUILD_EXAMPLES=OFF"

sudo docker run -v ${LLAMA_CPP_DIR}:/llamacpp ${AO_IMAGE} sh -c \
		"cd /llamacpp && emmake make llama common EMCC_CFLAGS='${EMXX_CFLAGS}'" 

sudo docker run -v ${LLAMA_CPP_DIR}:/llamacpp  -v ${AO_LLAMA_DIR}:/ao-llama ${AO_IMAGE} sh -c \
		"cd /ao-llama && ./build.sh"

# Fix permissions
sudo chmod -R 777 ${LLAMA_CPP_DIR}
sudo chmod -R 777 ${AO_LLAMA_DIR}

# Copy llama.cpp to the libs directory
mkdir -p $LIBS_DIR/llamacpp/common
cp ${LLAMA_CPP_DIR}/libllama.a $LIBS_DIR/llamacpp/libllama.a
cp ${LLAMA_CPP_DIR}/common/libcommon.a $LIBS_DIR/llamacpp/common/libcommon.a

# Copy ao-llama to the libs directory
mkdir -p $LIBS_DIR/ao-llama
cp ${AO_LLAMA_DIR}/libaollama.so $LIBS_DIR/ao-llama/libaollama.so
cp ${AO_LLAMA_DIR}/libaostream.so $LIBS_DIR/ao-llama/libaostream.so

# Remove .so files
rm -rf ${AO_LLAMA_DIR}/*.so