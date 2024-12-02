emcc llama-bindings.c -c -sMEMORY64=1 -Wno-experimental -o llama-bindings.o /lua-5.3.4/src/liblua.a -I/lua-5.3.4/src -I/llamacpp/ -I/llamacpp/common -I/llamacpp/spm-headers -I/llamacpp/ggml/include
emcc llama-run.cpp -c -sMEMORY64=1 -Wno-experimental -o llama-run.o -I/lua-5.3.4/src -I/llamacpp/ -I/llamacpp/common -I/llamacpp/spm-headers -I/llamacpp/ggml/include

emar rcs libaollama.so llama-bindings.o llama-run.o

rm llama-bindings.o llama-run.o


emcc stream-bindings.c -c -sMEMORY64=1 -o stream-bindings.o /lua-5.3.4/src/liblua.a -I/lua-5.3.4/src
emcc stream.c -c -sMEMORY64=1 -o stream.o /lua-5.3.4/src/liblua.a -I/lua-5.3.4/src

emar rcs libaostream.so stream-bindings.o stream.o

rm stream.o stream-bindings.o
