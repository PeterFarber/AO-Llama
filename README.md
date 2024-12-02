# AO-Llama

## USAGE

### Pull Submodules

We need to pull the latest aos.
```sh
git submodule update --init --recursive
```


### Build

Just run ./build.sh ( This will build the nessasary libraries inject them and compile the wasm)
```sh
./build.sh
```

### Testing

```sh
cd test-llm
yarn # or npm i
yarn test # or npm run test
```

### WASM-Metering
Need to update the ao-loader with newest version

### newest version
in ggml-cpu.c delete everything behind the GGML_USE_OPENMP macros