## Engine

Currently at arch stage and come implementations(can be changed)


### Build Debug

```sh
mkdir -p build/debug
cd build/debug
cmake --preset debug
cmake --build --preset build-debug
```

### Build Release:
```sh
mkdir -p build/release
cd build/release
cmake --preset release
cmake --build --preset build-release
```

### Vcpkg debug build:
```sh
cmake --preset vcpkg-debug
cmake --build --preset build-vcpkg-debug
```

### Vcpkg release  build:
```sh
cmake --preset vcpkg-release
cmake --build --preset build-vcpkg-release
```


### Build with sanitazers:

## Address sanitizer
```sh
cmake --preset debug-sanitize-address
cmake --build --preset build-debug-sanitize-address
```
## Thread sanitizer
```sh
cmake --preset debug-sanitize-thread
cmake --build --preset build-debug-sanitize-thread
```
## Undefined behavior sanitizer
```sh
cmake --preset debug-sanitize-undefined
cmake --build --preset build-debug-sanitize-undefined```
```
(specify sanitizer what you need)

### Testing

## Run all tests (release build)
```sh
ctest --preset test-all
```

## Run tests with address sanitizer
```sh
ctest --preset test-sanitize-address
```

## Run specific test suite
```sh
ctest --preset test-library1
```

## Run docs generation
```sh
cmake --build . --target docs
```