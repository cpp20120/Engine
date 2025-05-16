```sh
```sh
├── .github/workflows
│   └── build_cmake.yml
├── build(contains generation from cmake(ninja.build) and also contains compile_commands.json
├── cmake (contains cmake scripts for project)
├── docs
│   └── CMakeLists.txt
│   └── Doxyfile.in 
|   └── generate_docs.py
├── include
│   └── *.hpp
├── lib
|   └──CoreMath (contains MathLib)
|		├── cmake (contains cmake scripts for library)
|		├── include
│			└── *.hpp
|		├── src
│			└── CMakeLists.txt
|			└── *.cpp
|		├── test (contains test for math lib)
|		└── CMakeLists.txt
|   └──CoreConcurency (contains ConcurencyLib)
|		├── cmake (contains cmake scripts for library)
|		├── include
│			└── *.hpp
|		├── src
│			└── CMakeLists.txt
|			└── *.cpp
|		├── test (contains test for Concurency lib)
|		└── CMakeLists.txt
|   └──CoreMeta (contains MetaprogrammingLib)
|		├── cmake (contains cmake scripts for library)
|		├── include
│			└── *.hpp
|		├── src
│			└── CMakeLists.txt
|			└── *.cpp
|		├── test (contains test for meta lib)
|		└── CMakeLists.txt
|   └──CoreUtils (contains UtilsLib)
|		├── cmake (contains cmake scripts for library)
|		├── include
│			└── *.hpp
|		├── src
│			└── CMakeLists.txt
|			└── *.cpp
|		├── test (contains test for Utils lib)
|		└── CMakeLists.txt
│── shaders(for graphics project)
│   └── *.frag/.vert
├── src
│   └── CMakeLists.txt
│   └── *.cpp
├── test
│   └── CMakeLists.txt
│   └── test_*.cpp
├── .clang-format
├── .gitignore
├── build_all.(ps/sh) (build all script for unix and windows)
├── CMakeLists.txt
├── CMakePresets.json
├── compile_commands.json -> build/compile_commands.json(for clangd in nvim/vsc)
├── vcpkg.json
├── Dockerfile
├── LICENSE
└── README.md
```
