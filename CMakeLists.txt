cmake_minimum_required(VERSION 3.16)

project(GANeuralNetWasm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(nnetga nnetga.cpp)

target_compile_options(nnetga PRIVATE -O2)
target_link_options(nnetga PRIVATE
    "-sMODULARIZE=1"
    "-sEXPORT_ES6=1"
    "-sENVIRONMENT=web,worker"
    "-sALLOW_MEMORY_GROWTH=1"
    "-sWASM=1"
    "--bind"
)

set_target_properties(nnetga PROPERTIES
    OUTPUT_NAME "nnetga"
)