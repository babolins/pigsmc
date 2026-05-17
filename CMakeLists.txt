cmake_minimum_required(VERSION 3.15)
project(pigsmc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(PIGSMC_ENABLE_LTO "Enable link-time optimization" OFF)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 3.3 CONFIG REQUIRED)

pybind11_add_module(_engine MODULE src/pigsmc/_engine.cpp)

target_link_libraries(_engine PRIVATE Eigen3::Eigen)

target_compile_options(_engine PRIVATE
    -O3
    -march=native
    -fvisibility=hidden
)

if(PIGSMC_ENABLE_LTO)
    set_property(TARGET _engine PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

install(TARGETS _engine DESTINATION pigsmc)
