cmake_minimum_required(VERSION 3.12)
project(nano-akari)


set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_SHARED_LIBRARY_PREFIX_CXX "")
set(CMAKE_STATIC_LIBRARY_PREFIX_CXX "")
set(AKR_EXT_LIBS "")
set(AKR_EXT_INCLUDES "")
set(AKR_COMPILE_DEFINITIONS "")

option(AKR_BACKEND_EMBREE "Enable cpu backend with embree" OFF)
option(AKR_BACKEND_CUDA "Enable CUDA + Optix7 backend" OFF)
option(AKR_BUILD_GUI "Build GUI" OFF)

find_package(glm CONFIG REQUIRED)
find_package(cereal CONFIG REQUIRED)
find_package(assimp CONFIG REQUIRED)
find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)
set(AKR_EXT_LIBS ${AKR_EXT_LIBS} glm cereal)

message(STATUS "python:  " ${Python_EXECUTABLE})

if(AKR_BACKEND_EMBREE)
    find_package(embree 3 REQUIRED)
    set(AKR_COMPILE_DEFINITIONS AKR_BACKEND_EMBREE)
endif()

if(MSVC)
    set(AKR_CXX_FLAGS /std:c++17 /MP /arch:AVX2 /WX)
    set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ")
    set(CMAKE_STATIC_LINKER_FLAGS  "${CMAKE_STATIC_LINKER_FLAGS} ")
    set(CMAKE_SHARED_LINKER_FLAGS   "${CMAKE_SHARED_LINKER_FLAGS} ")
else()
    set(CMAKE_CXX_FLAGS -fPIC)
    set(AKR_CXX_FLAGS -mavx2)
    set(AKR_CORE_EXT_LIBS ${AKR_CORE_EXT_LIBS} stdc++fs)
endif()

function(set_output_dir target)
    set_target_properties(${target} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/dist"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/dist"
    )
endfunction()



add_subdirectory(lib)
add_subdirectory(app)