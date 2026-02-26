cmake_minimum_required(VERSION 3.20)
project(BPoly LANGUAGES CXX)

# KISS: Simple C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# KISS: Basic warnings and consistent runtime library
if(MSVC)
    add_compile_options(/W4)
    # Ensure consistent runtime library for all configurations
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-Wall -Wextra)
endif()

# KISS: Header-only libraries and test executables
enable_testing()

set(POLY_TYPES bpoly cpoly legpoly hpoly bspoly lagpoly)
set(POLY_TEST_NAMES BPoly CPoly LegPoly HPoly BsPoly LagPoly)

foreach(poly test_name IN ZIP_LISTS POLY_TYPES POLY_TEST_NAMES)
    add_library(${poly} INTERFACE)
    target_include_directories(${poly} INTERFACE include/)

    add_executable(${poly}_test ${poly}_test.cpp)
    target_link_libraries(${poly}_test PRIVATE ${poly})
    add_test(NAME ${test_name}Tests COMMAND ${poly}_test)
endforeach()

message(STATUS "Tests enabled (manual test suite)")

# Comparison test for from_derivatives accuracy
add_executable(compare_from_derivatives compare_from_derivatives.cpp)
target_link_libraries(compare_from_derivatives PRIVATE bpoly)

# Optional Python bindings via nanobind
option(BUILD_PYTHON "Build Python bindings (deriv_poly module)" OFF)

if(BUILD_PYTHON)
    find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

    # Find nanobind - try multiple methods
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
    find_package(nanobind CONFIG REQUIRED)

    # Create the Python module
    nanobind_add_module(deriv_poly NB_STATIC src/bindings.cpp)
    target_include_directories(deriv_poly PRIVATE ${CMAKE_SOURCE_DIR}/include)

    # Install the module
    install(TARGETS deriv_poly LIBRARY DESTINATION .)

    message(STATUS "Python bindings enabled (deriv_poly module)")
endif()

# KISS: Simple installation
install(FILES
    include/bpoly.h include/cpoly.h include/legpoly.h
    include/hpoly.h include/bspoly.h include/lagpoly.h
    include/poly_base.h
    DESTINATION include)

# KISS: Simple build summary
message(STATUS "BPoly: C++${CMAKE_CXX_STANDARD} header-only library")
