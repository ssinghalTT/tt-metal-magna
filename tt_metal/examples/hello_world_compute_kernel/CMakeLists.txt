cmake_minimum_required(VERSION 3.16..3.30)

add_executable(metal_example_hello_world_compute_kernel)
target_sources(
    metal_example_hello_world_compute_kernel
    PRIVATE
        hello_world_compute_kernel.cpp
        kernels/compute/void_compute_kernel.cpp
)

find_package(Metalium REQUIRED)
target_link_libraries(
    metal_example_hello_world_compute_kernel
    PUBLIC
        Metalium::Metal
    # m
    # pthread
)
set_target_properties(
    ${TEST_TARGET}
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/programming_examples/${SUBDIR}
)
