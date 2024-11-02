add_library(tt_metal)
add_library(Metalium::Metal ALIAS tt_metal)

target_sources(
    tt_metal
    PRIVATE
        tt_metal.cpp
        graph/graph_tracking.cpp
)

# if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.23)
#     target_sources(
#         tt_metal
#         PUBLIC
#             FILE_SET api
#             TYPE HEADERS
#             BASE_DIRS api
#             FILES api/tt_metal/host_api.hpp
#     )
# endif()

target_link_libraries(
    tt_metal
    PUBLIC
        Metalium::Metal::Impl
        metal_api
    PRIVATE
        metal_header_directories
        Metalium::Metal::STL
        umd::device
        metal_common_libs
        magic_enum
        fmt::fmt-header-only
        span
        $<$<BOOL:${ENABLE_TRACY}>:TracyClient>
        Metalium::Metal::Impl
        metal_header_directories
        profiler
        common
        jit_build
        llrt
        detail
        distributed
)

target_precompile_headers(
    tt_metal
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tracy/public/tracy/Tracy.hpp
        <functional>
        <map>
        <memory>
        <unordered_map>
        <variant>
        <vector>
)

target_include_directories(
    tt_metal
    PUBLIC
        # $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        # $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/api>
    PRIVATE
        ${UMD_HOME}
)
target_compile_options(tt_metal PUBLIC -Wno-int-to-pointer-cast)
add_dependencies(tt_metal hw_toolchain)

set_target_properties(
    tt_metal
    PROPERTIES
        INSTALL_RPATH
            "${PROJECT_BINARY_DIR}/lib"
        ADDITIONAL_CLEAN_FILES
            "${PROJECT_BINARY_DIR}/lib;${PROJECT_BINARY_DIR}/obj"
)

install(TARGETS tt_metal EXPORT Metal LIBRARY COMPONENT metal FILE_SET api COMPONENT metal-dev)
install(EXPORT Metal NAMESPACE Metalium:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Metalium COMPONENT metal-dev)

# 3rd party dependencies we export in our public API
install(TARGETS
    reflect
    magic_enum
 EXPORT Metal)

add_subdirectory(api)
add_subdirectory(hw)
add_subdirectory(common)
add_subdirectory(jit_build)
add_subdirectory(llrt)
add_subdirectory(tools)
add_subdirectory(impl)
add_subdirectory(detail)
add_subdirectory(distributed)
add_subdirectory(tt_stl)
add_subdirectory(examples)

if(BUILD_PROGRAMMING_EXAMPLES)
    add_subdirectory(programming_examples)
endif()
