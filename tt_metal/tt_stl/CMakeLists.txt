add_library(tt_stl INTERFACE)
add_library(Metalium::Metal::STL ALIAS tt_stl)

target_include_directories(tt_stl INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
target_link_libraries(tt_stl INTERFACE Reflect::Reflect)

install(TARGETS tt_stl EXPORT Metal FILE_SET api COMPONENT metal-dev)
