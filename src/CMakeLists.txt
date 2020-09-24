find_package(JlCxx)
get_target_property(JlCxx_location JlCxx::cxxwrap_julia LOCATION)
get_filename_component(JlCxx_location ${JlCxx_location} DIRECTORY)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${JlCxx_location}")

add_library("duneistl-julia" SHARED libduneistl-julia.cc)

target_link_dune_default_libraries("duneistl-julia")
target_link_libraries("duneistl-julia" JlCxx::cxxwrap_julia JlCxx::cxxwrap_julia_stl)

install(TARGETS duneistl-julia DESTINATION lib)