cmake_minimum_required(VERSION 3.10.1)

project(gomoku)

include_directories(D:/Jaysinco/Cxx/include)
link_directories(D:/Jaysinco/Cxx/lib)
add_compile_options(/W1 /D_SCL_SECURE_NO_WARNINGS)

add_executable(gomoku main.cc game.hpp player.hpp)
target_link_libraries(gomoku libmxnet.lib)