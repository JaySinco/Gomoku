cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(gomoku)

add_compile_options(/W1 /D_SCL_SECURE_NO_WARNINGS /source-charset:utf-8)
include_directories(D:/Jaysinco/Cxx/include)
link_directories(D:/Jaysinco/Cxx/lib)

add_executable(gomoku src/mcts.h src/game.h src/network.h src/vars.h src/train.h
                      src/main.cc src/mcts.cc src/game.cc src/network.cc src/train.cc)

set_property(TARGET gomoku PROPERTY CXX_STANDARD 11)
target_link_libraries(gomoku libmxnet.lib)
