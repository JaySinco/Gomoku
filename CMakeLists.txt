cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(gomoku)

add_compile_options(/W1 /D_SCL_SECURE_NO_WARNINGS)
include_directories(include D:/Jaysinco/Cxx/include)
link_directories(D:/Jaysinco/Cxx/lib)

add_executable(gomoku src/main.cc src/mcts.cc src/engine.cc src/network.cc 
                      include/mcts.h include/engine.h include/network.h)

set_property(TARGET gomoku PROPERTY CXX_STANDARD 11)
target_link_libraries(gomoku libmxnet.lib)

# source /opt/rh/devtoolset-7/enable
# g++ -L/usr/local/lib/python3.6/site-packages/mxnet -fpermissive -Iinclude -w -std=c++11 -lmxnet -O3 src/engine.cc src/network.cc src/mcts.cc src/main.cc -o gomoku
# export LD_LIBRARY_PATH=/usr/local/lib/python3.6/site-packages/mxnet:$LD_LIBRARY_PATH

