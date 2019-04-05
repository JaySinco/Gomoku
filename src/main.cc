#include <iostream>

//#include <ctime>
//#include "engine.h"
//#include "mcts.h"
//
//int main() {
//	std::srand(std::time(0));
//
//	auto p1 = RandomPlayer("p1");
//	auto p2 = HumanPlayer("jaysinco");
//	
//	auto p3 = RandomPlayer("p3");
//	auto p4 = MCTSPurePlayer("mcts1", 10000);
//	auto p5 = HumanPlayer("girl");
//	play(p4, p2, false);
//	//benchmark(p4, p2, 120, false);
//}

#include "network.h"

using namespace mxnet::cpp;

void display_shape(const NDArray &nd) {
	std::cout << "Shape(";
	auto shape = nd.GetShape();
	for (int i = 0; i < shape.size(); ++i) {
		if (i != 0) std::cout << ", ";
		std::cout << shape.at(i);
	}
	std::cout << ")";
}

void display_dict(const std::string &name, std::map<std::string, NDArray> &dict) {
	std::cout << "******* " << name << " *******" << std::endl;
	for (const auto &pair : dict) {
		std::cout << pair.first << " => ";
		display_shape(pair.second);
		std::cout << std::endl;
	}
}

void main() {
	//Symbol x("x"), y("y");
	//Symbol plc_m_loss = elemwise_mul(y, log_softmax(x));
	//Symbol y1 = sum(plc_m_loss, dmlc::optional<Shape>(Shape(1)));

	//std::map<std::string, NDArray> args_map;
	//auto ctx = Context::cpu();

	//float X[6] = { 1, 1, 2, 3, 3, 4 };
	//args_map["x"] = NDArray(&X[0], Shape(2, 3), ctx);

	//float Y[6] = { 2, 2, 3, 5, 5, 6 };
	//args_map["y"] = NDArray(&Y[0], Shape(2, 3), ctx);
	//
	//
	////std::cout << args_map["x"] << std::endl;

	//y1.InferArgsMap(ctx, &args_map, args_map);

	//display_dict("args_map1", args_map);
	//auto *exec = y1.SimpleBind(ctx, args_map);

	//exec->Forward(false);

	//std::cout << "out: " << exec->outputs[0] << std::endl;

	FIRNet net;
	float value[1];
	float policy[BOARD_SIZE];
	State game;
	game.next(Move(1, 2));
	net.forward(game, value, policy);
	std::cout << value[0] << std::endl;
	for (int i = 0; i < BOARD_SIZE; ++i) {
		std::cout << policy[i] << " ";
	}
}
