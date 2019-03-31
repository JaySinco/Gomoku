#include <iostream>

#include "player.hpp"

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


using namespace mxnet::cpp;

int main() {
	auto net = DeepNet();
	DeepNet::display_dict("args_map", net.args_map);
	net.plc_eval->Forward(true);
	DeepNet::display_shape(net.plc_eval->outputs[0]);
	net.val_eval->Forward(true);
	DeepNet::display_shape(net.val_eval->outputs[0]);
	net.loss_eval->Forward(true);
	DeepNet::display_shape(net.loss_eval->outputs[0]);

	//mx_float *haha = new mx_float[4]{ 2, 3, 4, 5 };
	//auto sample = NDArray(haha, Shape(3, 2, 8, 8), ctx);
	////std::cout << sample << std::endl;
	//sample.CopyTo(&args_map["data"]);
	//NDArray::WaitAll();
	//exec->Forward(true);
	//exec2->Forward(true);
	////DeepNet::display_shape(args_map["data"]);
	////DeepNet::display_shape(exec->outputs[0]);
	////std::cout << std::endl << args_map["plc_label"] << std::endl;
	//std::cout << exec2->outputs[0] << std::endl;
	//std::cout << exec->outputs[0] << std::endl;
}
