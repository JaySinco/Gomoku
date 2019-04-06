#include <iostream>

#include <ctime>
#include "engine.h"
#include "mcts.h"

int main() {
	//std::srand(std::time(0));
	//std::srand(1);

	//auto p1 = RandomPlayer("p1");
	//auto p2 = HumanPlayer("jaysinco");
	//
	//auto p3 = RandomPlayer("p3");
	//auto p4 = MCTSPurePlayer("mcts_pure");
	//auto p5 = HumanPlayer("girl");
	//auto p6 = MCTSDeepPlayer("mcts_deep");
	//play(p6, p2, false);
	//benchmark(p4, p6, 10, false);
	//MCTSDeepPlayer p1("p1", "FIR-6x6o4_50.param", 5000);
	MCTSDeepPlayer p2("p2", "FIR-6x6o4_18000.param", 10000);
	//auto p4 = MCTSPurePlayer("mcts_pure", 5000);
	auto p5 = HumanPlayer("jaysinco");
	play(p5, p2, false);
	//benchmark(p2, p5, 10, false);
	//FIRNet net("FIR-6x6o4_18000.param");
	//train_mcts_deep(&net);
}

//#include <mxnet-cpp/MxNetCpp.h>
//using namespace mxnet::cpp;
//
//void display_shape(const NDArray &nd) {
//	std::cout << "Shape(";
//	auto shape = nd.GetShape();
//	for (int i = 0; i < shape.size(); ++i) {
//		if (i != 0) std::cout << ", ";
//		std::cout << shape.at(i);
//	}
//	std::cout << ")";
//}
//
//void display_dict(const std::string &name, std::map<std::string, NDArray> &dict) {
//	std::cout << "******* " << name << " *******" << std::endl;
//	for (const auto &pair : dict) {
//		std::cout << pair.first << " => ";
//		display_shape(pair.second);
//		std::cout << std::endl;
//	}
//}
//
//void main() {
//	Symbol x("x"), yb("yb");
//	Symbol w("w"), b("b");
//	Symbol y = FullyConnected("fc", x, w, b, 1);
//	Symbol loss = MakeLoss(mean(square(elemwise_sub(y, yb))));
//	std::map<std::string, NDArray> args_map;
//	auto ctx = Context::cpu();
//
//	float X1[1*2] = { 1, 1};
//	float X2[3 * 2] = { 1, 2, 3, 4, 5, 6 };
//	float Y[3*1] = { .4, .5, .6 };
//	float W[1 * 2] = { .2, .3 };
//	float B[1] = { 0.5 };
//
//	auto ylabel = NDArray(&Y[0], Shape(2, 3), ctx);
//
//	args_map["x"] = NDArray(&X2[0], Shape(3, 2), ctx);
//	args_map["yb"] = NDArray(&Y[0], Shape(3, 1), ctx);
//	loss.InferArgsMap(ctx, &args_map, args_map);
//	display_dict("args_map", args_map);
//	args_map["w"].SyncCopyFromCPU(W, 2);
//	args_map["b"].SyncCopyFromCPU(B, 1);
//	auto *loss_exec = loss.SimpleBind(ctx, args_map);
//	args_map["x"] = NDArray(&X1[0], Shape(1, 2), ctx); 
//	auto *predict_exec = y.SimpleBind(ctx, args_map);
//	const float learning_rate = 0.01;
//	Optimizer* opt = OptimizerRegistry::Find("sgd");
//	opt->SetParam("lr", learning_rate);
//
//	loss_exec->Forward(true);
//	loss_exec->Backward();
//
//	NDArray::WaitAll();
//	std::cout << "loss_exec: " << loss_exec->outputs[0] << std::endl;
//	
//	auto loss_arg_names = loss.ListArguments();
//	for (int i = 0; i < loss_arg_names.size(); ++i) {
//		if (loss_arg_names[i] == "x" || loss_arg_names[i] == "yb")
//			continue;
//		std::cout << "**" << loss_exec->arg_arrays[i] << " " << loss_exec->grad_arrays[i] << std::endl;
//		opt->Update(i, loss_exec->arg_arrays[i], loss_exec->grad_arrays[i]);
//	}
//	NDArray::WaitAll();
//
//	predict_exec->Forward(false);
//	std::cout << "predict_exec: " << predict_exec->outputs[0] << std::endl;
//}
