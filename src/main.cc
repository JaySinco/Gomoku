#include <iostream>

#include "game.h"
#include "mcts.h"

std::random_device global_random_device;
std::mt19937 global_random_engine(global_random_device());

int main() {
	show_global_cfg(std::cout);
	auto net = std::make_shared<FIRNet>();
	//auto net1 = std::make_shared<FIRNet>("FIR-6x6by4_4850.param");
	//auto net2 = std::make_shared<FIRNet>("FIR-8x8by5_4300.param");
	//auto p1 = RandomPlayer("p1");
	//auto p2 = HumanPlayer("jaysinco");
	//auto p3 = MCTSPurePlayer("mcts_pure", 100000);
	//auto p4 = MCTSDeepPlayer("mcts_deep1", net1, 1000);
	//auto p5 = MCTSDeepPlayer("mcts_deep2", net2, 400);
	//auto p6 = MCTSDeepPlayer("mcts_deep2", net2, 4000);
	//play(p2, p4, false);
	//benchmark(p2, p4, 10, false);
	train_mcts_deep(net);
}