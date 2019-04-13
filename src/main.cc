#include <iostream>

#include "game.h"
#include "mcts.h"

std::random_device global_random_device;
std::mt19937 global_random_engine(global_random_device());

// to do:
// most-visited: softmax;
int main() {
	auto net = std::make_shared<FIRNet>("FIR-8x8by5_init_600.param");
	//auto net1 = std::make_shared<FIRNet>("FIR-8x8by5_600.param");
	//auto net2 = std::make_shared<FIRNet>("FIR-8x8by5_4100.param");
	//auto p1 = RandomPlayer("p1");
	//auto p2 = HumanPlayer("jaysinco");
	//auto p3 = MCTSDeepPlayer("mcts_pure", net1, 1000);
	//auto p4 = MCTSDeepPlayer("mcts_deep1", net, 1000);
	//auto p5 = MCTSDeepPlayer("mcts_deep2", net, 400);
	//auto p6 = MCTSDeepPlayer("mcts_deep2", net2, 4000);
	//play(p2, p3, false);
	//benchmark(p4, p5, 10, false);
	train_mcts_deep(net);
}