#include <iostream>

#include "engine.h"
#include "mcts.h"

std::random_device global_random_device;
std::mt19937 global_random_engine(global_random_device());

// to do:
// most-visited: softmax;
int main() {
	auto net = std::make_shared<FIRNet>();
	//auto net1 = std::make_shared<FIRNet>("FIR-6x6by4_23600.param");
	//auto net2 = std::make_shared<FIRNet>("FIR-6x6by4_9900.param");
	//auto p1 = RandomPlayer("p1");
	//auto p2 = HumanPlayer("jaysinco");
	//auto p3 = MCTSPurePlayer("mcts_pure", 3000);
	//auto p4 = MCTSDeepPlayer("mcts_deep1", net1, 2000);
	//auto p5 = MCTSDeepPlayer("mcts_deep2", net2, 1000);
	//play(p2, p4, false);
	//benchmark(p4, p3, 10, false);
	train_mcts_deep(net);
}