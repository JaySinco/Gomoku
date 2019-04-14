#include <iostream>

#include "game.h"
#include "mcts.h"

std::random_device global_random_device;
std::mt19937 global_random_engine(global_random_device());

// to do:
// most-visited: softmax;
int main() {
	auto net = std::make_shared<FIRNet>();
	//auto net1 = std::make_shared<FIRNet>("FIR-8x8by5_1000.param");
	//auto net2 = std::make_shared<FIRNet>("FIR-8x8by5_4100.param");
	//auto p1 = RandomPlayer("p1");
	//auto p2 = HumanPlayer("jaysinco");
	//auto p3 = MCTSDeepPlayer("mcts_pure", net, 1000);
	//auto p4 = MCTSDeepPlayer("mcts_deep1", net1, 2000);
	//auto p5 = MCTSDeepPlayer("mcts_deep2", net1, 400);
	//auto p6 = MCTSDeepPlayer("mcts_deep2", net2, 4000);
	//play(p4, p2, false);
	//benchmark(p4, p5, 10, false);
	train_mcts_deep(net);
}