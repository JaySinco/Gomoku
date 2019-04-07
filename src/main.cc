#include <iostream>

#include "engine.h"
#include "mcts.h"

std::mt19937 global_random_engine(1);

// to do:
// most-visited: softmax;
int main() {
	auto net = std::make_shared<FIRNet>();
	//auto net1 = std::make_shared<FIRNet>("FIR-6x6by4_100.param");
	//auto net2 = std::make_shared<FIRNet>("FIR-6x6by4_4800.param");
	//auto p1 = RandomPlayer("p1");
	//auto p2 = HumanPlayer("jaysinco");
	//auto p3 = MCTSPurePlayer("mcts_pure");
	//auto p4 = MCTSDeepPlayer("mcts_deep1", net1);
	//auto p5 = MCTSDeepPlayer("mcts_deep2", net2);
	//play(p4, p5, false);
	//benchmark(p4, p5, 10, false);
	train_mcts_deep(net);
}
