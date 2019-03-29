#include <iostream>

#include "game.hpp"
#include "player.hpp"

int main() {
	auto p1 = RandomPlayer("p1");
	auto p2 = HumanPlayer("jaysinco");
	
	auto p3 = RandomPlayer("p3");
	auto p4 = MTCSPurePlayer("mcts", 10000);
	auto p5 = HumanPlayer("girl");
	play(p4, p2, false);
	//benchmark(p1, p4, 120, false);
}