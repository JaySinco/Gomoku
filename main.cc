#include <iostream>

#include "game.hpp"
#include "player.hpp"

int main() {
	std::srand(std::time(0));

	auto p1 = RandomPlayer("p1");
	auto p2 = HumanPlayer("jaysinco");
	
	auto p3 = RandomPlayer("p3");
	auto p4 = MTCSPurePlayer("mcts1", 10000);
	auto p5 = HumanPlayer("girl");
	play(p4, p2, false);
	//benchmark(p4, p2, 120, false);
}