#pragma once

#include <iomanip>
#include <limits>
#include <string>
#include <sstream>

#include "game.hpp"

struct Player {
	Player() {}
	virtual void reset() = 0;
	virtual const std::string &name() const = 0;
	virtual Move play(const State &state) = 0;
	virtual ~Player() {};
};

inline Player &play(Player &p1, Player &p2, bool silent=true) {
	const std::map<Color, Player*> colorMap {
		{Color::Black, &p1}, 
		{Color::White, &p2}, 
		{Color::Empty, nullptr}
	};
	auto game = State();
	p1.reset();
	p2.reset();
	int turn = 0;
	while (!game.over()) {
		auto player = colorMap.at(game.current());
		auto act = player->play(game);
		game.next(act);
		++turn;
		if (!silent) std::cout << game << std::endl;
	}	
	auto winner = colorMap.at(game.winner());
	if (!silent) std::cout << "winner: " 
		<< (winner == nullptr ? "no winner, even!" : winner->name()) << std::endl;
	return *winner;
}

inline void benchmark(Player &p1, Player &p2, int round=1000, bool silent=true) {
	assert(round > 0);
	int p1win = 0, p2win = 0, even = 0;
	Player *temp = nullptr, *pblack = &p1, *pwhite = &p2;
	for (int i = 0; i < round; ++i) {
		temp = pblack, pblack = pwhite, pwhite = temp;
		Player *winner = &play(*pblack, *pwhite);
		if (winner == nullptr)
			++even;
		else if (winner == &p1)
			++p1win;
		else {
			assert(winner == &p2);	
			++p2win;
		}
		if (!silent) {
			std::cout << std::setfill('0') 
				<< "\rscore: total=" << std::setw(4) << i+1 << ", " 
				<< p1.name() << "=" << std::setw(4) << p1win << ", "
				<< p2.name() << "=" << std::setw(4) << p2win;
			std::cout.flush();
		}
	}
	if (!silent) { std::cout << std::endl; }
	float p1prob = float(p1win) / float(round);
	float p2prob = float(p2win) / float(round);
	std::cout << "benchmark(WIN%): " << p1.name() << "=" << p1prob << ", "
		<< p2.name() << "=" << p2prob << ", sim=" << round << std::endl;
}

class RandomPlayer: public Player {
	std::string id;
public:
	RandomPlayer(const char *name) : id(name) {}
	void reset() override {}
	const std::string &name() const override { return id; }
	Move play(const State &state) override { return state.getOpts()[0]; }
	~RandomPlayer() {};
};

class HumanPlayer: public Player {
	std::string id;
public:
	HumanPlayer(const char *name) : id(name) {}
	void reset() override {}
	const std::string &name() const override { return id; }
	Move play(const State &state) override {
		std::string buf, sep;
		int col, row;
		while (true) {
			std::cout << state.current() << "(" << id << "): ";
			std::cout.flush();
			if (std::getline(std::cin, buf)) {
				std::istringstream line(buf);
				if (line >> col >> sep >> row) {
					auto mv = Move(col, row);
					if (state.valid(mv))
						return mv;
				}
			}
		}	
	}
	~HumanPlayer() {};		
};

class MTCSPurePlayer: public Player {
	std::string id;
	int itermax;
	MCTSNode *root;
public:
	MTCSPurePlayer(const char *name, int itermax) 
		: id(name), itermax(itermax) {
		root = new MCTSNode(nullptr, 1.0f);
	}
	~MTCSPurePlayer() { delete root; }
	const std::string &name() const override { return id; }
	void reset() override {
		delete root;
		root = new MCTSNode(nullptr, 1.0f);
	}
	Move play(const State &state) override {
		if (!(state.lastmv() == Move()) && !root->isLeaf()) {
			MCTSNode *temp = root->cut(state.lastmv());
			delete root;
			root = temp;
		}	
		for (int i = 0; i < itermax; ++i) {
			State stateCopy(state);
			MCTSNode *node = root;
			while (!node->isLeaf()) {
				auto mvNode = node->select(1.0f);
				node = mvNode.second;
				stateCopy.next(mvNode.first);
			}
			Color enemy = stateCopy.current();
			Color winner = stateCopy.winner();
			if (winner == Color::Empty) {
				if (stateCopy.getOpts().size() > 0) {
					std::vector<std::pair<Move, float>> mvPriors;
					for (const auto mv : stateCopy.getOpts()) {
						mvPriors.push_back(std::make_pair(mv, 1.0f));
					}
					node->expand(mvPriors);
				}
				winner = stateCopy.nextRandTillEnd();
			}
			float leafValue;
			if (winner == enemy) leafValue = -1.0f;
			else if(winner == ~enemy) leafValue = 1.0f;
			else {
				assert(winner == Color::Empty);
				leafValue = 0.5f;
			}
			node->updateRec(leafValue);
		}
		Move act = root->mostVisted();
		MCTSNode *temp = root->cut(act);
		delete root;
		root = temp;
		return act;
	}
};

