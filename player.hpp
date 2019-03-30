#pragma once

#include <iomanip>
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
	const std::map<Color, Player*> player_color {
		{Color::Black, &p1}, 
		{Color::White, &p2}, 
		{Color::Empty, nullptr}
	};
	auto game = State();
	p1.reset();
	p2.reset();
	int turn = 0;
	while (!game.over()) {
		auto player = player_color.at(game.current());
		auto act = player->play(game);
		game.next(act);
		++turn;
		if (!silent) std::cout << game << std::endl;
	}	
	auto winner = player_color.at(game.get_winner());
	if (!silent) std::cout << "winner: " 
		<< (winner == nullptr ? "no winner, even!" : winner->name()) << std::endl;
	return *winner;
}

inline void benchmark(Player &p1, Player &p2, int round=100, bool silent=true) {
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
	float eprob = float(even) / float(round);
	std::cout << "benchmark player win probality: " << p1.name() << "=" << p1prob << ", "
		<< p2.name() << "=" << p2prob << ", even=" << eprob << ", sim=" << round << std::endl;
}

class RandomPlayer: public Player {
	std::string id;
public:
	RandomPlayer(const char *name) : id(name) {}
	void reset() override {}
	const std::string &name() const override { return id; }
	Move play(const State &state) override { return state.get_options()[0]; }
	~RandomPlayer() {};
};

class HumanPlayer: public Player {
	std::string id;
	bool get_move(int &row, int &col) {
		std::string line, srow;
		if (!std::getline(std::cin, line))
			return false;
		std::istringstream line_stream(line);
		if (!std::getline(line_stream, srow, ',') || !(line_stream >> col))
			return false;
		std::istringstream row_stream(srow);
		if (!(row_stream >> row))
			return false;
		return true;
	}
public:
	HumanPlayer(const char *name) : id(name) {}
	void reset() override {}
	const std::string &name() const override { return id; }
	Move play(const State &state) override {
		int col, row;
		while (true) {
			std::cout << state.current() << "(" << id << "): ";
			std::cout.flush();
			if (get_move(row, col)) {
				auto mv = Move(row, col);
				if (state.valid(mv))
					return mv;
			}
		}	
	}
	~HumanPlayer() {};		
};

class MTCSPurePlayer: public Player {
	std::string id;
	int itermax;
	float c_puct;
	MCTSNode *root;
	void swap_root(MCTSNode * new_root) {
		delete root;
		root = new_root;
	}
public:
	MTCSPurePlayer(const char *name, int itermax=10000, float c_puct=5.0f)
		: id(name), itermax(itermax), c_puct(c_puct){
		root = new MCTSNode(nullptr, 1.0f);
	}
	~MTCSPurePlayer() { delete root; }
	const std::string &name() const override { return id; }
	void reset() override {
		delete root;
		root = new MCTSNode(nullptr, 1.0f);
	}
	Move play(const State &state) override {
		if (!(state.get_last().z() == NO_MOVE_YET) && !root->is_leaf())
			swap_root(root->cut(state.get_last()));
		for (int i = 0; i < itermax; ++i) {
			State state_copied(state);
			MCTSNode *node = root;
			while (!node->is_leaf()) {
				auto move_node = node->select(c_puct);
				node = move_node.second;
				state_copied.next(move_node.first);
			}
			Color enemy_side = state_copied.current();
			Color winner = state_copied.get_winner();
			if (winner == Color::Empty) {
				int n_options = state_copied.get_options().size();
				if (n_options > 0) {
					std::vector<std::pair<Move, float>> move_priors;
					for (const auto mv : state_copied.get_options()) {
						move_priors.push_back(std::make_pair(mv, 1.0f/float(n_options)));
					}
					node->expand(move_priors);
				}
				winner = state_copied.next_rand_till_end();
			}
			float leaf_value;
			if (winner == enemy_side) 
				leaf_value = -1.0f;
			else if (winner == ~enemy_side) 
				leaf_value = 1.0f;
			else 
				leaf_value = 0.0f;;
			node->update_recursive(leaf_value);
		}
		Move act = root->most_visted();
		swap_root(root->cut(act));
		return act;
	}
};

