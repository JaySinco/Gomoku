#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <iomanip>
#include <iostream>
#include <map>
#include <vector>

/*
3 * 3 board looks like:
  0 1 2
 ------- Col
0|0 1 2
1|3 4 5
2|6 7 8
 |
Row  => move z(5) = (x(1), y(2))
*/

#define ON_BOARD(row, col) (row >= 0 && row < BOARD_MAX_ROW && col >= 0 && col < BOARD_MAX_COL)

constexpr int FIVE_IN_ROW = 4;
constexpr int BOARD_MAX_ROW = 6;
constexpr int BOARD_MAX_COL = 6;
constexpr int BOARD_SIZE = BOARD_MAX_ROW * BOARD_MAX_COL;

constexpr bool DEBUG_MCTS_NODE = false;

enum class Color {Empty, Black, White};

inline Color operator~(const Color c) {
	Color opposite;
	switch (c) {
	case Color::Black: opposite = Color::White; break;
	case Color::White: opposite = Color::Black; break;
	case Color::Empty: opposite = Color::Empty; break;
	}
	assert(opposite != Color::Empty);
	return opposite;
}

inline std::ostream &operator<<(std::ostream &out, Color c) {
	switch (c) {
	case Color::Empty: out << "  "; break;
	case Color::Black: out << "●"; break;
	case Color::White: out << "○"; break;
	}
	return out;
}

const int NO_MOVE_YET = -1;

class Move {
	int index;
public:
	Move(int z) : index(z) {
		assert((z >= 0 && z < BOARD_SIZE) || z == NO_MOVE_YET);
	}
	Move(int row, int col) { 
		assert(ON_BOARD(row, col));
		index = row * BOARD_MAX_COL + col; 
	}
	Move(const Move &mv) : index(mv.z()) {}
	int z() const { return index; }
	int r() const { return index / BOARD_MAX_COL; }
	int c() const { return index % BOARD_MAX_COL; }
	bool operator<(const Move &right) const { return index < right.index; }
	bool operator==(const Move &right) const { return index == right.index; }
};

inline std::ostream &operator<<(std::ostream &out, Move mv) {
	return out << "(" << mv.r() << ", " << mv.c() << ")";
}

class Board {
	Color grid[BOARD_SIZE];
public:
	Board() : grid{ Color::Empty } {}
	Color get(Move mv) const { 
		return grid[mv.z()];
	}
	void put(Move mv, Color c) {
		assert(get(mv) == Color::Empty);
		grid[mv.z()] = c;
	}
	void push_valid(std::vector<Move> &set) const {
		for (int i = 0; i < BOARD_SIZE; ++i)
			if (get(Move(i)) == Color::Empty)
				set.push_back(Move(i));
		
		std::random_shuffle(set.begin(), set.end());
	}
	bool win_from(Move mv) const {
		if (mv.z() == NO_MOVE_YET) return false;
		Color side = get(mv);
		assert(side != Color::Empty);
		int direct[4][2] = { {0, 1}, {1, 0}, {-1, 1}, {1, 1} };
		int sign[2] = { 1, -1 };
		for (auto d : direct) {
			int total = 0;
			for (auto s : sign) {
				Move probe = mv;
				while (get(probe) == side) {
					++total;
					auto r = probe.r() + d[0] * s;
					auto c = probe.c() + d[1] * s;
					if (!ON_BOARD(r, c)) break;
					probe = Move(r, c);
				}
			}
			if (total - 1 >= FIVE_IN_ROW) { return true; }
		}
		return false;
	}
};

inline std::ostream &operator<<(std::ostream &out, const Board &board) {
	out << " # ";
	for (int c = 0; c < BOARD_MAX_COL; ++c)
		out << std::right << std::setw(2) << c % 10 << " ";
	out << "\n";
	for (int r = 0; r < BOARD_MAX_ROW; ++r) {
		out << std::right << std::setw(2) << r % 10;
		for (int c = 0; c < BOARD_MAX_COL; ++c)
			out << "|" << board.get(Move(r, c));
		out << "|\n";
	}
	return out;
}

class State {
	friend std::ostream &operator<<(std::ostream &out, const State &state);
	Board board;
	Move last;
	Color winner;
	std::vector<Move> opts;
public:
	State() : last(NO_MOVE_YET), winner(Color::Empty) { 
		board.push_valid(opts); 
	}
	State(const State &state) = default;
	Move get_last() const { return last; }
	Color get_winner() const { return winner; }
	Color current() const {
		if (last.z() == NO_MOVE_YET) 
			return Color::Black;
		return ~board.get(last);
	}
	const std::vector<Move> &get_options() const {
		assert(winner  == Color::Empty);
		return opts;
	}
	bool valid(Move mv) const {
		return std::find(opts.cbegin(), opts.cend(), mv) != opts.end();
	}
	bool over() const {
		return winner != Color::Empty || opts.size() == 0;
	}
	void next(Move mv) {
		assert(valid(mv));
		Color side = current();
		board.put(mv, side);
		if (board.win_from(mv)) winner = side;
		last = mv;
		opts.erase(std::find(opts.cbegin(), opts.cend(), mv));
	}
	Color next_rand_till_end() {
		while (!over()) 
			next(opts[0]);
		return winner;
	}
};

inline std::ostream &operator<<(std::ostream &out, const State &state) {
	if (state.last.z() == NO_MOVE_YET)
		return out << state.board << "last move: None";
	else
		return out << state.board << "last move: " << ~state.current() << state.last;
}	

class MCTSNode {
	friend std::ostream &operator<<(std::ostream &out, MCTSNode &node);
	MCTSNode *parent;
	std::map<Move, MCTSNode*> children;
	int visits;
	float quality;
	float prior;
public:
	MCTSNode(MCTSNode *node_p, float prior_p)
		: parent(node_p), visits(0), quality(0), prior(prior_p) {}
	~MCTSNode() {
		for (const auto &mn : children)
			delete mn.second;	
	}
	void expand(const std::vector<std::pair<Move, float>> &set) {
		for (auto &mvp : set)
			children[mvp.first] = new MCTSNode(this, mvp.second);
	}
	MCTSNode *cut(Move occurred) {
		auto citer = children.find(occurred);
		assert(citer != children.end());
		auto child = citer->second;
		children.erase(occurred);
		child->parent = nullptr;
		return child;
	}
	std::pair<Move, MCTSNode*> select(float c_puct) const {
		std::pair<Move, MCTSNode*> picked(Move(NO_MOVE_YET), nullptr);
		float max_value = -1 * std::numeric_limits<float>::max();
		for (const auto &mn : children) {
			float value = mn.second->value(c_puct);
			if (value > max_value) {
				picked = mn;
				max_value = value;
			}
		}
		return picked;
	}
	Move most_visted() const {
		int max_visit = -1 * std::numeric_limits<int>::max();
		Move act(NO_MOVE_YET);
		for (const auto &mn : children) {
			if (DEBUG_MCTS_NODE) 
				std::cout << mn.first << ": " <<  *mn.second << std::endl;
			auto vn = mn.second->visits;
			if (vn > max_visit) {
				act = mn.first;
				max_visit = vn;
			}
		}
		return act;
	}
	void update(float leafValue) {
		++visits;
		float delta = (leafValue - quality) / float(visits);
		quality += delta;
	}
	void update_recursive(float leafValue) {
		if (parent != nullptr)
			parent->update_recursive(-1 * leafValue);
		update(leafValue);
	}
	float value(float c_puct) const {
		assert(!is_root());
		float N = float(parent->visits);
		float n = float(1 + visits);
		return quality + (c_puct * prior * std::sqrt(N) / n);
	}
	bool is_leaf() const { return children.size() == 0; }
	bool is_root() const { return parent == nullptr; }
};

inline std::ostream &operator<<(std::ostream &out, MCTSNode &node) {
	return out << "MCTSNode(" << node.parent << "): "
		<< node.children.size() << " children, "<< node.visits << " visits, "
		<< node.prior << " prior, " << node.quality << " quality";
}
