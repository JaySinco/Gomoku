#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <ctime>
#include <limits>
#include <iomanip>
#include <iostream>
#include <random>
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
Row
and move z(5) = (x(1), y(2))
*/

std::random_device RAND_DEVICE;
std::mt19937 RAND_ENGINE(RAND_DEVICE());
std::uniform_real_distribution<float> RAND_FLOAT(0.0f, 1.0f);

constexpr int FIVE_IN_ROW = 4;
constexpr int BOARD_MAX_ROW = 6;
constexpr int BOARD_MAX_COL = 6;
constexpr int BOARD_SIZE = BOARD_MAX_ROW * BOARD_MAX_COL;

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

class Move {
	int index;
public:
	Move() : index(-1) {}  // not move yet!
	Move(int z) : index(z) {}
	Move(int row, int col) {
		assert(row >= 0 && row < BOARD_MAX_ROW && col >= 0 && col < BOARD_MAX_COL);
		index = row * BOARD_MAX_COL + col;
	}
	Move(const Move &mv) : index(mv.z()) {}
	int z() const {
		return index;
	}
	int r() const {
		return index / BOARD_MAX_COL;
	}
	int c() const {
		return index % BOARD_MAX_COL;
	}
	bool operator<(const Move &right) const {
		return index < right.index;
	}
	bool operator==(const Move &right) const {
		return index == right.index;
	}
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
		assert(valid(mv));
		grid[mv.z()] = c;
	}
	bool within(Move mv) const {
		return mv.z() >= 0 && mv.z() < BOARD_SIZE;
	}
	bool valid(Move mv) const {
		return within(mv) && get(mv) == Color::Empty;
	}
	void options(std::vector<Move> &opts) const {
		for (int i = 0; i < BOARD_SIZE; ++i)
			if (get(Move(i)) == Color::Empty)
				opts.push_back(Move(i));
		std::shuffle(opts.begin(), opts.end(), RAND_ENGINE);
	}
	bool win(Move mv) const {
		if (mv == Move()) return false;
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
					if (!(r >= 0 && r < BOARD_MAX_ROW && c >= 0 && c < BOARD_MAX_COL))
						break;
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
	std::vector<Move> opts;
public:
	State() {
		board.options(opts);
	}
	State(const State &state) = default;
	~State() = default;
	Color current() const {
		if (last == Move())
			return Color::Black;
		return ~board.get(last);
	}
	Move lastmv() const { return last; }
	const std::vector<Move> &getOpts() const {
		assert(winner() == Color::Empty);
		return opts;
	}
	Color winner() const {
		if (board.win(last)) return ~current();
		return Color::Empty;
	}
	bool valid(Move mv) const {
		return std::find(opts.cbegin(), opts.cend(), mv) != opts.end();
	}
	bool over() const {
		return opts.size() == 0 || winner() != Color::Empty;
	}
	void next(Move mv) {
		board.put(mv, current());
		last = mv;
		auto choosed = std::find(opts.cbegin(), opts.cend(), mv);
		assert(choosed !=  opts.end());
		opts.erase(choosed);
	}
	Color nextRandTillEnd() {
		while (!over()) {
			auto randAct = opts[0];
			next(randAct);
		}	
		return winner();
	}
};

inline std::ostream &operator<<(std::ostream &out, const State &state) {
	if (state.last == Move())
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
	void expand(const std::vector<std::pair<Move, float>> &mvPriors) {
		for (auto &mvp : mvPriors) 
			children[mvp.first] = new MCTSNode(this, mvp.second);
	}
	MCTSNode *cut(Move occurred) {
		auto childIter = children.find(occurred);
		assert(childIter != children.end());
		auto child = childIter->second;
		children.erase(occurred);
		child->parent = nullptr;
		return child;
	}
	std::pair<Move, MCTSNode*> select(float c_puct) const {
		std::pair<Move, MCTSNode*> picked(Move(), nullptr);
		float maxValue = -1 * std::numeric_limits<float>::max();
		for (const auto &mn : children) {
			auto value = mn.second->value(c_puct);
			if (value > maxValue) {
				picked = mn;
				maxValue = value;
			}
		}
		return picked;
	}
	Move mostVisted() const {
		int maxVisit = -1 * std::numeric_limits<int>::max();
		Move act;
		for (const auto &mn : children) {
			// std::cout << mn.first << *mn.second << std::endl;
			auto vt = mn.second->visits;
			if ((vt > maxVisit) || (vt == maxVisit && RAND_FLOAT(RAND_ENGINE) > 0.5f)) {
				act = mn.first;
				maxVisit = vt;
			}
		}
		return act;
	}
	void update(float leafValue) {
		++visits;
		float delta = (leafValue - quality) / float(visits);
		quality += delta;
	}
	void updateRec(float leafValue) {
		if (parent != nullptr)
			parent->updateRec(-1 * leafValue);
		update(leafValue);
	}
	float value(float c_puct) const {
		assert(!isRoot());
		float N = float(parent->visits);
		float n = float(1 + visits);
		return quality + (c_puct * prior * std::sqrt(N) / n);
	}
	bool isLeaf() const { return children.size() == 0; }
	bool isRoot() const { return parent == nullptr; }
};

inline std::ostream &operator<<(std::ostream &out, MCTSNode &node) {
	return out << "MCTSNode(" << node.parent << "): "
		<< node.children.size() << " children, "<< node.visits << " visits, "
		<< node.prior << " prior, " << node.quality << " quality";
}
