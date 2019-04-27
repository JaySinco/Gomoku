#pragma once

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

#include "vars.h"

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

enum class Color {Empty, Black, White};
Color operator~(const Color c);
std::ostream &operator<<(std::ostream &out, Color c);

class Move {
    int index;
public:
    Move(int z) : index(z) { assert((z >= 0 && z < BOARD_SIZE) || z == NO_MOVE_YET); }
    Move(int row, int col) { assert(ON_BOARD(row, col)); index = row * BOARD_MAX_COL + col; }
    Move(const Move &mv) : index(mv.z()) {}
    int z() const { return index; }
    int r() const { assert(index >= 0 && index < BOARD_SIZE); return index / BOARD_MAX_COL; }
    int c() const { assert(index >= 0 && index < BOARD_SIZE); return index % BOARD_MAX_COL; }
    bool operator<(const Move &right) const { return index < right.index; }
    bool operator==(const Move &right) const { return index == right.index; }
};
std::ostream &operator<<(std::ostream &out, Move mv);

class Board {
    Color grid[BOARD_SIZE];
public:
    Board() : grid{ Color::Empty } {}
    Color get(Move mv) const { return grid[mv.z()]; }
    void put(Move mv, Color c) { assert(get(mv) == Color::Empty); grid[mv.z()] = c; }
    void push_valid(std::vector<Move> &set) const;
    bool win_from(Move mv) const;

};
std::ostream &operator<<(std::ostream &out, const Board &board);

class State {
    friend std::ostream &operator<<(std::ostream &out, const State &state);
    Board board;
    Move last;
    Color winner;
    std::vector<Move> opts;
public:
    State() : last(NO_MOVE_YET), winner(Color::Empty) { board.push_valid(opts); }
    State(const State &state) = default;
    Move get_last() const { return last; }
    Color get_winner() const { return winner; }
    Color current() const;
    bool first_hand() const { return current() == Color::Black; }
    void fill_feature_array(float data[INPUT_FEATURE_NUM * BOARD_SIZE]) const;
    const std::vector<Move> &get_options() const { assert(!over()); return opts; };
    bool valid(Move mv) const { return std::find(opts.cbegin(), opts.cend(), mv) != opts.end(); }
    bool over() const { return winner != Color::Empty || opts.size() == 0; }
    void next(Move mv);
    Color next_rand_till_end();
};
std::ostream &operator<<(std::ostream &out, const State &state);

struct Player {
    Player() {}
    virtual void reset() = 0;
    virtual const std::string &name() const = 0;
    virtual Move play(const State &state) = 0;
    virtual ~Player() {};
};

Player &play(Player &p1, Player &p2, bool silent = true);
float benchmark(Player &p1, Player &p2, int round, bool silent = true);

class RandomPlayer : public Player {
    std::string id;
public:
    RandomPlayer(const std::string &name) : id(name) {}
    void reset() override {}
    const std::string &name() const override { return id; }
    Move play(const State &state) override { return state.get_options()[0]; }
    ~RandomPlayer() {};
};

class HumanPlayer : public Player {
    std::string id;
    bool get_move(int &row, int &col);
public:
    HumanPlayer(const std::string &name) : id(name) {}
    void reset() override {}
    const std::string &name() const override { return id; }
    Move play(const State &state) override;
    ~HumanPlayer() {};
};