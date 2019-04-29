#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <map>

#include "game.h"

Color operator~(const Color c) {
    Color opposite;
    switch (c) {
    case Color::Black: opposite = Color::White; break;
    case Color::White: opposite = Color::Black; break;
    case Color::Empty: opposite = Color::Empty; break;
    }
    assert(opposite != Color::Empty);
    return opposite;
}

std::ostream &operator<<(std::ostream &out, Color c) {
    switch (c) {
    case Color::Empty: out << ((COLOR_OCCUPY_SPACE == 2) ? "  " : " "); break;
    case Color::Black: out << ((COLOR_OCCUPY_SPACE == 2) ? "●"  : "x"); break;
    case Color::White: out << ((COLOR_OCCUPY_SPACE == 2) ? "○"  : "o"); break;
    }
    return out;
}

std::ostream &operator<<(std::ostream &out, Move mv) {
    return out << "(" << std::setw(2) << mv.r() << ", "
        << std::setw(2) << mv.c() << ")";
}

void Board::push_valid(std::vector<Move> &set) const {
    for (int i = 0; i < BOARD_SIZE; ++i)
        if (get(Move(i)) == Color::Empty)
            set.push_back(Move(i));

    std::shuffle(set.begin(), set.end(), global_random_engine);
}

bool Board::win_from(Move mv) const {
    if (mv.z() == NO_MOVE_YET) return false;
    Color side = get(mv);
    assert(side != Color::Empty);
    int direct[4][2] = { { 0, 1 },{ 1, 0 },{ -1, 1 },{ 1, 1 } };
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

std::ostream &operator<<(std::ostream &out, const Board &board) {
    if (COLOR_OCCUPY_SPACE == 2)
        out << " ";
    out << "# ";
    for (int c = 0; c < BOARD_MAX_COL; ++c)
        out << std::right << std::setw(COLOR_OCCUPY_SPACE) << c % 10 << " ";
    out << "\n";
    for (int r = 0; r < BOARD_MAX_ROW; ++r) {
        out << std::right << std::setw(COLOR_OCCUPY_SPACE) << r % 10;
        for (int c = 0; c < BOARD_MAX_COL; ++c)
            out << "|" << board.get(Move(r, c));
        out << "|\n";
    }
    return out;
}

Color State::current() const {
    if (last.z() == NO_MOVE_YET)
        return Color::Black;
    return ~board.get(last);
}

void State::fill_feature_array(float data[INPUT_FEATURE_NUM * BOARD_SIZE]) const {
    if (last.z() == NO_MOVE_YET) {
        if (INPUT_FEATURE_NUM > 3) {
            for (int r = 0; r < BOARD_MAX_ROW; ++r)
                for (int c = 0; c < BOARD_MAX_COL; ++c)
                    data[3 * BOARD_SIZE + r * BOARD_MAX_COL + c] = 1.0f;
        }
        return;
    }
    auto own_side = current();
    auto enemy_side = ~own_side;
    float first = first_hand() ? 1.0f : 0.0f;
    for (int r = 0; r < BOARD_MAX_ROW; ++r) {
        for (int c = 0; c < BOARD_MAX_COL; ++c) {
            auto side = board.get(Move(r, c));
            if (side == own_side)
                data[r * BOARD_MAX_COL + c] = 1.0f;
            else if (side == enemy_side)
                data[BOARD_SIZE + r * BOARD_MAX_COL + c] = 1.0f;
            if (INPUT_FEATURE_NUM > 3)
                data[3 * BOARD_SIZE + r * BOARD_MAX_COL + c] = first;
        }
    }
    if (INPUT_FEATURE_NUM > 2)
        data[2 * BOARD_SIZE  + last.r() * BOARD_MAX_COL + last.c()] = 1.0f;
}

void State::next(Move mv) {
    assert(valid(mv));
    Color side = current();
    board.put(mv, side);
    if (board.win_from(mv)) winner = side;
    last = mv;
    opts.erase(std::find(opts.cbegin(), opts.cend(), mv));
}

Color State::next_rand_till_end() {
    while (!over())
        next(opts[0]);
    return winner;
}

std::ostream &operator<<(std::ostream &out, const State &state) {
    if (state.last.z() == NO_MOVE_YET)
        return out << state.board << "last move: None";
    else
        return out << state.board << "last move: " << ~state.current() << state.last;
}

Player &play(Player &p1, Player &p2, bool silent) {
    const std::map<Color, Player*> player_color{
        { Color::Black, &p1 },
        { Color::White, &p2 },
        { Color::Empty, nullptr }
    };
    auto game = State();
    p1.reset();
    p2.reset();
    int turn = 0;
    if (!silent) std::cout << game << std::endl;
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

float benchmark(Player &p1, Player &p2, int round, bool silent) {
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
                << "\rscore: total=" << std::setw(4) << i + 1 << ", "
                << p1.name() << "=" << std::setw(4) << p1win << ", "
                << p2.name() << "=" << std::setw(4) << p2win;
            std::cout.flush();
        }
    }
    if (!silent) { std::cout << std::endl; }
    float p1prob = float(p1win) / float(round);
    float p2prob = float(p2win) / float(round);
    float eprob = float(even) / float(round);
    if (!silent) {
        std::cout << "benchmark player win probality: " << p1.name() << "=" << p1prob << ", "
            << p2.name() << "=" << p2prob << ", even=" << eprob << ", sim=" << round << std::endl;
    }
    return p1prob;
}

bool HumanPlayer::get_move(int &row, int &col) {
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

Move HumanPlayer::play(const State &state) {
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
