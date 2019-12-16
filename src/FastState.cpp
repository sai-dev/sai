/*
    This file is part of SAI, which is a fork of Leela Zero.
    Copyright (C) 2017-2019 Gian-Carlo Pascutto and contributors
    Copyright (C) 2018-2019 SAI Team

    SAI is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    SAI is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with SAI.  If not, see <http://www.gnu.org/licenses/>.

    Additional permission under GNU GPL version 3 section 7

    If you modify this Program, or any covered work, by linking or
    combining it with NVIDIA Corporation's libraries from the
    NVIDIA CUDA Toolkit and/or the NVIDIA CUDA Deep Neural
    Network library and/or the NVIDIA TensorRT inference library
    (or a modified version of those libraries), containing parts covered
    by the terms of the respective license agreement, the licensors of
    this Program grant you additional permission to convey the resulting
    work.
*/

#include "config.h"
#include "FastState.h"

#include <algorithm>
#include <iterator>
#include <random>
#include <vector>

#include "FastBoard.h"
#include "GTP.h"
#include "Network.h"
#include "Random.h"
#include "Utils.h"
#include "Zobrist.h"
#include "GTP.h"

using namespace Utils;

void FastState::init_game(int size, float komi) {
    board.reset_board(size);

    m_movenum = 0;

    m_komove = FastBoard::NO_VERTEX;
    m_lastmove = FastBoard::NO_VERTEX;
    m_komi = komi;
    m_handicap = 0;
    m_passes = 0;

    m_randcount = 0;
    m_non_blunders = {};
    if (cfg_blunder_thr < 1.0f) {
        init_allowed_blunders();
    }
}

void FastState::set_non_blunders(const std::vector<int> & non_blunders) {
    m_non_blunders = non_blunders;
}

void FastState::set_komi(float komi) {
    m_komi = komi;
}

void FastState::add_komi(float delta) {
    m_komi += delta;
}

void FastState::reset_game() {
    reset_board();

    m_movenum = 0;
    m_passes = 0;
    m_handicap = 0;
    m_komove = FastBoard::NO_VERTEX;
    m_lastmove = FastBoard::NO_VERTEX;

    m_randcount = 0;
    m_non_blunders = {};
    if (cfg_blunder_thr < 1.0f) {
        init_allowed_blunders();
    }
}

void FastState::reset_board() {
    board.reset_board(board.get_boardsize());
}

bool FastState::is_move_legal(int color, int vertex) const {
    return !cfg_analyze_tags.is_to_avoid(color, vertex, m_movenum) && (
              vertex == FastBoard::PASS ||
                 vertex == FastBoard::RESIGN ||
                 (vertex != m_komove &&
                      board.get_state(vertex) == FastBoard::EMPTY &&
                      !board.is_suicide(vertex, color)));
}

void FastState::play_move(int vertex) {
    play_move(board.m_tomove, vertex);
}

void FastState::play_move(int color, int vertex) {
    board.m_hash ^= Zobrist::zobrist_ko[m_komove];
    if (vertex == FastBoard::PASS) {
        // No Ko move
        m_komove = FastBoard::NO_VERTEX;
    } else {
        m_komove = board.update_board(color, vertex);
    }
    board.m_hash ^= Zobrist::zobrist_ko[m_komove];

    m_lastmove = vertex;
    m_movenum++;

    if (!m_non_blunders.empty()) {
        m_blunder_chosen = end(m_non_blunders) ==
            std::find( begin(m_non_blunders), end(m_non_blunders), vertex );
        m_random_chosen = m_non_blunders[0] != vertex;
    }

    if (m_blunder_chosen && m_allowed_blunders > 0) {
        m_allowed_blunders--;
    }

    if (board.m_tomove == color) {
        board.m_hash ^= Zobrist::zobrist_blacktomove;
    }
    board.m_tomove = !color;

    board.m_hash ^= Zobrist::zobrist_pass[get_passes()];
    if (vertex == FastBoard::PASS) {
        increment_passes();
    } else {
        set_passes(0);
    }
    board.m_hash ^= Zobrist::zobrist_pass[get_passes()];
}

size_t FastState::get_movenum() const {
    return m_movenum;
}

int FastState::get_last_move() const {
    return m_lastmove;
}

int FastState::get_passes() const {
    return m_passes;
}

void FastState::set_passes(int val) {
    m_passes = val;
}

void FastState::increment_passes() {
    m_passes++;
    if (m_passes > 4) m_passes = 4;
}

int FastState::get_to_move() const {
    return board.m_tomove;
}

void FastState::set_to_move(int tom) {
    board.set_to_move(tom);
}

void FastState::display_state() {
    myprintf("\nPasses: %d            Black (X) Prisoners: %d\n",
             m_passes, board.get_prisoners(FastBoard::BLACK));
    if (board.black_to_move()) {
        myprintf("Black (X) to move");
    } else {
        myprintf("White (O) to move");
    }
    myprintf("    White (O) Prisoners: %d\n",
             board.get_prisoners(FastBoard::WHITE));
    myprintf("Move %3d             Komi: %.1f\n", get_movenum(), get_komi());

    board.display_board(get_last_move());
}

void FastState::display_legal(int color) {
    myprintf("\nPasses: %d            Black (X) Prisoners: %d\n",
             m_passes, board.get_prisoners(FastBoard::BLACK));
    if (board.black_to_move()) {
        myprintf("Black (X) to move");
    } else {
        myprintf("White (O) to move");
    }
    myprintf("    White (O) Prisoners: %d\n",
             board.get_prisoners(FastBoard::WHITE));

    int boardsize = board.get_boardsize();

    myprintf("\n   ");
    board.print_columns();
    for (int j = boardsize-1; j >= 0; j--) {
        myprintf("%2d", j+1);
        myprintf(" ");
        for (int i = 0; i < boardsize; i++) {
            if (is_move_legal(color, board.get_vertex(i,j))) {
                //                myprintf("O");
                myprintf("%1d", board.liberties_to_capture(board.get_vertex(i,j)));
            } else {
                myprintf(".");
            }
            myprintf(" ");
        }
        myprintf("%2d\n", j+1);
    }
    myprintf("   ");
    board.print_columns();
    myprintf("\n");
    //board.display_legal(color);
}

std::string FastState::move_to_text(int move) {
    return board.move_to_text(move);
}

float FastState::final_score() const {
    //    return board.area_score(get_komi() + get_handicap());
    // there appears to be no good reason to correct komi with
    // handicap stones number, for example sabaki doesn't
    return board.area_score(get_komi());
}

float FastState::get_komi() const {
    return m_komi;
}

void FastState::set_handicap(int hcap) {
    m_handicap = hcap;
}

int FastState::get_handicap() const {
    return m_handicap;
}

std::uint64_t FastState::get_symmetry_hash(int symmetry) const {
    return board.calc_symmetry_hash(m_komove, symmetry);
}

// void FastState::set_last_rnd_move_num(size_t num) {
//     m_lastrndmovenum = num;
// }

// size_t FastState::get_last_rnd_move_num() {
//     return m_lastrndmovenum;
// }

// void FastState::set_blunder_state(bool state) {
//     m_blunder_chosen = state;
// }

void FastState::init_allowed_blunders() {
    auto distribution = std::poisson_distribution<>{cfg_blunder_rndmax_avg};
    m_allowed_blunders = distribution(Random::get_Rng());
#ifndef NDEBUG
    myprintf("Initially allowed blunders: %d\n", m_allowed_blunders);
#endif
}

bool FastState::is_blunder_allowed() const {
    return (m_allowed_blunders != 0); // both positive values and -1 are ok
}

int FastState::get_allowed_blunders() const {
    return (m_allowed_blunders);
}

bool FastState::is_symmetry_invariant(const int symmetry) const {
    for (auto y = 0; y < BOARD_SIZE; y++) {
        for (auto x = 0; x < BOARD_SIZE; x++) {
            const auto sym_vertex =
                board.get_vertex(symmetry_nn_idx_table[symmetry][y * BOARD_SIZE + x]);
            if (board.get_state(x, y) != board.get_state(sym_vertex))
                return false;
        }
    }

    if(m_komove != 0) {
        if (m_komove != board.get_sym_move(m_komove, symmetry))
            return false;
    }

    return true;
}
