/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2018 Gian-Carlo Pascutto and contributors
    Copyright (C) 2018 SAI Team

    Leela Zero is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Leela Zero is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "config.h"
#include "KoState.h"

#include <cassert>
#include <algorithm>
#include <iterator>
#include <tuple>

#include "FastBoard.h"
#include "FastState.h"
#include "FullBoard.h"

void KoState::init_game(int size, float komi) {
    assert(size <= BOARD_SIZE);

    FastState::init_game(size, komi);

    m_ko_hash_history.clear();
    m_ko_hash_history.emplace_back(board.get_ko_hash());
}

bool KoState::superko(void) const {
    auto first = crbegin(m_ko_hash_history);
    auto last = crend(m_ko_hash_history);

    auto res = std::find(++first, last, board.get_ko_hash());

    return (res != last);
}

void KoState::reset_game() {
    FastState::reset_game();

    m_ko_hash_history.clear();
    m_ko_hash_history.push_back(board.get_ko_hash());
    set_eval(0.0f, 1.0f, 0.5f, 0.5f, 0.0f);
}

void KoState::play_move(int vertex) {
    play_move(board.get_to_move(), vertex);
}

void KoState::play_move(int color, int vertex) {
    if (vertex != FastBoard::RESIGN) {
        FastState::play_move(color, vertex);
    }
    m_ko_hash_history.push_back(board.get_ko_hash());
}

std::tuple<float,float,float,float,float> KoState::get_eval() {
    return std::make_tuple(m_alpkt,m_beta,m_pi,m_avg_eval,m_eval_bonus);
}

void KoState::set_eval(float alpkt,
		       float beta,
		       float pi,
		       float avg_eval,
		       float eval_bonus) {
    m_alpkt = alpkt;
    m_beta = beta;
    m_pi = pi;
    m_avg_eval = avg_eval;
    m_eval_bonus = eval_bonus;
}
