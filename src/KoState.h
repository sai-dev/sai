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

#ifndef KOSTATE_H_INCLUDED
#define KOSTATE_H_INCLUDED

#include "config.h"

#include <vector>
#include <tuple>

#include "FastState.h"
#include "FullBoard.h"

class KoState : public FastState {
public:
    void init_game(int size, float komi);
    bool superko(void) const;
    void reset_game();

    void play_move(int color, int vertex);
    void play_move(int vertex);
    std::tuple<float,float,float,float,float> get_eval();
    void set_eval(float alpkt, float beta, float pi,
		  float avg_eval, float eval_bonus, float eval_base);
private:
    std::vector<std::uint64_t> m_ko_hash_history;
    float m_alpkt = 0.0f;
    float m_beta = 1.0f;
    float m_pi = 0.5f;
    float m_avg_eval = 0.5f;
    float m_eval_bonus = 0.0f;
    float m_eval_base = 0.0f;
};

#endif
