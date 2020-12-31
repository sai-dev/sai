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

#ifndef KOSTATE_H_INCLUDED
#define KOSTATE_H_INCLUDED

#include "config.h"

#include <vector>
#include <tuple>

#include "FastState.h"
#include "FullBoard.h"

struct StateEval {
    size_t visits = 0;
    float alpkt = 0.0f;
    float beta = 1.0f;
    float pi = 0.5f;
    float agent_eval = 0.5f;
    float agent_x_lambda = 0.0f;
    float agent_x_mu = 0.0f;
    float agent_eval_avg = 0.5f;
    float alpkt_tree = 0.0f;

    StateEval(int visits, float alpkt, float beta, float pi,
              float agent_eval, float agent_x_lambda, float agent_x_mu,
              float agent_eval_avg,
              float alpkt_tree)
    : visits(visits), alpkt(alpkt), beta(beta), pi(pi),
      agent_eval(agent_eval), agent_x_lambda(agent_x_lambda),
      agent_x_mu(agent_x_mu), agent_eval_avg(agent_eval_avg),
      alpkt_tree(alpkt_tree)
    {}

    StateEval() {}
};

class KoState : public FastState {
public:
    void init_game(int size, float komi);
    bool superko() const;
    void reset_game();

    StateEval get_state_eval() const;
    void set_state_eval(const StateEval& ev);

    void play_move(int vertex);
    void play_move(int color, int vertex);

private:
    std::vector<std::uint64_t> m_ko_hash_history;
    StateEval m_ev;
};

#endif
