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

#ifndef GAMESTATE_H_INCLUDED
#define GAMESTATE_H_INCLUDED

#include <memory>
#include <string>
#include <vector>
#include <tuple>

#include "FastState.h"
#include "FullBoard.h"
#include "KoState.h"
#include "TimeControl.h"

class Network;

class GameState : public KoState {
public:
    explicit GameState() = default;
    explicit GameState(const KoState* rhs) {
        // Copy in fields from base class.
        *(static_cast<KoState*>(this)) = *rhs;
        anchor_game_history();
    }
    void init_game(int size, float komi, bool value_head_sai = true);
    void reset_game();
    bool set_fixed_handicap(int stones);
    int set_fixed_handicap_2(int stones);
    void place_free_handicap(int stones, Network & network);
    void anchor_game_history();

    void rewind(); /* undo infinite */
    bool undo_move();
    bool forward_move();
    std::shared_ptr<const KoState> get_past_state(int moves_ago) const;
    const FullBoard& get_past_board(int moves_ago) const;
    const std::vector<std::shared_ptr<const KoState>>& get_game_history() const;

    void play_move(int vertex);
    void play_move(int color, int vertex);
    bool play_textmove(std::string color, const std::string& vertex);

    void start_clock(int color);
    void stop_clock(int color);
    const TimeControl& get_timecontrol() const;
    void set_timecontrol(const TimeControl& timecontrol);
    void set_timecontrol(int maintime, int byotime, int byostones,
                         int byoperiods);
    void adjust_time(int color, int time, int stones);

    void display_state();
    bool has_resigned() const;
    int who_resigned() const;
    StateEval get_state_eval() const;
    void set_state_eval(const StateEval& ev);
    //    void copy_last_rnd_move_num ();
    std::string eval_comment(bool print_header = false) const;
    bool score_agreed() const;
    std::pair<int, int> get_accepted_scores() const { return m_acceptedscore; }
    float get_final_accepted_score() const;
    void update_accepted_score(std::tuple<float, float, float> node_stats, bool switch_player = false);
    bool is_cpu_color() const;
    void set_cpu_color(int which_color);
    std::pair<int, float> get_last_think() const;
    void set_last_think(float alpkt) {
        m_last_think_alpkt = alpkt;
        m_last_think_movenum = get_movenum();
    }
    float get_opp_avgloss() const;
    void add_opp_ptsloss(float alpkt_new);

private:
    bool valid_handicap(int stones);

    std::vector<std::shared_ptr<const KoState>> game_history;
    TimeControl m_timecontrol;
    int m_resigned{FastBoard::EMPTY};
    std::pair<int, int> m_acceptedscore = {-1 * NUM_INTERSECTIONS, NUM_INTERSECTIONS};
    int m_cpu_color{FastBoard::EMPTY};
    int m_last_think_movenum{-1};
    float m_last_think_alpkt{0.0f};
    int m_opp_movenum{0};
    float m_opp_lostpts{0.0f};
};

#endif
