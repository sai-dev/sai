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

#include "GameState.h"
#include "Network.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iterator>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>

#include "FastBoard.h"
#include "FastState.h"
#include "FullBoard.h"
#include "GTP.h"
#include "KoState.h"
#include "UCTSearch.h"
#include "Utils.h"

StateEval GameState::get_state_eval() const {
    return KoState::get_state_eval();
}

void GameState::set_state_eval(const StateEval& ev) {
    KoState::set_state_eval(ev);
}


void GameState::init_game(int size, float komi, bool value_head_sai) {
    if (!value_head_sai && komi != 7.5) {
        Utils::myprintf("Warning: komi set to 7.5 because network "
                        "does not allow variable komi.\n");
        komi = 7.5;
    }
    KoState::init_game(size, komi);

    game_history.clear();
    game_history.emplace_back(std::make_shared<KoState>(*this));

    m_timecontrol.reset_clocks();

    m_resigned = FastBoard::EMPTY;
    m_acceptedscore = {-1 * NUM_INTERSECTIONS, NUM_INTERSECTIONS };
    m_cpu_color = FastBoard::EMPTY;
    m_last_think_movenum = -1;
    m_last_think_alpkt = 0.0f;
    m_opp_movenum = 0;
    m_opp_lostpts = 0.0f;
    Utils::dump_agent_params();
}

void GameState::reset_game() {
    KoState::reset_game();

    game_history.clear();
    game_history.emplace_back(std::make_shared<KoState>(*this));

    m_timecontrol.reset_clocks();

    m_resigned = FastBoard::EMPTY;
    m_acceptedscore = {-1 * NUM_INTERSECTIONS, NUM_INTERSECTIONS };
    m_cpu_color = FastBoard::EMPTY;
    m_last_think_movenum = -1;
    m_last_think_alpkt = 0.0f;
    m_opp_movenum = 0;
    m_opp_lostpts = 0.0f;
}

bool GameState::forward_move() {
    auto movenum = get_movenum();
    if (game_history.size() > movenum + 1) {
        ++movenum;
        set_movenum(movenum);

        // this is not so nice, but it should work
        *(static_cast<KoState*>(this)) = *game_history[get_movenum()];

        // This also restores hashes as they're part of state
        return true;
    } else {
        return false;
    }
}

bool GameState::undo_move() {
    auto movenum = get_movenum();
    if (movenum > 0) {
        --movenum;
        set_movenum(movenum);

        // this is not so nice, but it should work
        *(static_cast<KoState*>(this)) = *game_history[movenum];

        // This also restores hashes as they're part of state
        return true;
    } else {
        return false;
    }
}

void GameState::rewind() {
    *(static_cast<KoState*>(this)) = *game_history[0];
    set_movenum(0);
}

void GameState::play_move(int vertex) {
    play_move(get_to_move(), vertex);
}

void GameState::play_move(int color, int vertex) {
    if (vertex == FastBoard::RESIGN) {
        m_resigned = color;
    } else {
      KoState::play_move(color, vertex);
    }

    // cut off any leftover moves from navigating
    game_history.resize(get_movenum());
    game_history.emplace_back(std::make_shared<KoState>(*this));

    // this is the place to reset state info for comments
    reset_comment_data();
}

bool GameState::play_textmove(std::string color, const std::string& vertex) {
    int who;
    transform(cbegin(color), cend(color), begin(color), tolower);
    if (color == "w" || color == "white") {
        who = FullBoard::WHITE;
    } else if (color == "b" || color == "black") {
        who = FullBoard::BLACK;
    } else {
        return false;
    }

    const auto move = board.text_to_move(vertex);
    if (move == FastBoard::NO_VERTEX ||
        (move != FastBoard::PASS && move != FastBoard::RESIGN && board.get_state(move) != FastBoard::EMPTY)) {
        return false;
    }

    set_to_move(who);
    play_move(move);

    return true;
}

void GameState::stop_clock(int color) {
    m_timecontrol.stop(color);
}

void GameState::start_clock(int color) {
    m_timecontrol.start(color);
}

void GameState::display_state() {
    FastState::display_state();

    m_timecontrol.display_times();
}

int GameState::who_resigned() const {
    return m_resigned;
}

bool GameState::has_resigned() const {
    return m_resigned != FastBoard::EMPTY;
}

const TimeControl& GameState::get_timecontrol() const {
    return m_timecontrol;
}

void GameState::set_timecontrol(const TimeControl& timecontrol) {
    m_timecontrol = timecontrol;
}

void GameState::set_timecontrol(int maintime, int byotime,
                                int byostones, int byoperiods) {
    TimeControl timecontrol(maintime, byotime,
                            byostones, byoperiods);

    m_timecontrol = timecontrol;
}

void GameState::adjust_time(int color, int time, int stones) {
    m_timecontrol.adjust_time(color, time, stones);
}

void GameState::anchor_game_history() {
    // handicap moves don't count in game history
    set_movenum(0);
    game_history.clear();
    game_history.emplace_back(std::make_shared<KoState>(*this));
}

bool GameState::set_fixed_handicap(int handicap) {
    if (!valid_handicap(handicap)) {
        return false;
    }

    int board_size = board.get_boardsize();

    int high = board_size >= 13 ? 3 : 2;
    int mid = board_size / 2;
    int low = board_size - 1 - high;

    if (handicap >= 2) {
        play_move(FastBoard::BLACK, board.get_vertex(low, low));
        play_move(FastBoard::BLACK, board.get_vertex(high, high));
    }

    if (handicap >= 3) {
        play_move(FastBoard::BLACK, board.get_vertex(high, low));
    }

    if (handicap >= 4) {
        play_move(FastBoard::BLACK, board.get_vertex(low, high));
    }

    if (handicap >= 5 && handicap % 2 == 1) {
        play_move(FastBoard::BLACK, board.get_vertex(mid, mid));
    }

    if (handicap >= 6) {
        play_move(FastBoard::BLACK, board.get_vertex(low, mid));
        play_move(FastBoard::BLACK, board.get_vertex(high, mid));
    }

    if (handicap >= 8) {
        play_move(FastBoard::BLACK, board.get_vertex(mid, low));
        play_move(FastBoard::BLACK, board.get_vertex(mid, high));
    }

    board.set_to_move(FastBoard::WHITE);

    anchor_game_history();

    set_handicap(handicap);

    return true;
}

int GameState::set_fixed_handicap_2(int handicap) {
    int board_size = board.get_boardsize();
    int low = board_size >= 13 ? 3 : 2;
    int mid = board_size / 2;
    int high = board_size - 1 - low;

    int interval = (high - mid) / 2;
    int placed = 0;

    while (interval >= 3) {
        for (int i = low; i <= high; i += interval) {
            for (int j = low; j <= high; j += interval) {
                if (placed >= handicap) return placed;
                if (board.get_state(i-1, j-1) != FastBoard::EMPTY) continue;
                if (board.get_state(i-1, j) != FastBoard::EMPTY) continue;
                if (board.get_state(i-1, j+1) != FastBoard::EMPTY) continue;
                if (board.get_state(i, j-1) != FastBoard::EMPTY) continue;
                if (board.get_state(i, j) != FastBoard::EMPTY) continue;
                if (board.get_state(i, j+1) != FastBoard::EMPTY) continue;
                if (board.get_state(i+1, j-1) != FastBoard::EMPTY) continue;
                if (board.get_state(i+1, j) != FastBoard::EMPTY) continue;
                if (board.get_state(i+1, j+1) != FastBoard::EMPTY) continue;
                play_move(FastBoard::BLACK, board.get_vertex(i, j));
                placed++;
            }
        }
        interval = interval / 2;
    }

    return placed;
}

bool GameState::valid_handicap(int handicap) {
    int board_size = board.get_boardsize();

    if (handicap < 2 || handicap > 9) {
        return false;
    }
    if (board_size % 2 == 0 && handicap > 4) {
        return false;
    }
    if (board_size == 7 && handicap > 4) {
        return false;
    }
    if (board_size < 7 && handicap > 0) {
        return false;
    }

    return true;
}

void GameState::place_free_handicap(int stones, Network & network) {
    int limit = board.get_boardsize() * board.get_boardsize();
    if (stones > limit / 2) {
        stones = limit / 2;
    }

    int orgstones = stones;

    int fixplace = std::min(9, stones);

    set_fixed_handicap(fixplace);
    stones -= fixplace;

    stones -= set_fixed_handicap_2(stones);

    for (int i = 0; i < stones; i++) {
        auto search = std::make_unique<UCTSearch>(*this, network);
        auto move = search->think(FastBoard::BLACK, UCTSearch::NOPASS);
        play_move(FastBoard::BLACK, move);
    }

    if (orgstones)  {
        board.set_to_move(FastBoard::WHITE);
    } else {
        board.set_to_move(FastBoard::BLACK);
    }

    anchor_game_history();

    set_handicap(orgstones);
}

std::shared_ptr<const KoState> GameState::get_past_state(int moves_ago) const {
    assert(moves_ago >= 0 && (unsigned)moves_ago <= get_movenum());
    assert(get_movenum() + 1 <= game_history.size());
    return game_history[get_movenum() - moves_ago];
}

std::string GameState::eval_comment(bool print_header) const {
    auto comstr = std::stringstream{};

    if (print_header) {
        comstr << "alpkt_tree" << ", "
               << "alpkt" << ", "
               << "beta" << ", "
               << "pi" << ", "
               << "agent_eval_avg" << ", "
               << "quantile_lambda" << ", "
               << "quantile_mu" << ", "
               << "bitfield";
    } else {
        const auto ev = get_state_eval();
        comstr << std::setprecision(3)
               << ev.alpkt_tree << ", "
               << ev.alpkt << ", "
               << ev.beta << ", "
               << ev.pi << ", "
               << ev.agent_eval_avg << ", "
               << ev.agent_x_lambda << ", "
               << ev.agent_x_mu << ", "
               << flags_to_text();
    }

    return comstr.str();
}

const std::vector<std::shared_ptr<const KoState>>& GameState::get_game_history() const {
    return game_history;
}


bool GameState::score_agreed() const {
    // Caution: there is no guarantee that first <= second, even if
    // this is usually the case. The case first > second happens when
    // the current player just understood that something bad happened
    // for its side and its previous expectations were too
    // optimistic. In this case, we wait the updated evaluation of the
    // opponent before agreeing on the final score.
    return m_acceptedscore.first == m_acceptedscore.second;
}

void GameState::update_accepted_score(std::tuple<float, float, float> node_stats, bool switch_player) {
    float alpkt, beta, black_eval;
    std::tie(alpkt, beta, black_eval) = node_stats;

    const auto komi = get_komi_adj();
    const auto black_alpha = alpkt + komi;
    const auto lead_eval = std::max(black_eval, 1.0f - black_eval);
    constexpr auto highest_conf = 0.99f;
    constexpr auto normal_conf = 0.90f;
    const auto a_over_b = std::log(1.0f - normal_conf) / std::log(1.0f - highest_conf);
    const auto normal_log_odds = std::log( normal_conf / (1.0f - normal_conf) );
    const auto adj_log_odds = std::log( lead_eval / (1.0f - lead_eval) ) / normal_log_odds;
    const auto exponent = std::pow(a_over_b, adj_log_odds);
    const auto confidence = 1.0f - 0.5f * std::pow (1.0f - highest_conf, exponent);

    const auto color = get_to_move();
    const auto is_white = ( color == FastBoard::WHITE && !switch_player )
                          || ( color == FastBoard::BLACK && switch_player);

    const auto range = std::log(confidence / (1.0f - confidence)) / beta;
    // Utils::myprintf("Update accepted score: black_alpha %.2f, beta %.2f, "
    //                 "confidence %.1f\% range %.2f, black_eval %.3f\n",
    //                 black_alpha, beta, 100.0f*confidence, range, black_eval);
    if (is_white) {
        auto new_score = int(std::ceil(black_alpha - range));
        // if the new score would make me lose but eval is still
        // uncertain, update with minimum score for winning or tying
        // instead
        if (new_score - komi > 0 && black_eval < 0.5f) {
            new_score = int(std::floor(komi));
        }
        // if the new score would make me tie but eval says I should
        // win, do not accept less than a 1 point win
        if (new_score - komi == 0 && black_eval < 0.1f) {
            new_score--;
        }
        m_acceptedscore.first = new_score;
    } else {
        auto new_score = int(std::floor(black_alpha + range));
        // if the new score would make me lose but eval is still
        // uncertain, update with minimum score for winning or tying
        // instead
        if (new_score - komi < 0 && black_eval > 0.5f) {
            new_score = int(std::ceil(komi));
        }
        // if the new score would make me tie but eval says I should
        // win, do not accept less than a 1 point win
        if (new_score - komi == 0 && black_eval > 0.9f) {
            new_score++;
        }
        m_acceptedscore.second = new_score;
    }
}

float GameState::get_final_accepted_score() const {
    if (score_agreed()) {
        return m_acceptedscore.first - get_komi_adj();
    } else {
        // return an impossible value
        return std::numeric_limits<float>::infinity();
    }
}

void GameState::set_cpu_color(int which_color) {
    auto new_color = m_cpu_color;
    std::string role;
    switch (which_color) {
    case FastBoard::BOTH_COLORS:
        new_color = FastBoard::INVAL;
        break;
    case FastBoard::THIS_COLOR:
        new_color = get_to_move();
        break;
    case FastBoard::OTHER_COLOR:
        new_color = !get_to_move();
        break;
    }
    switch (new_color) {
    case FastBoard::INVAL:
        role = "both players";
        break;
    case FastBoard::BLACK:
        role = "Black";
        break;
    case FastBoard::WHITE:
        role = "White";
        break;
    }
    if (m_cpu_color == FastBoard::EMPTY) {
        m_cpu_color = new_color;
        Utils::myprintf("CPU role fixed as %s.\n", role.c_str());
    } else if (!Utils::agent_color_dependent()) {
    } else if (which_color == FastBoard::BOTH_COLORS &&
               m_cpu_color != new_color) {
        Utils::myprintf("Warning: CPU role was previously fixed, and cannot be set to both players now.\n"
                        "The commands 'clear_board' and 'loadsgf' will reset it to undefined.\n");
    } else if (which_color == FastBoard::THIS_COLOR &&
               m_cpu_color == !new_color) {
        Utils::myprintf("Warning: CPU role was previously fixed differently, and cannot be changed now.\n"
                        "The commands 'clear_board' and 'loadsgf' will reset it to undefined.\n");
    }
}

bool GameState::is_cpu_color() const {
    if (m_cpu_color == FastBoard::INVAL) {
        return true;
    } else {
        return m_cpu_color == get_to_move();
    }
}

std::pair<int, float> GameState::get_last_think() const {
    return std::make_pair(m_last_think_movenum, m_last_think_alpkt);
}

float GameState::get_opp_avgloss() const {
    return m_opp_movenum ? m_opp_lostpts / float(m_opp_movenum) : 0.0f;
}

void GameState::add_opp_ptsloss(float alpkt_new) {
    if (!is_cpu_color()) {
        return;
    }
    const auto last = get_last_think();
    set_last_think(alpkt_new);
    if (last.first >= 0 && int(get_movenum()) == 2 + last.first) {
        const auto delta_alpha = (alpkt_new - last.second) * (get_to_move() == FastBoard::WHITE ? -1 : 1);
        m_opp_lostpts += delta_alpha;
        ++m_opp_movenum;
        // Utils::myprintf("Opponent level updated. Movenum %d, alpkt %5.2f. This move gain %5.2f pts. Avg gain %5.2f pts. Pts num %d.\n",
        //                 last.first, last.second, delta_alpha, get_opp_avgloss(), m_opp_movenum);
    // } else {
    //     Utils::myprintf("Opponent level not updated. Movenum %d, alpkt %5.2f\n", last.first, last.second);
    }
    // const auto curr = get_last_think();
    // Utils::myprintf("Last think updated. Movenum %d, alpkt %5.2f\n", curr.first, curr.second);
}
