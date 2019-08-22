/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2019 Gian-Carlo Pascutto and contributors
    Copyright (C) 2018-2019 SAI Team

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
#include "UCTSearch.h"

#include <boost/format.hpp>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <tuple>
#include <algorithm>
#include <iostream>

#include "FastBoard.h"
#include "FastState.h"
#include "FullBoard.h"
#include "GTP.h"
#include "GameState.h"
#include "TimeControl.h"
#include "Timing.h"
#include "Training.h"
#include "Utils.h"
#ifdef USE_OPENCL
#include "OpenCLScheduler.h"
#endif

#include "Network.h"

using namespace Utils;

constexpr int UCTSearch::UNLIMITED_PLAYOUTS;

class OutputAnalysisData {
public:
    OutputAnalysisData(const std::string& move, int visits,
                       float winrate, float policy_prior, std::string pv,
                       float lcb, bool lcb_ratio_exceeded)
    : m_move(move), m_visits(visits), m_winrate(winrate),
      m_policy_prior(policy_prior), m_pv(pv), m_lcb(lcb),
      m_lcb_ratio_exceeded(lcb_ratio_exceeded) {};

    std::string get_info_string(int order) const {
        auto tmp = "info move " + m_move
                 + " visits " + std::to_string(m_visits)
                 + " winrate "
                 + std::to_string(static_cast<int>(m_winrate * 10000))
                 + " prior "
                 + std::to_string(static_cast<int>(m_policy_prior * 10000.0f))
                 + " lcb "
                 + std::to_string(static_cast<int>(std::max(0.0f, m_lcb) * 10000));
        if (order >= 0) {
            tmp += " order " + std::to_string(order);
        }
        tmp += " pv " + m_pv;
        return tmp;
    }

    friend bool operator<(const OutputAnalysisData& a,
                          const OutputAnalysisData& b) {
        if (a.m_lcb_ratio_exceeded && b.m_lcb_ratio_exceeded) {
            if (a.m_lcb != b.m_lcb) {
                return a.m_lcb < b.m_lcb;
            }
        }
        if (a.m_visits == b.m_visits) {
            return a.m_winrate < b.m_winrate;
        }
        return a.m_visits < b.m_visits;
    }

private:
    std::string m_move;
    int m_visits;
    float m_winrate;
    float m_policy_prior;
    std::string m_pv;
    float m_lcb;
    bool m_lcb_ratio_exceeded;
};


UCTSearch::UCTSearch(GameState& g, Network& network)
    : m_rootstate(g), m_network(network) {
    set_playout_limit(cfg_max_playouts);
    set_visit_limit(cfg_max_visits);

    m_root = std::make_unique<UCTNode>(FastBoard::PASS, 0.0f);
}

void UCTSearch::reset() {
    set_playout_limit(cfg_max_playouts);
    set_visit_limit(cfg_max_visits);

    m_root = std::make_unique<UCTNode>(FastBoard::PASS, 0.0f);
    m_last_rootstate.reset(nullptr);
    m_nodes = m_root->count_nodes_and_clear_expand_state();
}

bool UCTSearch::advance_to_new_rootstate() {
    if (!m_root || !m_last_rootstate) {
        // No current state
        return false;
    }

    if (m_rootstate.get_komi() != m_last_rootstate->get_komi()) {
        return false;
    }

    auto depth = static_cast<ptrdiff_t>(m_rootstate.get_movenum() -
                                        m_last_rootstate->get_movenum());

    if (depth < 0) {
        return false;
    }


    auto test = std::make_unique<GameState>(m_rootstate);
    for (auto i = 0; i < depth; i++) {
        test->undo_move();
    }

    if (m_last_rootstate->board.get_hash() != test->board.get_hash()) {
        // m_rootstate and m_last_rootstate don't match
        return false;
    }

    // Make sure that the nodes we destroyed the previous move are
    // in fact destroyed.
    while (!m_delete_futures.empty()) {
        m_delete_futures.front().wait_all();
        m_delete_futures.pop_front();
    }

    // Try to replay moves advancing m_root
    for (auto i = 0; i < depth; i++) {
        ThreadGroup tg(thread_pool);

        test->forward_move();
        const auto move = test->get_last_move();

        auto oldroot = std::move(m_root);
        m_root = oldroot->find_child(move);

        // Lazy tree destruction.  Instead of calling the destructor of the
        // old root node on the main thread, send the old root to a separate
        // thread and destroy it from the child thread.  This will save a
        // bit of time when dealing with large trees.
        auto p = oldroot.release();
        tg.add_task([p]() { delete p; });
        m_delete_futures.push_back(std::move(tg));

        if (!m_root) {
            // Tree hasn't been expanded this far
            return false;
        }
        m_last_rootstate->play_move(move);
    }

    assert(m_rootstate.get_movenum() == m_last_rootstate->get_movenum());

    if (m_last_rootstate->board.get_hash() != test->board.get_hash()) {
        // Can happen if user plays multiple moves in a row by same player
        return false;
    }

    return true;
}

void UCTSearch::update_root(bool is_evaluating) {
    // Definition of m_playouts is playouts per search call.
    // So reset this count now.
    m_playouts = 0;

#ifndef NDEBUG
    auto start_nodes = m_root->count_nodes_and_clear_expand_state();
#endif

    if ( (!advance_to_new_rootstate() && !is_evaluating) || !m_root) {
        m_root = std::make_unique<UCTNode>(FastBoard::PASS, 0.0f);
    }

    // Clear last_rootstate to prevent accidental use.
    m_last_rootstate.reset(nullptr);

    // Check how big our search tree (reused or new) is.
    m_nodes = m_root->count_nodes_and_clear_expand_state();

    #ifndef NDEBUG
    if (m_nodes > 0) {
        myprintf("update_root, %d -> %d nodes (%.1f%% reused)\n",
            start_nodes, m_nodes.load(), 100.0 * m_nodes.load() / start_nodes);
    }
    #endif
}

float UCTSearch::get_min_psa_ratio() const {
    const auto mem_full = UCTNodePointer::get_tree_size() / static_cast<float>(cfg_max_tree_size);
    // If we are halfway through our memory budget, start trimming
    // moves with very low policy priors.
    if (mem_full > 0.5f) {
        // Memory is almost exhausted, trim more aggressively.
        if (mem_full > 0.95f) {
            // if completely full just stop expansion by returning an impossible number
            if (mem_full >= 1.0f) {
                return 2.0f;
            }
            return 0.01f;
        }
        return 0.001f;
    }
    return 0.0f;
}

SearchResult UCTSearch::play_simulation(GameState & currstate,
                                        UCTNode* const node) {
    auto result = SearchResult{};

    node->virtual_loss();

    if (node->expandable()) {
        if (currstate.get_passes() >= 2) {
            if (cfg_japanese_mode && m_chn_scoring) {
                result = SearchResult::from_eval(node->get_net_eval(),
                                                 node->get_net_alpkt(),
                                                 node->get_net_beta());
#ifndef NDEBUG
                myprintf(": Chn (net) %.3f\n", node->get_net_alpkt());
#endif
            } else {
                auto score = currstate.final_score();
                result = SearchResult::from_score(score);
                node->set_values(Utils::winner(score), score, 10.0f);
#ifndef NDEBUG
                myprintf(": TT (score) %.3f\n", score);
#endif
#ifdef USE_EVALCMD
                if (node->get_progid() != -1) {
                    node->set_progid(m_nodecounter++);
                }
#endif
            }
        } else {
                float value, alpkt, beta;
                const auto had_children = node->has_children();
            const auto success =
                node->create_children(m_network, m_nodes, currstate, value, alpkt, beta,
                                      get_min_psa_ratio());
            if (!had_children && success) {
#ifdef USE_EVALCMD
                node->set_progid(m_nodecounter++);
#endif
                result = SearchResult::from_eval(value, alpkt, beta);
#ifndef NDEBUG
                myprintf(": new %.3f\n", alpkt);
#endif
            } else {
#ifndef NDEBUG
                myprintf(": create_children() failed!\n");
#endif
            }
        }
    }

    auto restrict_return = false;
    auto update_with_current = false;

    if (node->has_children() && !result.valid()) {
        auto next = node->uct_select_child(currstate,
                                           node == m_root.get(),
                                           m_per_node_maxvisits,
                                           m_allowed_root_children,
                                           m_nopass);
        if (next != nullptr) {
            auto move = next->get_move();
            next->set_eval_bonus_father(node->get_eval_bonus());
            next->set_eval_base_father(node->get_eval_base());

            restrict_return = (cfg_restrict_tt &&
                               currstate.get_passes() == 1 &&
                               move == FastBoard::PASS);
            
            currstate.play_move(move);
            if (move != FastBoard::PASS && currstate.superko()) {
                next->invalidate();
            } else {
#ifndef NDEBUG
                myprintf("%4s:%2d ",
                         currstate.move_to_text(move).c_str(), next->get_visits());
#endif

                const auto allowed = m_allowed_root_children;
                m_allowed_root_children = {};
                if (m_nopass) {
                    currstate.set_passes(0);
                }
                result = play_simulation(currstate, next);
                m_allowed_root_children = allowed;
                if (m_stopping_flag && node == m_root.get()) {
                    m_bestmove = move;
                }
            }
            update_with_current = (restrict_return &&
                                   node->low_visits_child(next));
        }
    }

    auto current_node_result = SearchResult::from_eval(node->get_net_eval(),
                                                       node->get_net_alpkt(),
                                                       node->get_net_beta());

    if (result.valid()) {
        // If we are restricting Tromp-Taylor, this is a first pass
        // the selected child is second pass, then in some cases we
        // update this node with result (which would be TT score), and
        // in some cases we update with network's evaluation for this
        // node. In particular we update with TT only after the second
        // pass is visited a fair number of times. This way if the
        // second pass would lose because of dead groups and TT
        // scoring, but there are better and preferred moves, then
        // this node evaluation is not polluted with unrealistic
        // losing scores for the opponent and hence it will not
        // generally be chosen because of that. On the other hand, if
        // the second pass is actually a reasonable move even with TT
        // scoring (because it wins, or because there are no better
        // options) then this node is updated with TT score so that
        // current player can knowingly choose whether to try the
        // first pass again.
        const auto & result_for_updating = update_with_current ?
            current_node_result : result;
        const auto eval = m_network.m_value_head_sai ?
            result_for_updating.eval_with_bonus(node->get_eval_bonus_father(),
                                                node->get_eval_base_father()) :
            result_for_updating.eval();
        node->update(eval);
        // should check whether it is sai or lz before updating alpkt_median
        node->update_alpkt_median(result_for_updating.get_alpkt());

        if (m_stopping_visits >= 1 && m_stopping_moves.size() >= 1) {
            if (node->get_visits() >= m_stopping_visits) {
                if (is_stopping(node->get_move())) {
                    m_stopping_flag = true;
                }
            }
        }
    }
    node->virtual_loss_undo();

    // If we are restricting Tromp-Taylor, this is a first pass and
    // selected child is second pass, do not return result (which
    // would be TT score), but network's evaluation for this
    // node. This way TT score cannot propagate to more than two nodes
    // (the first and the second passes).
    return restrict_return ? current_node_result : result;
}

void UCTSearch::dump_stats(FastState & state, UCTNode & parent) {
    if (cfg_quiet || !parent.has_children()) {
        return;
    }

    const int color = state.get_to_move();

    auto max_visits = 0;
    for (const auto& node : parent.get_children()) {
        max_visits = std::max(max_visits, node->get_visits());
    }

    // sort children, put best move on top
    parent.sort_children(color, cfg_lcb_min_visit_ratio * max_visits);

    if (parent.get_first_child()->first_visit()) {
        return;
    }

    int movecount = 0;
    for (const auto& node : parent.get_children()) {
        // Always display at least two moves. In the case there is
        // only one move searched the user could get an idea why.
        if (++movecount > 2 && !node->get_visits()) break;

        auto move = state.move_to_text(node->get_move());
        auto tmpstate = FastState{state};
        tmpstate.play_move(node->get_move());
        auto pv = move + " " + get_pv(tmpstate, *node);
        
#ifdef NDEBUG
        myprintf("%4s -> %7d (V: %5.2f%%) (LCB: %5.2f%%) (N: %5.2f%%) (A: %4.1f) PV: %s\n",
                 move.c_str(),
                 node->get_visits(),
                 node->get_visits() ? node->get_raw_eval(color)*100.0f : 0.0f,
                 std::max(0.0f, node->get_eval_lcb(color) * 100.0f),
                 node->get_policy() * 100.0f,
                 node->get_alpkt_online_median(),
                 pv.c_str());
#else
        myprintf("%4s -> %7d (U: %5.2f%%, q: %5.2f%%, num: %5.2f, den: %4d) "
                 "(V: %5.2f%%) (LCB: %8.5f%%) (N: %5.2f%%) (A: %4.1f) PV: %s\n",
                 move.c_str(),
                 node->get_visits(),
                 node->get_urgency()[0] * 100.0f,
                 node->get_urgency()[2] * 100.0f,
                 node->get_urgency()[4],
                 int(node->get_urgency()[3]),
                 node->get_visits() ? node->get_raw_eval(color)*100.0f : 0.0f,
                 node->get_eval_lcb(color) * 100.0f,
                 node->get_policy() * 100.0f,
                 node->get_alpkt_online_median(),
                 pv.c_str());
#endif
    }
    tree_stats(parent);
}

void UCTSearch::output_analysis(FastState & state, UCTNode & parent) {
    // We need to make a copy of the data before sorting
    auto sortable_data = std::vector<OutputAnalysisData>();

    if (!parent.has_children()) {
        return;
    }

    const auto color = state.get_to_move();

    auto max_visits = 0;
    for (const auto& node : parent.get_children()) {
        max_visits = std::max(max_visits, node->get_visits());
    }

    for (const auto& node : parent.get_children()) {
        // Send only variations with visits, unless more moves were
        // requested explicitly.
        if (!node->get_visits()
            && sortable_data.size() >= cfg_analyze_tags.post_move_count()) {
            continue;
        }
        auto move = state.move_to_text(node->get_move());
        auto tmpstate = FastState{state};
        tmpstate.play_move(node->get_move());
        auto rest_of_pv = get_pv(tmpstate, *node);
        auto pv = move + (rest_of_pv.empty() ? "" : " " + rest_of_pv);
        auto move_eval = node->get_visits() ? node->get_raw_eval(color) : 0.0f;
        auto policy = node->get_policy();
        auto lcb = node->get_eval_lcb(color);
        auto visits = node->get_visits();
        // Need at least 2 visits for valid LCB.
        auto lcb_ratio_exceeded = visits > 2 &&
            visits > max_visits * cfg_lcb_min_visit_ratio;
        // Store data in array
        sortable_data.emplace_back(move, visits,
                                   move_eval, policy, pv, lcb, lcb_ratio_exceeded);
    }
    // Sort array to decide order
    std::stable_sort(rbegin(sortable_data), rend(sortable_data));

    auto i = 0;
    // Output analysis data in gtp stream
    for (const auto& node : sortable_data) {
        if (i > 0) {
            gtp_printf_raw(" ");
        }
        gtp_printf_raw(node.get_info_string(i).c_str());
        i++;
    }
    gtp_printf_raw("\n");
}

void UCTSearch::tree_stats(const UCTNode& node) {
    size_t nodes = 0;
    size_t non_leaf_nodes = 0;
    size_t depth_sum = 0;
    size_t max_depth = 0;
    size_t children_count = 0;

    std::function<void(const UCTNode& node, size_t)> traverse =
          [&](const UCTNode& node, size_t depth) {
        nodes += 1;
        non_leaf_nodes += node.get_visits() > 1;
        depth_sum += depth;
        if (depth > max_depth) max_depth = depth;

        for (const auto& child : node.get_children()) {
            if (child.get_visits() > 0) {
                children_count += 1;
                traverse(*(child.get()), depth+1);
            } else {
                nodes += 1;
                depth_sum += depth+1;
                if (depth >= max_depth) max_depth = depth+1;
            }
        }
    };

    traverse(node, 0);

    if (nodes > 0) {
        myprintf("%.1f average depth, %d max depth\n",
                 (1.0f*depth_sum) / nodes, max_depth);
        myprintf("%d non leaf nodes, %.2f average children\n",
                 non_leaf_nodes, (1.0f*children_count) / non_leaf_nodes);
    }
}


void UCTSearch::tree_stats() {
    tree_stats(*(m_root.get()));
    myprintf("%d visits, %d nodes\n", m_root->get_visits(), m_nodes.load());
    const auto maxplay = (m_maxplayouts == UNLIMITED_PLAYOUTS) ?
        "inf" : std::to_string(m_maxplayouts);
    const auto maxvisit = (m_maxvisits == UNLIMITED_PLAYOUTS) ?
        "inf" : std::to_string(m_maxvisits);
    myprintf("lambda: %.1f, mu: %.1f, maxvisits: %s, maxplayouts: %s\n\n",
             cfg_lambda, cfg_mu, maxvisit.c_str(), maxplay.c_str());
}


bool UCTSearch::should_resign(passflag_t passflag, float besteval) {
    if (passflag & UCTSearch::NORESIGN) {
        // resign not allowed
        return false;
    }

    if (cfg_resignpct == 0) {
        // resign not allowed
        return false;
    }

    const size_t num_intersections = m_rootstate.board.get_boardsize()
                                   * m_rootstate.board.get_boardsize();
    const auto move_threshold = num_intersections / 4;
    const auto movenum = m_rootstate.get_movenum();
    if (movenum <= move_threshold) {
        // too early in game to resign
        return false;
    }

    const auto color = m_rootstate.board.get_to_move();

    const auto is_default_cfg_resign = cfg_resignpct < 0;
    const auto resign_threshold = cfg_resign_threshold;

    if (besteval > resign_threshold) {
        // eval > cfg_resign
        return false;
    }

    if ((m_rootstate.get_handicap() > 0)
            && (color == FastBoard::WHITE)
            && is_default_cfg_resign) {
        const auto handicap_resign_threshold =
            resign_threshold / (1 + m_rootstate.get_handicap());

        // Blend the thresholds for the first ~215 moves.
        auto blend_ratio = std::min(1.0f, movenum / (0.6f * num_intersections));
        auto blended_resign_threshold = blend_ratio * resign_threshold
            + (1 - blend_ratio) * handicap_resign_threshold;
        if (besteval > blended_resign_threshold) {
            // Allow lower eval for white in handicap games
            // where opp may fumble.
            return false;
        }
    }

    if (!m_rootstate.is_move_legal(color, FastBoard::RESIGN)) {
        return false;
    }

    return true;
}

int UCTSearch::get_best_move(passflag_t passflag) {
    const int color = m_rootstate.board.get_to_move();

    auto max_visits = 0;
    for (const auto& node : m_root->get_children()) {
        max_visits = std::max(max_visits, node->get_visits());
    }

    // Make sure best is first
    m_root->sort_children(color,  cfg_lcb_min_visit_ratio * max_visits);

    // Check whether to randomize the best move proportional
    // to the playout counts, early game only.
    const auto movenum = m_rootstate.get_movenum();

    // following code requires that there are children!
    assert(!m_root->get_children().empty());

    if (movenum < static_cast<size_t>(cfg_random_cnt)) {

        auto is_blunder = false;
        auto non_blunders = std::vector<int>{};
        tie(is_blunder,non_blunders) =
            m_root->randomize_first_proportionally(color,
                m_rootstate.is_blunder_allowed());
        m_rootstate.set_non_blunders(non_blunders);

        if (should_resign(passflag, m_root->get_first_child()->get_eval(color))) {
            myprintf("Random move would lead to immediate resignation... \n"
                     "Reverting to best move.\n");
            m_root->sort_children(color,  cfg_lcb_min_visit_ratio * max_visits);
        } else if (is_blunder) {
            myprintf("Random move is a blunder.\n");
        }
    } else {
        const auto non_blunders = std::vector<int>{m_root->get_first_child()->get_move()};
        m_rootstate.set_non_blunders(non_blunders);
    }

    auto first_child = m_root->get_first_child();
    assert(first_child != nullptr);

    auto bestmove = first_child->get_move();
    auto besteval = first_child->first_visit() ? 0.5f : first_child->get_raw_eval(color);

    // do we want to fiddle with the best move because of the rule set?
    if (passflag & UCTSearch::NOPASS || cfg_japanese_mode) {
        // were we going to pass?
        if (bestmove == FastBoard::PASS) {
            UCTNode * nopass = m_root->get_nopass_child(m_rootstate);

            if (nopass != nullptr) {
                myprintf("Preferring not to pass.\n");
                bestmove = nopass->get_move();
                if (nopass->first_visit()) {
                    besteval = 1.0f;
                } else {
                    besteval = nopass->get_raw_eval(color);
                }
            } else {
                myprintf("Pass is the only acceptable move.\n");
            }
        }
    } else if (!cfg_dumbpass) {
        const auto relative_score =
            (color == FastBoard::BLACK ? 1 : -1) * m_rootstate.final_score();
        if (bestmove == FastBoard::PASS) {
            // Either by forcing or coincidence passing is
            // on top...check whether passing loses instantly
            // do full count including dead stones.
            // In a reinforcement learning setup, it is possible for the
            // network to learn that, after passing in the tree, the two last
            // positions are identical, and this means the position is only won
            // if there are no dead stones in our own territory (because we use
            // Trump-Taylor scoring there). So strictly speaking, the next
            // heuristic isn't required for a pure RL network, and we have
            // a commandline option to disable the behavior during learning.
            // On the other hand, with a supervised learning setup, we fully
            // expect that the engine will pass out anything that looks like
            // a finished game even with dead stones on the board (because the
            // training games were using scoring with dead stone removal).
            // So in order to play games with a SL network, we need this
            // heuristic so the engine can "clean up" the board. It will still
            // only clean up the bare necessity to win. For full dead stone
            // removal, kgs-genmove_cleanup and the NOPASS mode must be used.

            // Do we lose by passing?
            if (relative_score < 0.0f) {
                myprintf("Passing loses :-(\n");
                // Find a valid non-pass move.
                UCTNode * nopass = m_root->get_nopass_child(m_rootstate);
                if (nopass != nullptr) {
                    myprintf("Avoiding pass because it loses.\n");
                    bestmove = nopass->get_move();
                    if (nopass->first_visit()) {
                        besteval = 1.0f;
                    } else {
                        besteval = nopass->get_raw_eval(color);
                    }
                } else {
                    myprintf("No alternative to passing.\n");
                }
            } else if (relative_score > 0.0f) {
                myprintf("Passing wins :-)\n");
            } else {
                myprintf("Passing draws :-|\n");
                // Find a valid non-pass move.
                const auto nopass = m_root->get_nopass_child(m_rootstate);
                if (nopass != nullptr && !nopass->first_visit()) {
                    const auto nopass_eval = nopass->get_raw_eval(color);
                    if (nopass_eval > 0.5f) {
                        myprintf("Avoiding pass because there could be a winning alternative.\n");
                        bestmove = nopass->get_move();
                        besteval = nopass_eval;
                    }
                }
                if (bestmove == FastBoard::PASS) {
                    myprintf("No seemingly better alternative to passing.\n");
                }
            }
        } else if (m_rootstate.get_last_move() == FastBoard::PASS) {
            // Opponents last move was passing.
            // We didn't consider passing. Should we have and
            // end the game immediately?

            if (!m_rootstate.is_move_legal(color, FastBoard::PASS)) {
                myprintf("Passing is forbidden, I'll play on.\n");
            // do we lose by passing?
            } else if (relative_score < 0.0f) {
                myprintf("Passing loses, I'll play on.\n");
            } else if (relative_score > 0.0f) {
                myprintf("Passing wins, I'll pass out.\n");
                bestmove = FastBoard::PASS;
            } else {
                myprintf("Passing draws, make it depend on evaluation.\n");
                if (besteval < 0.5f) {
                    bestmove = FastBoard::PASS;
                }
            }
        }
    }

    // if we aren't passing, should we consider resigning?
    if (bestmove != FastBoard::PASS) {
      //      myprintf("Eval (%.2f%%)\n", 100.0f * besteval);
        if (should_resign(passflag, besteval)) {
            myprintf("Eval (%.2f%%) looks bad. Resigning.\n",
                     100.0f * besteval);
            bestmove = FastBoard::RESIGN;
        }
    }

    return bestmove;
}

std::string UCTSearch::get_pv(FastState & state, UCTNode& parent) {
    if (!parent.has_children()) {
        return std::string();
    }

    if (parent.expandable()) {
        // Not fully expanded. This means someone could expand
        // the node while we want to traverse the children.
        // Avoid the race conditions and don't go through the rabbit hole
        // of trying to print things from this node.
        return std::string();
    }

    auto& best_child = parent.get_best_root_child(state.get_to_move());
    if (best_child.first_visit()) {
        return std::string();
    }
    auto best_move = best_child.get_move();
    auto res = state.move_to_text(best_move);

    state.play_move(best_move);

    auto next = get_pv(state, best_child);
    if (!next.empty()) {
        res.append(" ").append(next);
    }
    return res;
}

std::string UCTSearch::get_analysis(int playouts) {
    FastState tempstate = m_rootstate;
    int color = tempstate.board.get_to_move();

    auto pvstring = get_pv(tempstate, *m_root);
    float winrate = 100.0f * m_root->get_raw_eval(color);
    return str(boost::format("Playouts: %d, Win: %5.2f%%, PV: %s")
        % playouts % winrate % pvstring.c_str());
}

bool UCTSearch::is_running() const {
    return m_run && UCTNodePointer::get_tree_size() < cfg_max_tree_size;
}

int UCTSearch::est_playouts_left(int elapsed_centis, int time_for_move) const {
    auto playouts = m_playouts.load();
    const auto playouts_left =
        std::max(0, std::min(m_maxplayouts - playouts,
                             m_maxvisits - m_root->get_visits()));
    // Wait for at least 1 second and 100 playouts
    // so we get a reliable playout_rate.
    if (elapsed_centis < 100 || playouts < 100) {
        return playouts_left;
    }
    const auto playout_rate = 1.0f * playouts / elapsed_centis;
    const auto time_left = std::max(0, time_for_move - elapsed_centis);
    return std::min(playouts_left,
                    static_cast<int>(std::ceil(playout_rate * time_left)));
}

size_t UCTSearch::prune_noncontenders(int color, int elapsed_centis, int time_for_move, bool prune) {
    auto lcb_max = 0.0f;
    auto Nfirst = 0;
    // There are no cases where the root's children vector gets modified
    // during a multithreaded search, so it is safe to walk it here without
    // taking the (root) node lock.
    for (const auto& node : m_root->get_children()) {
        if (node->valid()) {
            const auto visits = node->get_visits();
            if (visits > 0) {
                lcb_max = std::max(lcb_max, node->get_eval_lcb(color));
            }
            Nfirst = std::max(Nfirst, visits);
        }
    }
    const auto min_required_visits =
        Nfirst - est_playouts_left(elapsed_centis, time_for_move);
    auto pruned_nodes = size_t{0};
    for (const auto& node : m_root->get_children()) {
        if (node->valid()) {
            const auto visits = node->get_visits();
            const auto has_enough_visits =
                visits >= min_required_visits;
            // Avoid pruning moves that could have the best lower confidence
            // bound.
            const auto high_winrate = visits > 0 ?
                node->get_raw_eval(color) >= lcb_max : false;
            const auto prune_this_node = !(has_enough_visits || high_winrate);

            if (prune) {
                node->set_active(!prune_this_node);
            }
            if (prune_this_node) {
                ++pruned_nodes;
            }
        }
    }

    assert(pruned_nodes < m_root->get_children().size());
    return pruned_nodes;
}

bool UCTSearch::have_alternate_moves(int elapsed_centis, int time_for_move) {
    if (cfg_timemanage == TimeManagement::OFF) {
        return true;
    }
    auto my_color = m_rootstate.get_to_move();
    // For self play use. Disables pruning of non-contenders to not bias the training data.
    auto prune = cfg_timemanage != TimeManagement::NO_PRUNING;
    auto pruned = prune_noncontenders(my_color, elapsed_centis, time_for_move, prune);
    if (pruned < m_root->get_children().size() - 1) {
        return true;
    }
    // If we cannot save up time anyway, use all of it. This
    // behavior can be overruled by setting "fast" time management,
    // which will cause Leela to quickly respond to obvious/forced moves.
    // That comes at the cost of some playing strength as she now cannot
    // think ahead about her next moves in the remaining time.
    auto tc = m_rootstate.get_timecontrol();
    if (!tc.can_accumulate_time(my_color)
        || m_maxplayouts < UCTSearch::UNLIMITED_PLAYOUTS) {
        if (cfg_timemanage != TimeManagement::FAST) {
            return true;
        }
    }
    // In a timed search we will essentially always exit because
    // the remaining time is too short to let another move win, so
    // avoid spamming this message every move. We'll print it if we
    // save at least half a second.
    if (time_for_move - elapsed_centis > 50) {
        myprintf("%.1fs left, stopping early.\n",
                    (time_for_move - elapsed_centis) / 100.0f);
    }
    return false;
}

bool UCTSearch::stop_thinking(int elapsed_centis, int time_for_move) const {
    return m_playouts >= m_maxplayouts
           || m_root->get_visits() >= m_maxvisits
           || elapsed_centis >= time_for_move;
}

void UCTWorker::operator()() {
    do {
        auto currstate = std::make_unique<GameState>(m_rootstate);
        auto result = m_search->play_simulation(*currstate, m_root);
        if (result.valid()) {
            m_search->increment_playouts();
        }
    } while (m_search->is_running());
}

void UCTSearch::increment_playouts() {
    m_playouts++;
    //    myprintf("\n");
}


void UCTSearch::print_move_choices_by_policy(KoState & state, UCTNode & parent, int at_least_as_many, float probab_threash) {
    parent.sort_children_by_policy();
    int movecount = 0;
    float policy_value_of_move = 1.0f;
    for (const auto& node : parent.get_children()) {
        if (++movecount > at_least_as_many && policy_value_of_move<probab_threash)
            break;

        policy_value_of_move = node.get_policy();
        std::string tmp = state.move_to_text(node.get_move());
        myprintf("%4s %4.1f",
                 tmp.c_str(),
                 policy_value_of_move * 100.0f);
    }
    myprintf("\n");
}


int UCTSearch::think(int color, passflag_t passflag) {
    myprintf("Debug...\n");
    // Start counting time for us
    m_rootstate.start_clock(color);

    // set up timing info
    Time start;

    update_root();
    // set side to move
    m_rootstate.board.set_to_move(color);

    auto time_for_move =
        m_rootstate.get_timecontrol().max_time_for_move(
            m_rootstate.board.get_boardsize(),
            color, m_rootstate.get_movenum());

    myprintf("Thinking at most %.1f seconds...\n", time_for_move/100.0f);

    // create a sorted list of legal moves (make sure we
    // play something legal and decent even in time trouble)
    m_root->prepare_root_node(m_network, color, m_nodes, m_rootstate);

    if (m_rootstate.get_movenum() < static_cast<size_t>(cfg_random_cnt)) {
        m_per_node_maxvisits = static_cast<int>((1.0 - cfg_noise_weight) * m_maxvisits);
    } else {
        m_per_node_maxvisits = 0;
    }

#ifndef NDEBUG
    myprintf("We are at root. Move choices by policy are: ");
    print_move_choices_by_policy(m_rootstate, *m_root, 5, 0.01f);
    myprintf("\n");
#endif

    m_run = true;
    int cpus = cfg_num_threads;
    myprintf("cpus=%i\n", cpus);
    ThreadGroup tg(thread_pool);
    for (int i = 1; i < cpus; i++) {
      tg.add_task(UCTWorker(m_rootstate, this, m_root.get()));
    }

    auto keeprunning = true;
    auto last_update = 0;
    auto last_output = 0;
    do {
        auto currstate = std::make_unique<GameState>(m_rootstate);

        auto result = play_simulation(*currstate, m_root.get());
        if (result.valid()) {
          increment_playouts();
        }

        Time elapsed;
        int elapsed_centis = Time::timediff_centis(start, elapsed);

        if (cfg_analyze_tags.interval_centis() &&
            elapsed_centis - last_output > cfg_analyze_tags.interval_centis()) {
            last_output = elapsed_centis;
            output_analysis(m_rootstate, *m_root);
        }

        // output some stats every few seconds
        // check if we should still search
        if (!cfg_quiet && elapsed_centis - last_update > 250) {
            last_update = elapsed_centis;
            myprintf("%s\n", get_analysis(m_playouts.load()).c_str());
        }
        keeprunning  = is_running();
        keeprunning &= !stop_thinking(elapsed_centis, time_for_move);
        if (m_per_node_maxvisits == 0) {
            keeprunning &= have_alternate_moves(elapsed_centis, time_for_move);
        }
    } while (keeprunning);

    // Make sure to post at least once.
    if (cfg_analyze_tags.interval_centis() && last_output == 0) {
        output_analysis(m_rootstate, *m_root);
    }

    // Stop the search.
    m_run = false;
    tg.wait_all();

    // Reactivate all pruned root children.
    for (const auto& node : m_root->get_children()) {
        node->set_active(true);
    }

    m_rootstate.stop_clock(color);
    if (!m_root->has_children()) {
        return FastBoard::PASS;
    }

    // Display search info.
    myprintf("\n");
    dump_stats(m_rootstate, *m_root);

    int bestmove = get_best_move(passflag);
    float est_score;
    if (cfg_japanese_mode) {
        if (!is_better_move(bestmove, FastBoard::PASS, est_score)) {
            auto chn_endstate = std::make_unique<GameState>(m_rootstate);
            chn_endstate->add_komi(est_score);
#ifndef NDEBUG
            chn_endstate->display_state();
            myprintf("Komi modified to %.1f. Roll-out starting.\n",
                     chn_endstate->get_komi());
#endif
            auto FRO_tree = std::make_unique<UCTSearch>(*chn_endstate, m_network);
            FRO_tree->fast_roll_out();
#ifndef NDEBUG
            myprintf("Roll-out completed.\n");
            chn_endstate->display_state();
#endif

            bestmove = FastBoard::PASS;
            auto jap_endboard = std::make_unique<FullBoard>(m_rootstate.board);
            if (jap_endboard->remove_dead_stones(chn_endstate->board)) {
#ifndef NDEBUG
                myprintf("Removal of dead stones completed.\n");
                jap_endboard->display_board();
#endif
                select_dame_sequence(jap_endboard.get());
                bestmove = m_bestmove;
#ifndef NDEBUG
                myprintf("Chosen move is %s.\n",
                         jap_endboard->move_to_text(bestmove).c_str());
#endif
            } else {
                myprintf ("Removal didn't work!\n");
            }
        }
    }

    Training::record(m_network, m_rootstate, *m_root);

    // The function set_eval() updates the current KoState but not
    // GameState history; when the next move is played, the updated
    // KoState is written in history, hence the current evaluation is
    // always stored into the following move record. This is correct,
    // as we are going to store the statistics of the chosen
    // move. Exception: if the best move is RESIGN, play_move()
    // updates the last state and does not create a new one.
    if(bestmove != FastBoard::RESIGN) {
        auto chosen_child = m_root->get_first_child();
        if(chosen_child->get_move() != bestmove) {
            for(auto& child : m_root->get_children()) {
                if(child->get_move() == bestmove) {
                    chosen_child = child.get();
                    break;
                }
            }
        }
        if(chosen_child) {
            const auto alpkt = chosen_child->get_net_alpkt();
            const auto beta = chosen_child->get_net_beta();
            const auto x_lambda = chosen_child->get_eval_bonus();
            const auto x_mu = chosen_child->get_eval_base();
            const StateEval ev(chosen_child->get_visits(),
                               alpkt, beta, sigmoid(alpkt, beta, 0.0f).first,
                               Utils::sigmoid_interval_avg(alpkt, beta,
                                                           x_mu, x_lambda),
                               x_lambda, x_mu,
                               chosen_child->get_eval(FastBoard::BLACK),
                               chosen_child->estimate_alpkt(0, true),
                               chosen_child->get_alpkt_online_median());
            m_rootstate.set_eval(ev);
            
#ifndef NDEBUG
            myprintf("visits=%d, alpkt=%.2f, beta=%.3f, pi=%.3f, agent=%.3f, "
                     "avg=%.3f, alpkt_med=%.3f, alpkt_online=%.3f, "
                     "x_mu=%.1f, x_lambda=%.1f\n",
                     ev.visits, ev.alpkt, ev.beta, ev.pi, ev.agent_eval,
                     ev.agent_eval_avg, ev.alpkt_median, ev.alpkt_online_median,
                     ev.agent_x_mu, ev.agent_x_lambda);
#endif
        }
    }

    Time elapsed;
    int elapsed_centis = Time::timediff_centis(start, elapsed);
    myprintf("%d visits, %d nodes, %d playouts, %.0f n/s\n\n",
             m_root->get_visits(),
             m_nodes.load(),
             m_playouts.load(),
             (m_playouts * 100.0) / (elapsed_centis+1));

#ifdef USE_OPENCL
#ifndef NDEBUG
    myprintf("batch stats: %d %d\n",
        batch_stats.single_evals.load(),
        batch_stats.batch_evals.load()
    );
#endif
#endif

    //    int bestmove = get_best_move(passflag);

    // Save the explanation.
    m_think_output =
        str(boost::format("move %d, %c => %s\n%s")
        % m_rootstate.get_movenum()
        % (color == FastBoard::BLACK ? 'B' : 'W')
        % m_rootstate.move_to_text(bestmove).c_str()
        % get_analysis(m_root->get_visits()).c_str());

    // Copy the root state. Use to check for tree re-use in future calls.
    m_last_rootstate = std::make_unique<GameState>(m_rootstate);
    return bestmove;
}


#ifdef USE_EVALCMD
Network::Netresult UCTSearch::dump_evals(int req_playouts, std::string & dump_str,
                                         std::string & sgf_str) {
    update_root(true);
    //    m_rootstate.board.set_to_move(color);
    m_root->prepare_root_node(m_network, m_rootstate.board.get_to_move(), m_nodes, m_rootstate);

    for (auto n=0 ; n < req_playouts ; n++) {
        // todo: check rootnode visits instead of playouts
        auto currstate = std::make_unique<GameState>(m_rootstate);

        auto result = play_simulation(*currstate, m_root.get());
        if (!result.valid()) {
            myprintf("Invalid result at n=%d.\n",n);
        } else {
        increment_playouts();
        }
    }

    auto color = m_rootstate.board.get_to_move();
    std::vector<float> value_vec;
    std::vector<float> alpkt_vec;
    std::vector<float> beta_vec;
    dump_evals_recursion(dump_str, m_root.get(), -2, color, sgf_str,
                         value_vec, alpkt_vec, beta_vec);

    Network::Netresult result;
    {
        std::vector<float> freq_visits;
        m_root->get_children_visits(m_rootstate, *(m_root.get()), freq_visits, true);

        std::copy(freq_visits.begin(), freq_visits.end()-1, result.policy.begin());
        result.policy_pass = freq_visits.back();
    }

    result.value = Utils::median(value_vec);
    const auto alpkt_median = Utils::median(alpkt_vec);
    result.alpha = (alpkt_median + m_rootstate.get_komi())
        * (color==FastBoard::BLACK ? 1.0 : -1.0);
    result.beta = Utils::median(beta_vec);

    return result;
}

void UCTSearch::dump_evals_recursion(std::string & dump_str,
                                     UCTNode* const node,
                                     int father_progid, int color,
                                     std::string & sgf_str,
                                     std::vector<float> & value_vec,
                                     std::vector<float> & alpkt_vec,
                                     std::vector<float> & beta_vec) {
    node->sort_children(color, 0.0f);
    std::vector<UCTNode *> visited_children;
    for (const UCTNodePointer& it : node->get_children()) {
        if ( it.get_visits()!=0 ) {
            visited_children.push_back(it.get());
        }
    }

    {
        std::stringstream ss;

        if (dump_str.size() == 0) {
          ss << "move"
             << ",prog_id"
             << ",father_prog_id"
             << ",policy"
             << ",net_eval"
             << ",alpkt"
             << ",beta"
             << ",bonus"
             << ",base"
             << ",visits"
             << ",agent_eval"
#ifndef NDEBUG
             << ",urgency"
             << ",psa"
             << ",q"
             << ",denom"
             << ",numer"
#endif
             << ",children"
             << std::endl;
        }

        ss << m_rootstate.board.move_to_text(node->get_move());
        ss << "," << node->get_progid();
        ss << "," << father_progid;
        ss << "," << node->get_policy();
        ss << "," << node->get_net_eval();
        ss << "," << node->get_net_alpkt();
        ss << "," << node->get_net_beta();
        ss << "," << node->get_eval_bonus();
        ss << "," << node->get_eval_base();
        ss << "," << node->get_visits();
        ss << "," << node->get_agent_eval(FastBoard::BLACK);
#ifndef NDEBUG
        ss << "," << node->get_urgency()[0];
        ss << "," << node->get_urgency()[1];
        ss << "," << node->get_urgency()[2];
        ss << "," << node->get_urgency()[3];
        ss << "," << node->get_urgency()[4];
#endif
        ss << "," << visited_children.size();
        for (auto childptr : visited_children) {
          ss << "," << childptr->get_visits();
        }
        ss << std::endl;

        dump_str.append(ss.str());
    }

    value_vec.push_back(node->get_net_eval());
    alpkt_vec.push_back(node->get_net_alpkt());
    beta_vec.push_back(node->get_net_beta());

    if (father_progid >= -1) {
        std::string movestr = m_rootstate.board.move_to_text_sgf(node->get_move());
        if (color==FastBoard::BLACK) {
            sgf_str.append(" ;W[" + movestr + "]");
        } else {
            sgf_str.append(" ;B[" + movestr + "]");
        }

        std::stringstream cs;
        cs << node->get_policy();
        cs << ", " << node->get_net_eval();
        cs << ", " << node->get_net_alpkt();
        cs << ", " << node->get_net_beta();
        cs << ", " << node->get_visits();
        cs << ", " << node->get_agent_eval(FastBoard::BLACK);

        sgf_str.append("C[" + cs.str() + "]");
    }

    for (auto childptr : visited_children) {
        sgf_str.append(" (");
        dump_evals_recursion(dump_str, childptr, node->get_progid(), 1-color,
                             sgf_str, value_vec, alpkt_vec, beta_vec);
        sgf_str.append(")");
    }
    if (visited_children.size()) {
        sgf_str.append("\n");
    }
}
#endif

void UCTSearch::select_playable_dame(FullBoard *board) {

    for (const auto& node : m_root->get_children()) {
        const auto move = node->get_move();
#ifndef NDEBUG
        myprintf("Considering move %s...\n",
                 board->move_to_text(move).c_str());
#endif
        if (!board->is_dame(move)) {
            continue;
        }
#ifndef NDEBUG
        myprintf("         Dame found!\n");
#endif
        float est_score;
        if (!is_better_move(FastBoard::PASS, move, est_score)) {
            m_bestmove = move;
            return;
        }
    }

    m_bestmove = FastBoard::PASS;
    return;
}

void UCTSearch::select_dame_sequence(FullBoard *board) {
    const auto stop_moves = m_stopping_moves;

    board->reset_territory();
    board->find_dame(m_stopping_moves);

    if(m_stopping_moves.size() == 0) {
#ifndef NDEBUG
        myprintf("No dames left. Passing.\n");
#endif
        m_bestmove = FastBoard::PASS;
        return;
    }

    select_playable_dame(board);
    if (m_bestmove != FastBoard::PASS) {
#ifndef NDEBUG
        myprintf("Playable dame found.\n");
#endif
        return;
    }

    // There are still dames, but they cannot be
    // played directly by current player, so
    // expand the main UCT tree looking for short
    // sequences leading to a dame being played,
    // without losing points.
#ifndef NDEBUG
    myprintf("No dame directly playable. Looking for longer sequences.\n");
#endif

    const auto stop_visits = m_stopping_visits;
    const auto stop_flag = m_stopping_flag;

    m_stopping_visits = EXPLORE_MOVE_VISITS;
    m_stopping_flag = false;

    do {
        auto currstate = std::make_unique<GameState>(m_rootstate);

        play_simulation(*currstate, m_root.get());
    } while (!m_stopping_flag);

    m_stopping_moves = stop_moves;
    m_stopping_visits = stop_visits;
    m_stopping_flag = stop_flag;

    return;
}


// Brief output from last think() call.
std::string UCTSearch::explain_last_think() const {
    return m_think_output;
}

void UCTSearch::ponder() {
    auto disable_reuse = cfg_analyze_tags.has_move_restrictions();
    if (disable_reuse) {
        m_last_rootstate.reset(nullptr);
    }

    update_root();

    m_root->prepare_root_node(m_network, m_rootstate.board.get_to_move(),
                              m_nodes, m_rootstate);

    m_run = true;
    ThreadGroup tg(thread_pool);
    for (auto i = size_t{1}; i < cfg_num_threads; i++) {
        tg.add_task(UCTWorker(m_rootstate, this, m_root.get()));
    }
    Time start;
    auto keeprunning = true;
    auto last_output = 0;
    do {
        auto currstate = std::make_unique<GameState>(m_rootstate);
        auto result = play_simulation(*currstate, m_root.get());
        if (result.valid()) {
            increment_playouts();
        }
        if (cfg_analyze_tags.interval_centis()) {
            Time elapsed;
            int elapsed_centis = Time::timediff_centis(start, elapsed);
            if (elapsed_centis - last_output > cfg_analyze_tags.interval_centis()) {
                last_output = elapsed_centis;
                output_analysis(m_rootstate, *m_root);
            }
        }
        keeprunning  = is_running();
        keeprunning &= !stop_thinking(0, 1);
    } while (!Utils::input_pending() && keeprunning);

    // Make sure to post at least once.
    if (cfg_analyze_tags.interval_centis() && last_output == 0) {
        output_analysis(m_rootstate, *m_root);
    }

    // Stop the search.
    m_run = false;
    tg.wait_all();

    // Display search info.
    myprintf("\n");
    dump_stats(m_rootstate, *m_root);

    myprintf("\n%d visits, %d nodes\n\n", m_root->get_visits(), m_nodes.load());

    // Copy the root state. Use to check for tree re-use in future calls.
    if (!disable_reuse) {
        m_last_rootstate = std::make_unique<GameState>(m_rootstate);
    }
}

void UCTSearch::set_playout_limit(int playouts) {
    static_assert(std::is_convertible<decltype(playouts),
                                      decltype(m_maxplayouts)>::value,
                  "Inconsistent types for playout amount.");
    m_maxplayouts = std::min(playouts, UNLIMITED_PLAYOUTS);
}

void UCTSearch::set_visit_limit(int visits) {
    static_assert(std::is_convertible<decltype(visits),
                                      decltype(m_maxvisits)>::value,
                  "Inconsistent types for visits amount.");
    // Limit to type max / 2 to prevent overflow when multithreading.
    m_maxvisits = std::min(visits, UNLIMITED_PLAYOUTS);
}

float SearchResult::eval_with_bonus(float xbar, float xbase) const {
    return Utils::sigmoid_interval_avg(m_alpkt, m_beta, xbase, xbar);
}

bool UCTSearch::is_better_move(int move1, int move2, float & estimated_score) {
    bool is_better = true;

    const auto move1_nodeptr = m_root->select_child(move1);
    const auto move2_nodeptr = m_root->select_child(move2);
    if (move1_nodeptr == nullptr || move2_nodeptr == nullptr) {
        return false;
    }
    explore_move(move1);
    explore_move(move2);

    const auto color = m_rootstate.get_to_move();
    const auto passes = m_rootstate.get_passes() + 1;
    const auto move1_passes = passes * (move1 == FastBoard::PASS ? 1 : 0);
    const auto move2_passes = passes * (move2 == FastBoard::PASS ? 1 : 0);
    auto move1_score = move1_nodeptr->get_net_alpkt();
    auto move2_score = move2_nodeptr->get_net_alpkt();
    auto move1_median_score = move1_nodeptr->estimate_alpkt(move1_passes);
    auto move2_median_score = move2_nodeptr->estimate_alpkt(move2_passes);
    const auto komi = m_rootstate.get_komi();
    estimated_score = std::round(move1_median_score + komi) - komi;
    const auto delta_mesh = std::abs(estimated_score + komi
                                     -  std::round(move2_median_score + komi));
    if (color == FastBoard::WHITE) {
        move1_score *= -1.0;
        move2_score *= -1.0;
        move1_median_score *= -1.0;
        move2_median_score *= -1.0;
    }
    const auto delta = move1_median_score - move2_median_score;

    if (delta_mesh < 0.5 && delta < 0.5) {
        is_better = false;
    }
#ifndef NDEBUG
    const auto move1_eval = move1_nodeptr->get_eval(color);
    const auto move2_eval = move2_nodeptr->get_eval(color);
    myprintf("Komi: %.1f, delta: %.2f, mesh: %.2f.\n"
             "Move2 (%s) winrate drop: %5.2f%%.\n"
             "Points drop (net): %.2f-%.2f=%.2f.\n"
             "Points drop (subtree median): %.2f-%.2f=%.2f.\n",
             komi, delta, delta_mesh, m_rootstate.board.move_to_text(move2).c_str(),
             (move1_eval - move2_eval)*100.0f,
             move1_score, move2_score,
             move1_score - move2_score,
             move1_median_score, move2_median_score,
             move1_median_score - move2_median_score
             );
#endif

    return is_better;
}

void UCTSearch::fast_roll_out() {
    // consider putting time management here

    // Explore tree for at most 120% of FAST_ROLL_OUT_VISITS
    // per node but stop whenever the best two have at least
    // FAST_ROLL_OUT_VISITS.  In this way if one is better, it
    // is allowed to have more visits and hence the order of
    // the two moves after sort is meaningful.
    const auto old_maxvisits = m_per_node_maxvisits;
    m_per_node_maxvisits = FAST_ROLL_OUT_VISITS + FAST_ROLL_OUT_VISITS / 5;

    // Double pass is scored with Tromp-Taylor in this tree,
    // so that the exploration leads to actually capturing all
    // the dead stones. (Since the komi was set as to result
    // in a jigo with perfect play.)
    const auto scoring = m_chn_scoring;
    m_chn_scoring = false;

    // Last move was chosen by scoring double pass with Chinese score
    // estimation, so there may have been a pass as last move with the
    // current player losing and not removing dead stones, so set
    // passes to zero before starting roll-out.
    m_rootstate.set_passes(0);

#ifndef NDEBUG
    auto step = 0;
#endif
    do {
        int consec_invalid = 0;
        auto chosenmove = FastBoard::PASS;

        update_root();

        m_root->prepare_root_node(m_network, m_rootstate.board.get_to_move(),
                                  m_nodes, m_rootstate, true);

#ifndef NDEBUG
        myprintf("Fast roll-out. Step %d. Komi %f\n", step++, m_rootstate.get_komi());
#endif
        do {
            auto currstate = std::make_unique<GameState>(m_rootstate);

            auto result = play_simulation(*currstate, m_root.get());

            if (result.valid()) {
                increment_playouts();
                consec_invalid = 0;
            } else {
                consec_invalid++;
            }
            m_root->sort_children(m_rootstate.get_to_move(), 0.0f);
            const auto second = m_root->get_nopass_child(m_rootstate);
            const auto first = m_root->select_child(FastBoard::PASS);
            //            const auto second = m_root->get_second_child();
            if (first == nullptr || second == nullptr || consec_invalid >= 3) {
                break;
            }

            const auto second_move = second->get_move();

            if (second_move == FastBoard::PASS) {
                break;
            }

            if (first->get_visits() < FAST_ROLL_OUT_VISITS ||
                second->get_visits() < FAST_ROLL_OUT_VISITS) {
                continue;
            }

#ifndef NDEBUG
            const auto first_move = FastBoard::PASS;

            myprintf("Roll-out step ends.\n"
                     "Best two moves (visits) are %s (%d) and %s (%d).\n",
                     m_rootstate.board.move_to_text(first_move).c_str(),
                     first->get_visits(),
                     m_rootstate.board.move_to_text(second_move).c_str(),
                     second->get_visits());
#endif
            // We choose the non-pass move if it doesn't seem to lose
            // points.
            const auto sign = m_rootstate.get_to_move() ==
                FastBoard::WHITE ? -1.0 : 1.0;
            const auto first_score = sign *
                first->estimate_alpkt(m_rootstate.get_passes()+1, true);
            const auto second_score = sign *
                second->estimate_alpkt(0, true);

#ifndef NDEBUG
            myprintf("Score estimation: %s=%f, %s=%f.\n",
                     m_rootstate.board.move_to_text(first_move).c_str(),
                     first_score,
                     m_rootstate.board.move_to_text(second_move).c_str(),
                     second_score);
#endif

            if (second_score > first_score - 0.5) {
                chosenmove = second_move;
                // If the best move is pass and the second-best loses
                // points, we pass
            } else {
                chosenmove = FastBoard::PASS;
            }
            break;
        } while (true);

        m_last_rootstate = std::make_unique<GameState>(m_rootstate);
        m_rootstate.play_move(chosenmove);
#ifndef NDEBUG
        myprintf("Chosen move: %s", m_rootstate.board.move_to_text(chosenmove).c_str());
        m_rootstate.display_state();
#endif
    } while(m_rootstate.get_passes() < 2);

    m_per_node_maxvisits = old_maxvisits;
    m_chn_scoring = scoring;
}

void UCTSearch::explore_move(int move) {
    const auto nodeptr = m_root->select_child(move);

    const auto allowed = m_allowed_root_children;

    while (nodeptr->get_visits() < EXPLORE_MOVE_VISITS) {
        auto currstate = std::make_unique<GameState>(m_rootstate);
        const auto nopass_old = m_nopass;

        m_nopass = true;
        m_allowed_root_children = {move};
        play_simulation(*currstate, m_root.get());
        m_nopass = nopass_old;
    }
    m_allowed_root_children = allowed;
}

void UCTSearch::explore_root_nopass() {
    while (m_root->get_visits() < EXPLORE_MOVE_VISITS) {
        auto currstate = std::make_unique<GameState>(m_rootstate);
        const auto nopass_old = m_nopass;

        m_nopass = true;
        play_simulation(*currstate, m_root.get());
        m_nopass = nopass_old;
    }
}

bool UCTSearch::is_stopping (int move) const {
    for (auto& stopping : m_stopping_moves) {
        if (move == stopping) {
            return true;
        }
    }

    return false;
}


float UCTSearch::final_japscore() {
    update_root();

    m_rootstate.set_passes(0);

    m_root->prepare_root_node(m_network, m_rootstate.board.get_to_move(),
                              m_nodes, m_rootstate, true);

    explore_root_nopass();


    const auto komi = m_rootstate.get_komi();
    const auto estimated_score =
        std::round(m_root->estimate_alpkt(0) + komi);
#ifndef NDEBUG
    myprintf("Estimated Chinese score of the board: %f.\n",
             estimated_score);
#endif

    auto chn_endstate = std::make_unique<GameState>(m_rootstate);
    chn_endstate->set_komi(estimated_score);
    auto FRO_tree = std::make_unique<UCTSearch>(*chn_endstate, m_network);

    FRO_tree->fast_roll_out();

    auto jap_endboard = std::make_unique<FullBoard>(m_rootstate.board);
    if (jap_endboard->remove_dead_stones(chn_endstate->board)) {
        return jap_endboard->territory_score(komi);
    } else {
        return NUM_INTERSECTIONS * 100.0;
    }
}
