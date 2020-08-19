/*
    This file is part of SAI, which is a fork of Leela Zero.
    Copyright (C) 2018-2019 Gian-Carlo Pascutto
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

#include <algorithm>
#include <cassert>
#include <iterator>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "FastBoard.h"
#include "FastState.h"
#include "KoState.h"
#include "Random.h"
#include "UCTNode.h"
#include "Utils.h"
#include "GTP.h"
#include "Network.h"

/*
 * These functions belong to UCTNode but should only be called on the root node
 * of UCTSearch and have been seperated to increase code clarity.
 */

using Utils::myprintf;

UCTNode* UCTNode::get_first_child() const {
    if (m_children.empty()) {
        return nullptr;
    }

    m_children.front().inflate();
    return m_children.front().get();
}

UCTNode* UCTNode::get_second_child() const {
    if (m_children.size() < 2) {
        return nullptr;
    }

    m_children[1].inflate();
    return m_children[1].get();
}

void UCTNode::kill_superkos(const GameState& state) {
    UCTNodePointer *pass_child = nullptr;
    size_t valid_count = 0;

    for (auto& child : m_children) {
        auto move = child->get_move();
        if (move != FastBoard::PASS) {
            KoState mystate = state;
            mystate.play_move(move);

            if (mystate.superko()) {
                // Don't delete nodes for now, just mark them invalid.
                child->invalidate();
            }
        } else {
            pass_child = &child;
        }
        if (child->valid()) {
            valid_count++;
        }
    }

    if (valid_count > 1 && pass_child &&
            !state.is_move_legal(state.get_to_move(), FastBoard::PASS)) {
        // Remove the PASS node according to "avoid" -- but only if there are
        // other valid nodes left.
        (*pass_child)->invalidate();
    }

    // Now do the actual deletion.
    m_children.erase(
        std::remove_if(begin(m_children), end(m_children),
                       [](const auto &child) { return !child->valid(); }),
        end(m_children)
    );
}

void UCTNode::dirichlet_noise(float epsilon, float alpha) {
    auto child_cnt = m_children.size();

    auto dirichlet_vector = std::vector<float>{};
    std::gamma_distribution<float> gamma(alpha, 1.0f);
    for (size_t i = 0; i < child_cnt; i++) {
        dirichlet_vector.emplace_back(gamma(Random::get_Rng()));
    }

    auto sample_sum = std::accumulate(begin(dirichlet_vector),
                                      end(dirichlet_vector), 0.0f);

    // If the noise vector sums to 0 or a denormal, then don't try to
    // normalize.
    if (sample_sum < std::numeric_limits<float>::min()) {
        return;
    }

    for (auto& v : dirichlet_vector) {
        v /= sample_sum;
    }

    child_cnt = 0;
    for (auto& child : m_children) {
        auto policy = child->get_policy();
        auto eta_a = dirichlet_vector[child_cnt++];
        policy = policy * (1 - epsilon) + epsilon * eta_a;
        child->set_policy(policy);
    }
}

FastState::move_flags_t
    UCTNode::randomize_first_proportionally(int color, bool is_blunder_allowed) {

    assert(!m_children.empty());

    FastState::move_flags_t flags;
    
    // if no choice is possible or when the number of visits is too low: nothing to do
    if (m_children.size() < 2 ||
        m_children.front()->get_visits() <= cfg_random_min_visits ){
        return flags;
    }

    auto accum = 0.0f;
    auto accum_vector = std::vector<double>{};
    auto accum_num = size_t{0};
    auto blunder_vector = std::vector<bool>{};

    double norm_factor = m_children.front()->get_visits();
    auto first_child_eval = m_children.front()->get_eval(color);

    for (const auto& child : m_children) {
        const auto visits = child->get_visits();

        if (visits <= cfg_random_min_visits) {
            continue;
        }

        const auto eval = child->get_eval(color);

        // Choose randomly among the moves whose evaluation is greater
        // than both resign threshold and cfg_blunder_thr (which
        // defaults to 5%). In this way also '-r 0' games will avoid
        // choosing very bad moves.
        const auto child_is_not_blunder =
            (eval >= first_child_eval-cfg_blunder_thr);
        if ( (child_is_not_blunder || is_blunder_allowed) &&
             eval > std::max(cfg_resign_threshold, cfg_losing_thr)) {
            accum += std::pow(visits / norm_factor, 1.0 / cfg_random_temp);
            accum_num++;
        }
        accum_vector.push_back(accum);
        blunder_vector.push_back(!child_is_not_blunder);

#ifndef NDEBUG
        // myprintf("--> %d. blunder? %s, drop=%f, "
        //           "accum=%f <--\n",
        //           accum_vector.size()-1,
        //           child_is_blunder ? "yes" : "no",
        //           first_child_eval - child->get_eval(color),
        //           accum);
#endif
    }

#ifndef NDEBUG
    if (is_blunder_allowed) {
        myprintf("Rnd_first: blunders still allowed. "
                 "Choice between %d moves with %d non-blunders.\n",
                 accum_num,
                 std::count(begin(blunder_vector),end(blunder_vector),false));
    } else {
        myprintf("Rnd_first: blunders NOT allowed. "
                 "Choice between %d good of %d moves with enough visits.\n",
                 accum_num,
                 accum_vector.size());
    }
#endif

    // In the case that all moves are below threshold, or only the
    // best move has enough visits, revert to best move.
    if(accum == 0.0f || accum_vector.size() < 2) {
        return flags;
    }

    auto distribution = std::uniform_real_distribution<double>{0.0, accum};
    auto pick = distribution(Random::get_Rng());
    const auto index = std::upper_bound( begin(accum_vector),
                                         end(accum_vector),
                                         pick ) - begin(accum_vector);

    assert(m_children.size() > static_cast<size_t>(index));

    flags[FastState::RANDOM] = index != 0;
    flags[FastState::BLUNDER] = blunder_vector[index];

#ifndef NDEBUG
    myprintf("Accum=%f, pick=%f, index=%d.\n", accum, pick, index);
    myprintf("Winrate=%.2f and winrate0=%.2f, move is %s\n",
             100.0f * m_children[index]->get_eval(color),
             100.0f * first_child_eval,
             (blunder_vector[index] ? "blunder" : "ok") );
#endif

    // Take the early out
    if (index != 0) {
        // Now swap the child at index with the first child
        std::iter_swap(begin(m_children), begin(m_children) + index);
    }

    return flags;
}

UCTNode* UCTNode::get_nopass_child(FastState& state) const {
    for (const auto& child : m_children) {
        /* If we prevent the engine from passing, we must bail out when
           we only have unreasonable moves to pick, like filling eyes.
           Note that this knowledge isn't required by the engine,
           we require it because we're overruling its moves. */
        if (child->m_move != FastBoard::PASS
            && !state.board.is_eye(state.get_to_move(), child->m_move)) {
            return child.get();
        }
    }
    return nullptr;
}

// Used to find new root in UCTSearch.
std::unique_ptr<UCTNode> UCTNode::find_child(const int move) {
    for (auto& child : m_children) {
        if (child.get_move() == move) {
             // no guarantee that this is a non-inflated node
            child.inflate();
            return std::unique_ptr<UCTNode>(child.release());
        }
    }

    // Can happen if we resigned or children are not expanded
    return nullptr;
}

void UCTNode::inflate_all_children() {
    for (const auto& node : get_children()) {
        node.inflate();
    }
}

void UCTNode::prepare_root_node(Network & network, int color,
                                std::atomic<int>& nodes,
                                GameState& root_state,
                                bool fast_roll_out) {
    float root_value, root_alpkt, root_beta;

    const auto had_children = has_children();
    if (expandable()) {
        create_children(network, nodes, root_state, root_value, root_alpkt, root_beta);
    }
    if (has_children() && !had_children) {
            // blackevals is useless here because root nodes are never
           // evaluated, nevertheless the number of visits must be updated
            update(0);
    }

    //    root_eval = get_net_eval(color);
    //    root_eval = (color == FastBoard::BLACK ? root_eval : 1.0f - root_eval);

#ifndef NDEBUG
    myprintf("NN eval=%f. Agent eval=%f\n", get_net_eval(color), get_agent_eval(color));
#else
    if (!fast_roll_out) {
        auto x = get_eval_base();
        auto y = get_eval_bonus();
        if (x > y) {
            auto z = x;
            x = y;
            y = z;
        }
        myprintf("NN eval=%f. Agent eval=%f (lambda=%.2f, mu=%.2f, interval [%.1f ; %.1f])\n", get_net_eval(color), get_agent_eval(color), cfg_lambda, cfg_mu, x, y);
    }
#endif

    // There are a lot of special cases where code assumes
    // all children of the root are inflated, so do that.
    inflate_all_children();

    // Remove illegal moves, so the root move list is correct.
    // This also removes a lot of special cases.
    kill_superkos(root_state);

    if (fast_roll_out) {
        return;
    }

    if (cfg_noise) {
        // Adjust the Dirichlet noise's alpha constant to the board size
        auto alpha = cfg_noise_value * 361.0f / NUM_INTERSECTIONS;
        dirichlet_noise(cfg_noise_weight, alpha);
    }

    if (cfg_japanese_mode) {
        for (auto& child : m_children) {
            auto policy = child->get_policy();
            policy *= 0.8f;
            if (child->get_move() == FastBoard::PASS) {
                policy += 0.2f;
            }
            child->set_policy(policy);
        }
    }
}

bool UCTNode::get_children_visits(const GameState& state, const UCTNode& root,
                                  std::vector<float> & probabilities,
                                  bool standardize) {

    probabilities.resize(POTENTIAL_MOVES);

    // Get total visit amount. We count rather
    // than trust the root to avoid ttable issues.
    auto sum_visits = 0.0;
    for (const auto& child : root.get_children()) {
        sum_visits += child->get_visits();
    }
    //   myprintf("Children: %d, Total visits: %f\n", root.get_children().size(),
    //             sum_visits);

    // In a terminal position (with 2 passes), we can have children, but we
    // will not able to accumulate search results on them because every attempt
    // to evaluate will bail immediately. So in this case there will be 0 total
    // visits, and we should not construct the (non-existent) probabilities.
    if (sum_visits <= 0.0) {
        return false;
    }

    // If --recordvisits option is used, then the training data
    // includes the actual number of visits for each move, instead of
    // probabilities. This number can be not integer in case of symmetries.
    if (!standardize) {
        sum_visits = 1.0;
    }

    std::vector<int> stabilizer_subgroup;

    for (auto i = 0; i < 8; i++) {
        if(i == 0 || (cfg_exploit_symmetries && state.is_symmetry_invariant(i))) {
            stabilizer_subgroup.emplace_back(i);
        }
    }

    for (const auto& child : root.get_children()) {
        auto prob = static_cast<float>(child->get_visits() / sum_visits);
        auto move = child->get_move();
        if (move != FastBoard::PASS) {
            const auto frac_prob = prob / stabilizer_subgroup.size();
            for (auto sym : stabilizer_subgroup) {
                const auto sym_move = state.board.get_sym_move(move, sym);
                const auto sym_idx = state.board.get_index(sym_move);
                probabilities[sym_idx] += frac_prob;
                //            myprintf("Vertex: %d, probs %f\n", sym_idx, probabilities[sym_idx]);
            }
        } else {
            probabilities[NUM_INTERSECTIONS] = prob;
        }
    }

    return true;
}
