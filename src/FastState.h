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

#ifndef FASTSTATE_H_INCLUDED
#define FASTSTATE_H_INCLUDED

#include <cstddef>
#include <array>
#include <string>
#include <vector>
#include <bitset>

#include "FullBoard.h"

class FastState {
public:
    enum : char {
        RANDOM, BLUNDER, INTERESTING,
        NUM_FLAGS // must be last
    };

    typedef std::bitset<NUM_FLAGS>  move_flags_t;
  
    void init_game(int size, float komi);
    void reset_game();
    void reset_board();

    bool is_move_legal(int color, int vertex) const;

    void set_komi(float komi);
    void add_komi(float delta);
    float get_komi() const;
    void set_handicap(int hcap);
    int get_handicap() const;
    int get_passes() const;
    int get_to_move() const;
    void set_to_move(int tomove);
    void set_passes(int val);
    void increment_passes();

    float final_score() const;
    std::uint64_t get_symmetry_hash(int symmetry) const;

    size_t get_movenum() const;
    int get_last_move() const;
    void display_state();
    void display_legal(int color);
    std::string move_to_text(int move);

    void set_last_move_flags(const move_flags_t & flags);
    bool is_blunder() const { return m_last_move_flags[BLUNDER]; };
    bool is_random() const { return m_last_move_flags[RANDOM]; };
    bool is_interesting() const { return m_last_move_flags[INTERESTING]; };
    std::string flags_to_text() const { return m_last_move_flags.to_string(); } 

    size_t get_randcount() const { return m_randcount; }
    void inc_randcount() { ++m_randcount; }

    void init_allowed_blunders();
    bool is_blunder_allowed() const;
    int get_allowed_blunders() const;

    bool is_symmetry_invariant(const int symmetry) const;

    void play_move(int vertex);
    void play_move(int color, int vertex);

    FullBoard board;

private:
    float m_komi;
    int m_handicap;
    int m_passes;
    int m_komove;
    size_t m_movenum;
    int m_lastmove;

    // the flags attached to the last chosen move
    move_flags_t m_last_move_flags;

    // number of moves chosen randomly until now
    size_t m_randcount;

    // keeps count of the number of blunders we are
    // still allowed to play; -1 means no limit
    int m_allowed_blunders = -1;

protected:
    void set_movenum(size_t movenum);
};

#endif
