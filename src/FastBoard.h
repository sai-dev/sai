/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2018 Gian-Carlo Pascutto and contributors

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

#ifndef FASTBOARD_H_INCLUDED
#define FASTBOARD_H_INCLUDED

#include "config.h"

#include <array>
#include <queue>
#include <string>
#include <utility>
#include <vector>

class FastBoard {
    friend class FastState;
public:
    /*
        neighbor counts are up to 4, so 3 bits is ok,
        but a power of 2 makes things a bit faster
    */
    static constexpr int NBR_SHIFT = 4;
    static constexpr int NBR_MASK = (1 << NBR_SHIFT) - 1;

    /*
        number of vertices in a "letterboxed" board representation
    */
    static constexpr int NUM_VERTICES = ((BOARD_SIZE + 2) * (BOARD_SIZE + 2));

    /*
        no applicable vertex
    */
    static constexpr int NO_VERTEX = 0;
    /*
        vertex of a pass
    */
    static constexpr int PASS   = -1;
    /*
        vertex of a "resign move"
    */
    static constexpr int RESIGN = -2;

    /*
        possible contents of a vertex
    */
    enum vertex_t : char {
        BLACK = 0, WHITE = 1, EMPTY = 2, INVAL = 3
    };

    enum territory_t : char {
        B_STONE = 0, W_STONE = 1, EMPTY_I = 2, INVAL_I = 3,
        DAME = 4, SEKI = 5, SEKI_EYE = 6,
        W_TERR = 7, B_TERR = 8
    };


    int get_boardsize() const;
    vertex_t get_state(int x, int y) const;
    vertex_t get_state(int vertex) const ;
    int get_vertex(int x, int y) const;
    int get_vertex(const int index) const;
    int get_index(const int vertex) const;
    void set_state(int x, int y, vertex_t content);
    void set_state(int vertex, vertex_t content);
    std::pair<int, int> get_xy(int vertex) const;

    bool is_suicide(int i, int color) const;
    int count_pliberties(const int i) const;
    bool is_eye(const int color, const int vtx) const;

    float area_score(float komi) const;
    float territory_score(float komi);

    int get_prisoners(int side) const;
    bool black_to_move() const;
    bool white_to_move() const;
    int get_to_move() const;
    void set_to_move(int color);

    std::string move_to_text(int move) const;
    int text_to_move(std::string move) const;
    std::string move_to_text_sgf(int move) const;
    std::string get_stone_list() const;
    std::string get_string(int vertex) const;

    void reset_board(int size);
    void display_board(int lastmove = -1) const;

    static bool starpoint(int size, int point);
    static bool starpoint(int size, int x, int y);

    int liberties_to_capture(int vtx) const;
    int get_sym_move(const int vertex, const int symmetry) const;

    void find_dame(std::vector<int>& all_dames);
    void reset_territory();
    bool is_dame(int vertex) const;

protected:
    /*
        bit masks to detect eyes on neighbors
    */
    static const std::array<int,      2> s_eyemask;
    static const std::array<vertex_t, 4> s_cinvert; /* color inversion */

    std::array<vertex_t, NUM_VERTICES>         m_state;      /* board contents */
    std::array<unsigned short, NUM_VERTICES+1> m_next;       /* next stone in string */
    std::array<unsigned short, NUM_VERTICES+1> m_parent;     /* parent node of string */
    std::array<unsigned short, NUM_VERTICES+1> m_libs;       /* liberties per string parent */
    std::array<unsigned short, NUM_VERTICES+1> m_stones;     /* stones per string parent */
    std::array<unsigned short, NUM_VERTICES>   m_neighbours; /* counts of neighboring stones */
    std::array<int, 4>                         m_dirs;       /* movement directions 4 way */
    std::array<int, 2>                         m_prisoners;  /* prisoners per color */
    std::array<unsigned short, NUM_VERTICES>   m_empty;      /* empty intersections */
    std::array<unsigned short, NUM_VERTICES>   m_empty_idx;  /* intersection indices */
    int m_empty_cnt;                                         /* count of empties */

    int m_tomove;
    int m_numvertices;

    int m_boardsize;
    int m_sidevertices;

    std::array<territory_t, NUM_VERTICES> m_territory;

    int calc_reach_color(int color, int color_spread,
                         std::vector<bool> & bd, bool territory) const;
    int calc_reach_color(int color) const;
    void find_dame();
    void find_seki();
    std::pair<int,int> find_territory();
    std::pair<int,int> compute_territory();

    int count_neighbours(const int color, const int i) const;
    void merge_strings(const int ip, const int aip);
    void add_neighbour(const int i, const int color);
    void remove_neighbour(const int i, const int color);
    void print_columns() const;
};

#endif
