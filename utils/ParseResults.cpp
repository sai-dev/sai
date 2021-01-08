#include <functional>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <map>

using namespace std;

struct sainet {
    string hash;
    long int cumul;
    long int steps;
    long int games;
    int blocks;
    int filters;
    string descr;
    int gener;
    float rating = -1.0f;
    int index = -1;

    bool is_valid() const noexcept { return !hash.empty() && cumul >= 0.0; }
    bool has_rating() const noexcept { return rating >= 0.0f; }

    friend inline bool operator<(const sainet & net1, const sainet & net2) noexcept{
        return net1.cumul < net2.cumul;
    }
};

struct match {
    string hash1;
    string hash2;
    int idx1 = -1;
    int idx2 = -1;
    int h1wins = 0;
    int h2wins = 0;
    int num = 0;

    int jigos() const noexcept { return num - h1wins - h2wins; }

    bool irrelevant() const noexcept {
        return idx1 == -1 || idx2 == -1 || num == 0;
    }

    bool is_valid() const noexcept {
        return !hash1.empty() && !hash2.empty()
            && num >= 0 && h1wins >= 0 && h2wins >= 0
            && num >= h1wins + h2wins;
    }

#if 0
    void flip() noexcept {
        swap(hash1,  hash2);
        swap(idx1,   idx2);
        swap(h1wins, h2wins);
    }
#endif
};

struct net_status{
    int hook_dist = -1;
    unsigned int rank = 0;
    bool always_won = true;
    bool always_lost = true;
};

using nets_t    = vector<sainet>;
using matches_t = vector<match>;
using stats_t   = vector<net_status>;

using tab_t = vector< vector<int> >;


enum filter_mode {
    KEEP_ALL_CONNECTED,
    DROP_ONLY_LEAFS,
    DROP_LEAFS_AND_BEFORE_HOOK
};

namespace {

inline void print_match(const match & m){
    cerr << "Match: ";
    if(min(m.idx1, m.idx2) != -1)
        cerr << "[" << m.idx1 << "-" << m.idx2 << "] ";
    cerr << m.hash1.substr(0,8)
         << " vs "    << m.hash2.substr(0,8)
         << " (" << m.h1wins << "/" << m.jigos() << "/" << m.h2wins << ")\n";
}


enum tags {
    HASH,
    BLOCKS,
    DESCRIPTION,
    FILTERS,
    TRAINING_COUNT,
    TRAINING_STEPS,
    GAME_COUNT,
    NA, // for un-recognised tages [must be last]
};

inline tags tag_type(const string & tag)
{
    static const vector<string> known_tags = {
        "\"hash\"",            // HASH
        "\"blocks\"",          // BLOCKS
        "\"description\"",     // DESCRIPTION
        "\"filters\"",         // FILTERS
        "\"training_count\"",  // TRAINING_COUNT
        "\"training_steps\"",  // TRAINING_STEPS
        "\"game_count\""       // GAME_COUNT
    };

    const vector<string>::const_iterator it =
        find(begin(known_tags), end(known_tags), tag);

    return it == end(known_tags) ? NA : static_cast<tags>( distance(begin(known_tags), it) );
}

void get_first_line(ifstream& data, string& s, const int header_lines = 4) {
    auto j = 0;
    auto position = data.tellg();
    do {
        position = data.tellg();
        getline(data, s);
        ++j;
    } while (s.front() != '{');
    data.seekg(position);
    if (j-1 != header_lines) {
        // Different number of header lines were expected
        cerr << "Skipped " << j-1
             << " header lines. Did something change?" << endl;
    }
}

inline bool quote_opened (const string &s) {
    const auto n = std::count(s.begin(), s.end(), '"');
    return n % 2;
}

inline void complete_quote (ifstream &netsdata, string &s) {
    string tmp;
    while (netsdata >> tmp) {
        s.append(tmp);
        if (!quote_opened(s))
            break;
    }
}

inline int get_index(const vector<sainet> & nets, string hash) {
    for (auto & j : nets) {
        if (j.hash == hash) {
            return j.index;
        }
    }
    return -1;
}

void sort_and_set_indices(vector<sainet> & nets, matches_t & matches) {
    // sort by cumul
    stable_sort(begin(nets), end(nets));

    int i = 0;
    for(auto & net : nets){
        net.index = i++;
    }

    for(auto & match : matches){
        match.idx1 = get_index(nets, match.hash1);
        match.idx2 = get_index(nets, match.hash2);
    }

    // drop matches against unknown or dropped nets
    matches.erase(remove_if(begin(matches), end(matches),
                            mem_fn(&match::irrelevant)),
                  end(matches));
}

#ifndef NDEBUG
bool check_indices(const vector<sainet> & nets) {
    if (!is_sorted(begin(nets), end(nets)))
        return false;

    int i = 0;
    for (auto & net : nets) {
        if( net.index != i++ )
            return false;
    }

    return true;
}
#endif

} // end namespace

nets_t load_netsdata(const string & filename) {
    ifstream netsdata;

    netsdata.open(filename);
    if (!netsdata) {
        cerr << "Unable to open nets data file " << filename
             << "." << endl;
        exit (1);
    } else {
        cerr << "File " + filename + " opened..." << endl;
    }

    string s;
    get_first_line(netsdata, s, 4); // Four header lines are expected
    netsdata >> s;

    nets_t nets;

    auto i = 0;
    string tag;
    sainet tmp;
    while (netsdata >> s) {
        if (quote_opened(s)) {
            complete_quote(netsdata, s);
        }
        if (i%3 == 0) {
            if (s == "{") {
                continue;
            }
            if (s == "}") {
                if (!tmp.is_valid()){
                    cerr << "Found invalid net at row " << nets.size() << "\n";
                    exit (1);
                }

                nets.push_back(move(tmp));
                // reset variables
                tmp = sainet{};
                i = 0;
                continue;
            }
            tag = s;
            ++i;
            continue;
        } else if (i%3 == 1) {
            if (s != ":") {
                cerr << "Nets data file " << filename
                     << " has unexpected format. Fields should be assigned with ':'. Quitting." << endl;
                exit (1);
            }
            ++i;
            continue;
        }
        ++i;
        switch (tag_type(tag)) {
            // i counts the words of the current line
            // string format is              i
            // { "_id" : ObjectId("..."),    0  1  2  3
            // "hash" : "...",               4  5  6
            // "blocks" : 12,                7  8  9
            // "description" : "g05c-1dc3", 10 11 12
            // "filters" : 256,             13 14 15
            // "ip" : "192.167.12.14",      16 17 18
            // "training_count" : ...,      19 20 21
            // "training_steps" : ... }     22 23 24 25
        case HASH:
            // hash
            tmp.hash = s.substr(1, 64);
            tmp.cumul = 0;
            tmp.steps = 0;
            tmp.games = 0;
            tmp.blocks = 0;
            tmp.filters = 0;
            tmp.gener = 0;
            break;
        case BLOCKS:
            // blocks
            if (s.back() == ',') {
                s.pop_back();
            }
            tmp.blocks = stoi(s);
            break;
        case DESCRIPTION:
            // description
            if (s.back() == ',') {
                s.pop_back();
            }
            if (s.back() == '"') {
                s.pop_back();
            }
            if (s.front() == '"') {
                s.erase(s.begin());
            }
            tmp.descr = s;
            if (s.front() == 'g') {
                s.erase(s.begin());
                auto dashpos = s.find("-");
                auto hexgen = s.substr(0, dashpos);
                tmp.gener = stol(hexgen, nullptr, 16);
            }
            else {
                tmp.gener = -1;
            }
        
            break;
        case FILTERS:
            // filters
            if (s.back() == ',') {
                s.pop_back();
            }
            tmp.filters = stoi(s);
            break;
        case TRAINING_COUNT:
            // training_count
            if (s.back() == ',') {
                s.pop_back();
            }
            tmp.cumul = stoi(s);
            break;
        case TRAINING_STEPS:
            // training_steps
            if (s.back() == ',') {
                s.pop_back();
            }
            if (s == "null") {
                tmp.steps = 0;
            } else {
                tmp.steps = stoi(s);
            }
            break;
        case GAME_COUNT:
            // game_count
            if (s.back() == ',') {
                s.pop_back();
            }
            if (s == "null") {
                tmp.games = 0;
            } else {
                tmp.games = stoi(s);
            }
            break;
        case NA:
        default:
            break;
        }
    } 

    netsdata.close();

    cerr << "Nets found: " << nets.size() << "\n";

    return nets;
}

matches_t load_matchdata(string filename) {
    ifstream matchdata;

    matchdata.open(filename);
    if (!matchdata) {
        cerr << "Unable to open matchdata file " << filename
             << "." << endl;
        exit (1);
    }

    cerr << "File " + filename + " opened...\n";

    string s;
    get_first_line(matchdata, s, 4); // Four header lines are expected

    matches_t matches;
    auto i = 0;
    match tmp;
    while (matchdata >> s) {
        switch (i) {
        case 3:
            // network1
            tmp.hash1 = s.substr(1, 64);
            break;
        case 6:
            // network2
            tmp.hash2 = s.substr(1, 64);
            break;
        case 9:
            // network1_losses
            if (s.back() == ',') {
                s.pop_back();
            }
            tmp.h2wins = stoi(s);
            break;
        case 12:
            // network1_wins
            if (s.back() == ',') {
                s.pop_back();
            }
            tmp.h1wins = stoi(s);
            break;
        case 15:
            // game_count
            if (s.back() == ',') {
                s.pop_back();
            }
            tmp.num = stoi(s);
            break;
        }
        ++i;
        if (s == "}") {
            if (!tmp.is_valid()){
                cerr << "Found invalid match at row " << matches.size() << "\n";
                exit (1);
            }

            matches.push_back(move(tmp));
            // reset variables
            tmp = match{};
            i = 0;
        }
    } 
    matchdata.close();

    cerr << "Matches found: " << matches.size() << "\n";

    return matches;
}


stats_t get_stats(const matches_t & matches, size_t n, size_t hook_index){
    assert(hook_index < n);

    stats_t stats(n);

    // first pass to compute node ranks...
    // ...and mark the always won/lost
    for (auto & match : matches) {
        auto & stat1 = stats[match.idx1];
        auto & stat2 = stats[match.idx2];

        ++stat1.rank;
        ++stat2.rank;

        if(match.h1wins != match.num)
            stat1.always_won = stat2.always_lost = false;

        if(match.h2wins != match.num)
            stat2.always_won = stat1.always_lost = false;
    }

    // mark the hook net
    stats[hook_index].hook_dist = 0;

    int dist = 0;
    bool modified;
    do {
        // cerr << "Loop: " << dist << "\n";
        modified = false;
        for (auto & match : matches) {
            auto & stat1 = stats[match.idx1];
            auto & stat2 = stats[match.idx2];

            if (stat1.hook_dist == -1 && stat2.hook_dist == dist){
                stat1.hook_dist = 1 + dist;
                modified = true;
            }

            if (stat2.hook_dist == -1 && stat1.hook_dist == dist){
                stat2.hook_dist = 1 + dist;
                modified = true;
            }
        }
        ++dist;
    } while (modified);

    return stats;
}


void create_connected_graph_from_hook(nets_t & nets, matches_t & matches, const string & hook, filter_mode fm){
    assert(( (void)"Indices must be set", check_indices(nets) ));

    const size_t n = nets.size();

    const auto hook_index = get_index(nets, hook);
    if (hook_index == -1) {
        cerr << "Unable to find hook hash " << hook
             << " in nets file." << endl;
        exit (1);
    }

    // cerr << "Hook index: " << hook_index << "\n";

    if (fm == DROP_LEAFS_AND_BEFORE_HOOK && hook_index > 0){
        // drop matches before hook
        const auto drop = [hook_index](const match & m){
            return max(m.idx1, m.idx2) < hook_index;
        };

        matches.erase(remove_if(begin(matches), end(matches), drop),
                      end(matches));
    }

    // get the stats
    const stats_t stats = get_stats(matches, n, hook_index);

    // small utility library od lambdas
    const auto is_hook = [&hook_index](size_t idx){
        return idx == size_t(hook_index);
    };

    const auto is_connected = [&stats](size_t idx){
        return stats[idx].hook_dist > 0;
    };

    const auto is_leaf = [&is_connected,&stats](size_t idx){
        return is_connected(idx) && stats[idx].rank <= 1;
    };

    const auto one_way = [&is_connected, &stats](size_t idx){
        return is_connected(idx) && (stats[idx].always_lost || stats[idx].always_won);
    };

    const auto keep =
        [fm,&is_hook,&is_connected,&is_leaf,&one_way](size_t idx){
        if (is_hook(idx))             return true;
        if (!is_connected(idx))       return false;
        if (fm == KEEP_ALL_CONNECTED) return true;
        return !(is_leaf(idx) || one_way(idx));
    };

    // this lambda gets a predicate on a index
    // and returns another lambda: a predicate on a net
    const auto wrapper = [](const auto & pred){
        return [&pred](const sainet & net){
            return pred(net.index);
        };
    };

    // drop unneded nets
    const nets_t::iterator first_to_prune =
              stable_partition(begin(nets), end(nets), wrapper(keep));

    cerr << "Dropping " << distance(first_to_prune, end(nets)) << " nets";

    if (fm != KEEP_ALL_CONNECTED){
        const nets_t::iterator first_non_leaf =
            stable_partition(first_to_prune, end(nets), wrapper(is_leaf));

        const size_t num_leaf = distance(first_to_prune,first_non_leaf);
        const size_t num_one_way = count_if(first_non_leaf, end(nets),
                                            wrapper(one_way));

        cerr << " of which " << num_leaf    << " are leafs"
             << " and "      << num_one_way << " are one way";
    }
    cerr << "\n";

    nets.erase(first_to_prune, end(nets));

    sort_and_set_indices(nets, matches);
}

float delta_rating(float prior_elo_std, float wins, float jigos, float losses) {
    assert(( (void)"Invalid match", (wins >= 0.0f && jigos >= 0.0f && losses >= 0.0f) ));

    constexpr float ELO_FACTOR = 400.0f / log(10.0f);

    const float num = wins + jigos + losses;

    if (num == 0.0f)
        return 0.0f;

    const float T = powf(prior_elo_std / ELO_FACTOR, 2.0f);

    const float score = 0.5f + 0.5f * (wins - losses)
        / (num + 4.0f * (wins + losses) / num / T );

    return ELO_FACTOR * (log(score) - log(1.0f-score));
}

void load_existing_ratings(string filename, nets_t & nets)
{
    assert(( (void)"Indices must be set", check_indices(nets) ));

    ifstream ratingsfile;

    ratingsfile.open(filename);
    if (!ratingsfile){
        cerr << "No previous ratings file " << filename << " found." << endl;
        exit(1);
    }

    cerr << "File " << filename << " opened...\n";

    size_t count = 0;
    string s;
    while (ratingsfile >> s) {
        const auto hash = s.substr(0,64);
        auto commapos = s.find(',');
        for (auto i=0 ; i<7 && commapos!=string::npos ; ++i) {
            commapos = s.find(',', commapos + 1);
        }
        if (commapos == string::npos) {
            cerr << "Ratings file: no rating found for net "
                 << hash << endl;
            exit(1);
        }

        const float rating = stof(s.substr(commapos+1));
        ++count;

        auto index = get_index(nets, hash);
        if( index == -1 )
            continue;

        nets[index].rating = rating;
    }

    ratingsfile.close();

    cerr << "Found " << count
         << " nets with previous rating values\n";
}

void rate_connected_nets(nets_t & nets, const matches_t & matches, float prior_elo_std) {
    assert(( (void)"Indices must be set", check_indices(nets) ));

    // get a rough estimate of the missing ratings
    for (auto & net : nets) {

        if (net.has_rating())
            continue;

        float sum_ratings = 0.0f;
        size_t sum_games = 0;

        const auto update_sums = [&sum_ratings, &sum_games, &prior_elo_std]
            (const match & m, const sainet & opp, float sign){

            if( !opp.has_rating() )
                return;

            const float delta = delta_rating(prior_elo_std,
                                             m.h1wins, m.jigos(), m.h2wins);

            sum_ratings += float(m.num) * max(0.0f, opp.rating + sign * delta);
            sum_games += m.num;
        };

        for (auto & match : matches) {
            if (net.index == match.idx1)
                update_sums(match, nets[match.idx2],  1.0f);
            else if(net.index == match.idx2)
                update_sums(match, nets[match.idx1], -1.0f);
        }

        if (sum_games == 0) {
            cerr << "Something's wrong. Net " << net.hash.substr(0,8)
                 << " has no rating, and no connections were found." << endl;
            exit(1);
        }

        net.rating = sum_ratings / float(sum_games);
    }
}

#if 0
void random_init(vector<double> & vec) {
    random_device rd;
    mt19937 gen(rd());
    auto unif_law = uniform_real_distribution<double>{0.0, 10.0};
    for (auto & x : vec) {
        x = unif_law(gen);
    }
}

double log_likely(const vector<double> & r, tab_t & table) {
    const auto n = r.size();

    auto l = 0.0;

    for (size_t i=0 ; i < n ; ++i) {
        for (size_t j=0 ; j < n ; ++j) {
            const auto tmp = r[i] - log( exp(r[i]) + exp(r[j]) );
            l += table[i][j] * tmp;
        }
    }
    return l;
}


double gradient_desc(vector<double> & r,
                     double h, tab_t & table, double & der_la) {
    const auto n = r.size();

    vector<double> s(n);

    for (size_t i=0 ; i < n ; i++) {
        s[i] = r[i];
    }
    
    vector<double> grad(n);
    for (size_t i=0 ; i < n ; i++) {
        grad[i] = 0.0;
        for (size_t j=0 ; j < n ; j++) {
            const auto num = exp(r[j]) * table[i][j]
                - exp(r[i]) * table[j][i];
            const auto den = exp(r[i]) + exp(r[j]);
            grad[i] += num / den;
        }
    }

    auto grad_norm = 0.0;
    for (size_t i=0 ; i < n ; i++) {
        grad_norm += grad[i] * grad[i];
    }
    grad_norm = pow(grad_norm, 0.25);

    for (size_t i=0 ; i < n ; i++) {
        r[i] += 0.5 * h * grad[i] / grad_norm;
    }

    for (size_t i=0 ; i < n ; i++) {
        grad[i] = 0.0;
        for (size_t j=0 ; j < n ; j++) {
            const auto num = exp(r[j]) * table[i][j]
                - exp(r[i]) * table[j][i];
            const auto den = exp(r[i]) + exp(r[j]);
            grad[i] += num / den;
        }
    }

    grad_norm = 0.0;
    for (size_t i=0 ; i < n ; i++) {
        grad_norm += grad[i] * grad[i];
    }
    grad_norm = pow(grad_norm, 0.25);

    auto min = 10.0;
    for (size_t i=0 ; i < n ; i++) {
        r[i] = s[i] + h * grad[i] / grad_norm;
        if (r[i] < min) {
            min = r[i];
        }
    }
    for (size_t i=0 ; i < n ; i++) {
        r[i] -= min;
    }

    der_la = 0.0;
    for (size_t i=0 ; i < n ; i++) {
        for (size_t j=0 ; j < n ; j++) {
            der_la += table[i][j] * (r[i] - r[j])
                * exp(r[j]) / (exp(r[i]) + exp(r[j]));
        }
    }
    for (size_t i=0 ; i < n ; i++) {
        r[i] *= exp( std::max(std::min(h * der_la / grad_norm, 0.1 * h), -0.1 * h) );
    }

    auto norm = 0.0;
    for (size_t i=0 ; i < n ; i++) {
        auto tmp = r[i] - s[i];
        norm += tmp * tmp;
    }    
    return sqrt(norm);
}

void fit_ratings(vector< vector<int> > & table){
    vector<double> rats(n);
    random_init(rats);

    auto l = log_likely(rats, table);
    cerr << l << endl;
    double delta;
    auto h = 0.1;
    auto it = 1.0;
    do {
        it *= 1.1;
        auto der_la = 0.0;
        delta = gradient_desc(rats, h, table, der_la);
        l = log_likely(rats, table);
        cerr << h << ", "
             << delta/h << ", "
             << der_la << ", "
             << l << ", "
             << rats[n-1] << endl;
        if (delta/h < 100.0 * it * h) {
            h *= 0.995;
        } else {
            h *= 1.002;
        }
    } while (delta > .01);
    cerr << log_likely(rats, table);
    for (auto & r : rats) {
        cerr << r << "\n";
    }
}
#endif

auto populate_tables(const nets_t & nets, const matches_t & matches) {
    assert(( (void)"Indices must be set", check_indices(nets) ));

    const size_t n = nets.size();

    std::pair<tab_t, tab_t> tables;
    get<0>(tables).resize(n, tab_t::value_type(n));
    get<1>(tables) = get<0>(tables);

    for (auto & match : matches) {
        get<0>(tables)[match.idx1][match.idx2] = match.h1wins;
        get<0>(tables)[match.idx2][match.idx1] = match.h2wins;
        get<1>(tables)[match.idx1][match.idx2] = match.num;
        get<1>(tables)[match.idx2][match.idx1] = match.num;
    }

    for (auto & num : get<1>(tables)){
        if (search_n(begin(num),end(num),1,0,not_equal_to<>{}) == end(num)){
            cerr << "Num table contains rows with only zeros" << endl;
            exit(1);
        }
    }

    return tables;
}

void write_table(const string & filename, const tab_t & table) {
    ofstream tabledump;

    tabledump.open(filename);
    if (!tabledump) {
        cerr << "Unable to open file " << filename
             << " to dump table." << endl;
        exit (1);
    }

    cerr << "Writing '" << filename << "' file...\n";

    const auto n = table.size();

    tabledump << n << "\n";
    
    for (size_t i=0 ; i < n ; ++i) {
        assert(( (void)"Table must be a rectangle", table[i].size() == n ));

        for (size_t j=0 ; j < n ; ++j) {
            if (j>0) {
                tabledump << ",";
            }
            tabledump << table[i][j];
        }
        tabledump << "\n";
    }

    tabledump.close();
}

void write_netlist(const string & filename, const nets_t & nets) {
    ofstream netlistdump;

    netlistdump.open(filename);
    if (!netlistdump) {
        cerr << "Unable to open file " << filename
             << " to dump netlist." << endl;
        exit (1);
    }

    cerr << "Writing '" << filename << "' file...\n";

    for (auto & net : nets) {
        netlistdump << net.hash
                    << "," << net.descr
                    << "," << net.blocks
                    << "," << net.filters
                    << "," << net.cumul
                    << "," << net.steps
                    << "," << net.games
                    << "," << net.gener
                    << "," << net.rating
                    << "\n";
    }

    netlistdump.close();
}


int main(int argc, char* argv[]) {
    if (argc <= 3) {
        cerr << "Syntax: pseres <saiXX> <sha256hash> <stdev> [-p]" << endl
             << "net hash is hook/root with Elo fixed to 0" << endl
             << "standard deviation [in Elo points] is for the prior rating difference"
             << "  -p        prune leaf nodes" << endl;
        exit (1);
    }

    string saiXX(argv[1]);
    string hook_net_hash(argv[2]);

    const float prior_elo_std = stof(string(argv[3]));

    const bool prune = argc > 4 && string(argv[4]) == "-p";

    nets_t nets = load_netsdata(saiXX + "-netdata.xls");

    matches_t matches = load_matchdata(saiXX + "-matches.xls");

    sort_and_set_indices(nets, matches);

    create_connected_graph_from_hook(nets, matches,
                                     hook_net_hash, KEEP_ALL_CONNECTED);

    cerr << "Nets linked to hook: " << nets.size() << "\n";
    cerr << "Hook position: " << get_index(nets, hook_net_hash) << "\n";

    load_existing_ratings(saiXX + "-ratings.csv", nets);
    rate_connected_nets(nets, matches, prior_elo_std);

    write_netlist(saiXX + "-rated-nets.csv", nets);

    if (prune){
        cerr << "Pruning leaf nodes..." << endl;
    
        create_connected_graph_from_hook(nets, matches,
                                         hook_net_hash, DROP_ONLY_LEAFS);

        cerr << "Nets linked to hook after pruning leafs: " << nets.size() << "\n";
        cerr << "Relevant matches after pruning leafs: " << matches.size() << "\n";
        cerr << "Hook position after pruning leafs: " << get_index(nets, hook_net_hash) << "\n";

        write_netlist(saiXX + "-nets.csv", nets);

        cerr << "Deep pruning uninteresting nodes..." << endl;

        create_connected_graph_from_hook(nets, matches,
                                         hook_net_hash, DROP_LEAFS_AND_BEFORE_HOOK);

        const auto hook_pos = get_index(nets, hook_net_hash);
        cerr << "Nets linked to hook after deep pruning: " << nets.size() << "\n";
        cerr << "Relevant matches after deep pruning: " << matches.size() << "\n";
        cerr << "Hook position after deep pruning: " << hook_pos << "\n";
        cout << hook_pos << endl;

        const auto tables = populate_tables(nets,matches);

        write_netlist(saiXX + "-pruned-nets.csv", nets);
        write_table(saiXX + "-vs.csv",  get<0>(tables));
        write_table(saiXX + "-num.csv", get<1>(tables));
    }

    cerr << "Done\n";

    return 0;
}
