#include<algorithm>
#include<cmath>
#include<fstream>
#include<iostream>
#include<map>
#include<random>
#include<string>
#include<vector>

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
    int index = 0;
    int hookdist = -1;
    int rank = 0;
    bool won_once = false;
    bool lost_once = false;
};

struct match {
    string hash1;
    string hash2;
    int h1wins;
    int h2wins;
    int n;
};

struct wincount {
    int idx1;
    int idx2;
    int wins;
    int num;
};

enum tags {
    NA = 0,  // Zero is 'na': if a new tag is found, accessing the map
             // inserts a new pair whose value is initialized to 0
    HASH,
    BLOCKS,
    DESCRIPTION,
    FILTERS,
    TRAINING_COUNT,
    TRAINING_STEPS,
    GAME_COUNT };

static std::map<std::string, tags> tag_v =
    {
        { "\"hash\"", HASH },
        { "\"blocks\"", BLOCKS },
        { "\"description\"", DESCRIPTION },
        { "\"filters\"", FILTERS },
        { "\"training_count\"", TRAINING_COUNT },
        { "\"training_steps\"", TRAINING_STEPS },
        { "\"game_count\"", GAME_COUNT }
    };

typedef vector< vector<int> > tab_t;

vector<sainet> nets;
vector<match> mats;
vector<wincount> wins;


void get_first_line(ifstream& data, string& s, const int header_lines = 4) {
    auto j = 0;
    auto position = data.tellg();
    do {
        position = data.tellg();
        getline(data, s);
        j++;
    } while (s.front() != '{');
    data.seekg(position);
    if (j-1 != header_lines) {
        // Different number of header lines were expected
        cout << "Skipped " << j-1
             << " header lines. Did something change?" << endl;
    }
}


bool quote_opened (const string &s) {
    const auto n = std::count(s.begin(), s.end(), '"');
    return n % 2;
}

void complete_quote (ifstream &netsdata, string &s) {
    string tmp;

    while (netsdata >> tmp) {
	s.append(tmp);
	if (!quote_opened(s))
	    break;
    }
}

void load_netsdata(string filename, string hook) {
    ifstream netsdata;

    netsdata.open(filename);
    if (!netsdata) {
        cerr << "Unable to open nets data file " << filename
             << "." << endl;
        exit (1);
    } else {
	cerr << "File " + filename + " opened." << endl;
    }

    string s;
    get_first_line(netsdata, s, 4); // Four header lines are expected
    netsdata >> s;

    bool hookfound = false;
    auto i = 0;
    string tag;
    sainet tmp;
    while (netsdata >> s) {
        //        cout << i << " : " << s << endl;
	if (quote_opened(s)) {
	    complete_quote(netsdata, s);
	}
        if (i%3 == 0) {
            if (s == "{") {
                continue;
            }
            if (s == "}") {
                // cout << tmp.index << ". "
                //           << tmp.hash << ", "
                //           << tmp.cumul << ", "
                //           << tmp.steps << ", "
                //           << tmp.games << endl;
                nets.emplace_back(tmp);
                tmp.index++;
                i = 0;
                continue;
            }
            tag = s;
            i++;
            continue;
        } else if (i%3 == 1) {
            if (s != ":") {
                cerr << "Nets data file " << filename
                     << " has unexpected format. Fields should be assigned with ':'. Quitting." << endl;
                exit (1);
            }
            i++;
            continue;
        }
        //        cout << s << endl;
        i++;
        //        cout << "tag: " << tag << ", val: " << tag_v[tag] << endl;
        switch (tag_v[tag]) {
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
        case NA:
            break;
        case HASH:
            // hash
            tmp.hash = s.substr(1, 64);
            tmp.cumul = 0;
            tmp.steps = 0;
            tmp.games = 0;
            tmp.blocks = 0;
            tmp.filters = 0;
            tmp.gener = 0;
            tmp.rank = 0;
            if (tmp.hash == hook) {
                tmp.hookdist = 0;
                hookfound = true;
            } else {
                tmp.hookdist = -1;
            }
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
        }
    } 

    netsdata.close();
    if (!hookfound) {
        cerr << "Unable to find hook hash " << hook
             << " in nets file." << endl;
        exit (1);
    }
}


void load_matchdata(string filename) {
    ifstream matchdata;

    matchdata.open(filename);
    if (!matchdata) {
        cerr << "Unable to open matchdata file " << filename
             << "." << endl;
        exit (1);
    } else {
	cerr << "File " + filename + " opened." << endl;
    }

    string s;
    get_first_line(matchdata, s, 4); // Four header lines are expected

    auto i = 0;
    match tmp;
    while (matchdata >> s) {
        switch (i) {
        case 3:
            // network1
            tmp.hash1 = s.substr(1, 64);
            tmp.hash2 = "";
            tmp.h1wins = 0;
            tmp.h2wins = 0;
            tmp.n = 0;
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
            tmp.n = stoi(s);
            break;
        }
        i++;
        if (s == "}") {
            // cout << tmp.hash1.substr(0,8) << ", "
            //           << tmp.hash2.substr(0,8) << ", "
            //           << tmp.h1wins << ", "
            //           << tmp.h2wins << endl;
            mats.emplace_back(tmp);
            i = 0;
        }
    } 
    matchdata.close();
}

int index(string hash) {
    for (auto & j : nets) {
        if (j.hash == hash) {
            return j.index;
        }
    }
    return -1;
}


void pt(const wincount & w) {
    cout << w.idx1 << ", "
              << w.idx2 << ", "
              << w.wins << endl;
}

void list_wins() {
    for (auto & vs : mats) {
        wincount ij, ji;
        ij.idx1 = ji.idx2 = index(vs.hash1);
        ij.idx2 = ji.idx1 = index(vs.hash2);
        if (ij.idx1 == -1 || ij.idx2 == -1) {
            continue;
        }
        ij.wins = vs.h1wins;
        ji.wins = vs.h2wins;
        ij.num = ji.num = vs.n;
        wins.emplace_back(ij);
        wins.emplace_back(ji);
        //        pt(ij);
        //        pt(ji);
    }
}

void check_indices() {
    for (auto i = 0 ; i < int(nets.size()) ; i++) {
        if (i != nets[i].index) {
            cerr << "Wrong index in nets table: "
                      << i << ", " << nets[i].index << endl;
            exit (1);
        }
    }
}


int connect_graph() {
    int dist = 0;
    int cnt_nets = 0;
    bool modified;
    do {
        modified = false;
        for (auto & edge : wins) {
            auto & net1 = nets[edge.idx1];
            auto & net2 = nets[edge.idx2];

            if (dist == 0) {
                // on the first pass, compute node ranks...
                net1.rank++;
                net2.rank++;
                // (rank is twice the number of node's edges)
                // ...and mark the first wins and first losses
                if (edge.wins > 0) {
                    net1.won_once = true;
                    net2.lost_once = true;
                }
            }

            if (net1.hookdist >= 0)
                continue;
            if (net2.hookdist == dist) {
                net1.hookdist = 1 + dist;
                modified = true;
                cnt_nets++;
                //                cout << ".";
            }
        }
        dist++;
        //        cout << dist << endl;
    } while (modified);
    return cnt_nets;
}

void remove_unconnected_nets(bool prune) {
    for (auto itnet = nets.begin() ; itnet != nets.end() ; ) {
        auto & net = *itnet;
        if (net.hookdist == -1 ||
            (prune && net.hookdist > 0 && net.rank <= 2) ) {
            // if the node is disconnected from hook
            // or if pruning is enabled and the node is a leaf
            // different from hook
            // remove the node
            itnet = nets.erase(itnet);
        } else if (prune && !net.won_once) {
            cout << "Net " << net.hash.substr(0,8)
                 << " never won. Pruning." << endl;
            if (net.hookdist == 0) {
                cerr << "Error: pruning hook not allowed. Quitting." << endl;
                exit(1);
            }
            itnet = nets.erase(itnet);
        } else if (prune && !net.lost_once) {
            cout << "Net " << net.hash.substr(0,8)
                 << " never lost. Pruning." << endl;
            if (net.hookdist == 0) {
                cerr << "Error: pruning hook not allowed. Quitting." << endl;
                exit(1);
            }
            itnet = nets.erase(itnet);
        } else {
            itnet++;
        }
    }
    auto i = 0;
    for (auto & net : nets) {
        net.index = i++;
    }

    check_indices();
    wins.clear();
    list_wins();
}


float delta_rating(unsigned int wins, unsigned int num, unsigned int losses = 0) {
    // e^-0.5 quantile corresponding asymptotically to 1 draw and n-1 losses
    constexpr float C = exp(-0.5f);
    constexpr float ELO_FACTOR = 400.0f / log(10.0f);

    if (wins + losses > num) {
        cerr << "Something's wrong: wins + losses > num.   " << wins
             << " + " << losses << " > " << num << endl;
        exit(1);
    }

    const auto draws = num - wins - losses;
    const auto bound = pow(C, 1.0f/num);
    auto score = float((1.0f * wins + 0.5f * draws) / num);
    if (score > bound)
        score = bound;
    else if (score < 1-bound)
        score = 1-bound;
    
    return ELO_FACTOR * (log(score) - log(1.0f-score));
}


bool rate_connected_nets(string filename) {
    ifstream ratingsfile;

    ratingsfile.open(filename);
    if (!ratingsfile) {
        cerr << "No previous ratings file found." << endl;
        return false;
    }
    cerr << "File " + filename + " opened." << endl;        

    std::map<std::string, float> net_rats;
    string s;

    while (ratingsfile >> s) {
        const auto hash = s.substr(0,64);
        auto commapos = s.find(',');
        for (auto i=0 ; i<7 && commapos!=string::npos ; i++) {
            commapos = s.find(',', commapos + 1);
        }
        if (commapos == string::npos) {
            cerr << "Ratings file: no rating found for net "
                 << hash << endl;
            exit(1);
        }
        const auto rating = float(stof(s.substr(commapos+1)));
        net_rats[hash] = rating;
    }
    cerr << "Found " << net_rats.size()
         << " nets with previous rating values." << endl;
    ratingsfile.close();

    for (auto & net : nets) {
        if (net_rats.count(net.hash)) {
            net.rating = net_rats[net.hash];
        }
    }
    
    for (auto & net : nets) {
        if (!net_rats.count(net.hash) && net.rank > 2 && !net.won_once) {
            auto worst_rate = 0.0f;
            auto found = false;

            for (auto & edge : wins) {
                auto & net1 = nets[edge.idx1];
                auto & net2 = nets[edge.idx2];
                if (net.index != net2.index)
                    continue;
                if (!net_rats.count(net1.hash))
                    continue;
                const auto estim_rate = net_rats[net1.hash]
                    - delta_rating(edge.wins, edge.num);
                if (!found || estim_rate < worst_rate) {
                    worst_rate = estim_rate;
                    found = true;
                }
            }
            if (!found) {
                cerr << "Something's wrong. Net " << net.hash.substr(0,8)
                     << " has no rating, at least two connections"
                    " with no wins and no rating." << endl;
                exit(1);
            }
            net.rating = worst_rate;
        }
    }

    for (auto & net : nets) {
        if (!net_rats.count(net.hash) && net.rank > 2 && !net.lost_once) {
            auto best_rate = 0.0f;
            auto found = false;

            for (auto & edge : wins) {
                auto & net1 = nets[edge.idx1];
                auto & net2 = nets[edge.idx2];
                if (net.index != net1.index)
                    continue;
                if (!net_rats.count(net2.hash))
                    continue;
                const auto estim_rate = net_rats[net2.hash]
                    + delta_rating(edge.wins, edge.num);
                if (!found || estim_rate > best_rate) {
                    best_rate = estim_rate;
                    found = true;
                }
            }
            if (!found) {
                cerr << "Something's wrong. Net " << net.hash.substr(0,8)
                     << " has no rating, at least two connections"
                    " with no losses and no rating." << endl;
                exit(1);
            }
            net.rating = best_rate;
        }
    }

    for (auto & net : nets) {
        if (!net_rats.count(net.hash) && net.rank <= 2) {
            auto opponent = net;
            unsigned int wons, nums, losses;

            for (auto & edge : wins) {
                auto & net1 = nets[edge.idx1];
                auto & net2 = nets[edge.idx2];
                if (net.index == net1.index) {
                    if (opponent.index == net.index) {
                        opponent = net2;
                    } else if (opponent.index != net2.index) {
                        cerr << "Something's wrong. Net " << net.hash.substr(0,8)
                             << " matches corrupted. net2: " << net2.hash.substr(0,8)
                             << " and " << opponent.hash.substr(0,8) << endl;
                        exit(1);
                    }
                    wons = edge.wins;
                    nums = edge.num;
                } else if (net.index == net2.index) {
                    if (opponent.index == net.index) {
                        opponent = net1;
                    } else if (opponent.index != net1.index) {
                        cerr << "Something's wrong. Net " << net.hash.substr(0,8)
                             << " matches corrupted. net1: " << net1.hash.substr(0,8)
                             << " and " << opponent.hash.substr(0,8) << endl;
                        exit(1);
                    }
                    losses = edge.wins;
                }
            }

            if (opponent.index == net.index) {
                cerr << "Something's wrong. Net " << net.hash.substr(0,8)
                     << " has no rating, but no connections were found." << endl;
                exit(1);
            }

            if (opponent.rating < 0.0f) {
                cerr << "Something's wrong. Net " << net.hash.substr(0,8)
                     << " and its only connection " << opponent.hash.substr(0,8)
                     << " have no rating: " << opponent.rating << endl;
                exit(1);
            }

            net.rating = opponent.rating + delta_rating(wons, nums, losses);
        }
    }

    return true;
}

void populate_table(tab_t & table) {
    for (auto & vs : wins) {
        table[vs.idx1][vs.idx2] = vs.wins;
    }
}


void populate_tables(tab_t & table, tab_t & table_num) {
    for (auto & vs : wins) {
        table[vs.idx1][vs.idx2] = vs.wins;
        table_num[vs.idx1][vs.idx2] = vs.num;
    }
}


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

    for (size_t i=0 ; i < n ; i++) {
        for (size_t j=0 ; j < n ; j++) {
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

void write_table(tab_t & table, string filename) {
    ofstream tabledump;

    tabledump.open(filename);
    if (!tabledump) {
        cerr << "Unable to open file " << filename
             << " to dump table." << endl;
        exit (1);
    }

    const auto n = table.size();

    tabledump << n << endl;
    
    for (size_t i=0 ; i < n ; i++) {
        for (size_t j=0 ; j < n ; j++) {
            if (j>0) {
                tabledump << ",";
            }
            tabledump << table[i][j];
        }
        tabledump << endl;
    }

    tabledump.close();
}


void write_netlist(string filename, bool rated=false) {
    ofstream netlistdump;

    netlistdump.open(filename);
    if (!netlistdump) {
        cerr << "Unable to open file " << filename
             << " to dump netlist." << endl;
        exit (1);
    }

    const auto n = nets.size();
    for (size_t i=0 ; i < n ; i++) {
        netlistdump << nets[i].hash << ","
                    << nets[i].descr << ","
                    << nets[i].blocks << ","
                    << nets[i].filters << ","
                    << nets[i].cumul << ","
                    << nets[i].steps << ","
                    << nets[i].games << ","
                    << nets[i].gener;
        if (rated) {
            netlistdump << "," << nets[i].rating;
        }
        netlistdump << endl;
    }
    netlistdump.close();
}


int main(int argc, char* argv[]) {
    if (argc <= 2) {
        cerr << "Syntax: pseres <saiXX> <sha256hash> [-p]" << endl
             << "net hash is hook/root with Elo fixed to 0" << endl
             << "  -p        prune leaf nodes" << endl;
        exit (1);
    }
    string saiXX(argv[1]);
    string hook(argv[2]);

    bool prune = false;
    if (argc >= 4) {
        string option(argv[3]);
        if (option == "-p") {
            prune = true;
        }
    }

    load_netsdata(saiXX + "-netdata.xls", hook);
    load_matchdata(saiXX + "-matches.xls");

    check_indices();
    list_wins();
    const auto c_nets = connect_graph();

    cout << "Nets found: " << nets.size() << ". "
        "Linked to hook: " << c_nets << ". "
        "Matches found: " << mats.size() << endl;

    remove_unconnected_nets(false);
    const auto ratings_available = rate_connected_nets(saiXX + "-ratings.csv");
    write_netlist(saiXX + "-rated-nets.csv", ratings_available);
    remove_unconnected_nets(prune);

    const auto n = nets.size();
    if (prune) {
        cout << "Pruning non-root leaf nodes." << endl;
    }
    cout << "Remaining nets: " << n << endl;
    vector< vector<int> > table(n, vector<int>(n));
    vector< vector<int> > table_num(n, vector<int>(n));
    populate_tables(table, table_num);

    write_netlist(saiXX + "-nets.csv");
    write_table(table, saiXX + "-vs.csv");
    write_table(table_num, saiXX + "-num.csv");
    
    // vector<double> rats(n);
    // random_init(rats);

    // auto l = log_likely(rats, table);
    // cout << l << endl;
    // double delta;
    // auto h = 0.1;
    // auto it = 1.0;
    // do {
    //     it *= 1.1;
    //     auto der_la = 0.0;
    //     delta = gradient_desc(rats, h, table, der_la);
    //     l = log_likely(rats, table);
    //     cout << h << ", "
    //          << delta/h << ", "
    //          << der_la << ", "
    //          << l << ", "
    //          << rats[n-1] << endl;
    //     if (delta/h < 100.0 * it * h) {
    //         h *= 0.995;
    //     } else {
    //         h *= 1.002;
    //     }
    // } while (delta > .01);
    // cout << log_likely(rats, table);
    // for (auto & r : rats) {
    //     cout << r << endl;
    // }

    
    return 0;
}
