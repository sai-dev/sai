#include<algorithm>
#include<cmath>
#include<fstream>
#include<iostream>
#include<random>
#include<string>
#include<vector>

using namespace std;

struct sainet {
    string hash;
    long int cumul;
    long int steps;
    long int games;
    int index = 0;
    int hookdist = -1;
};

struct match {
    string hash1;
    string hash2;
    int h1wins;
    int h2wins;
};

struct wincount {
    int idx1;
    int idx2;
    int wins;
};

typedef vector< vector<int> > tab_t;

vector<sainet> nets;
vector<match> mats;
vector<wincount> wins;


string hook =
    "fec1ac47a2db190bb39345e5c9466d305d22ecac95f45cb9d6fbd28f40720126";


void load_netsdata() {
    ifstream netsdata;

    netsdata.open("sai29-netdata.xls");
    if (!netsdata) {
        cerr << "Unable to open nets data file.";
        exit (1);
    }

    string s;
    getline(netsdata, s);
    getline(netsdata, s);
    getline(netsdata, s);

    auto i = 0;
    sainet tmp;
    while (netsdata >> s) {
        switch (i) {
        case 6:
            tmp.hash = s.substr(1, 64);
            tmp.cumul = 0;
            tmp.steps = 0;
            tmp.games = 0;
            if (tmp.hash == hook) {
                tmp.hookdist = 0;
            } else {
                tmp.hookdist = -1;
            }
            break;
        case 21:
            if (s.back() == ',') {
                s.pop_back();
            }
            tmp.cumul = stoi(s);
            break;
        case 24:
            if (s.back() == ',') {
                s.pop_back();
            }
            if (s == "null") {
                tmp.steps = 0;
            } else {
                tmp.steps = stoi(s);
            }
            break;
        case 27:
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
        i++;
        if (s == "}") {
            // cout << tmp.index << ". "
            //           << tmp.hash << ", "
            //           << tmp.cumul << ", "
            //           << tmp.steps << ", "
            //           << tmp.games << endl;
            nets.emplace_back(tmp);
            tmp.index++;
            i = 0;
        }
    } 

    netsdata.close();
}


void load_matchdata() {
    ifstream matchdata;

    matchdata.open("sai29-matches.xls");
    if (!matchdata) {
        cerr << "Unable to open matchdata file.";
        exit (1);
    }

    string s;
    getline(matchdata, s);
    getline(matchdata, s);
    getline(matchdata, s);

    auto i = 0;
    match tmp;
    while (matchdata >> s) {
        switch (i) {
        case 3:
            tmp.hash1 = s.substr(1, 64);
            tmp.hash2 = "";
            tmp.h1wins = 0;
            tmp.h2wins = 0;
            break;
        case 6:
            tmp.hash2 = s.substr(1, 64);
            break;
        case 9:
            if (s.back() == ',') {
                s.pop_back();
            }
            tmp.h2wins = stoi(s);
            break;
        case 12:
            if (s.back() == ',') {
                s.pop_back();
            }
            tmp.h1wins = stoi(s);
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
        ij.wins = vs.h1wins;
        ji.wins = vs.h2wins;
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
        for (auto edge : wins) {
            auto & net1 = nets[edge.idx1];
            const auto & net2 = nets[edge.idx2];
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

void remove_unconnected_nets() {
    for (auto itnet = nets.begin() ; itnet != nets.end() ; ) {
        auto & net = *itnet;
        if (net.hookdist == -1) {
            itnet = nets.erase(itnet);
        } else {
            itnet++;
        }
    }

    auto i = 0;
    for (auto & net : nets) {
        net.index = i++;
    }
}

void populate_table(tab_t & table) {
    for (auto & vs : wins) {
        table[vs.idx1][vs.idx2] = vs.wins;
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

void write_table(tab_t & table) {
    ofstream tabledump;

    tabledump.open("sai29-vs.csv");
    if (!tabledump) {
        cerr << "Unable to open file to dump table.";
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


void write_netlist() {
    ofstream netlistdump;

    netlistdump.open("sai29-nets.csv");
    if (!netlistdump) {
        cerr << "Unable to open file to dump netlist.";
        exit (1);
    }

    const auto n = nets.size();
    for (size_t i=0 ; i < n ; i++) {
        netlistdump << nets[i].hash << ","
                    << nets[i].cumul << ","
                    << nets[i].steps << ","
                    << nets[i].games
                    << endl;
    }
    netlistdump.close();
}


int main() {

    load_netsdata();
    load_matchdata();

    check_indices();
    list_wins();
    const auto c_nets = connect_graph();

    cout << "Nets found: "
              << nets.size() << ". Linked to hook: "
              << c_nets << endl;

    remove_unconnected_nets();
    check_indices();
    wins.clear();
    list_wins();

    const auto n = nets.size();
    vector< vector<int> > table(n, vector<int>(n));
    populate_table(table);

    write_netlist();
    write_table(table);
    
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
