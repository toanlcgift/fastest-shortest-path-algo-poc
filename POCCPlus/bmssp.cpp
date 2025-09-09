// bmssp_full_impl.cpp
// C++17 single-file port of the Python BMSSP practical implementation.
// Includes: graph generator, Dijkstra, DataStructureD, FIND_PIVOTS, BASECASE, BMSSP recursion, instrumentation, and test harness.

#include <bits/stdc++.h>
using namespace std;

using Node = int;
using Weight = double;
using Edge = tuple<Node, Node, Weight>;
using AdjList = vector<vector<pair<Node, Weight>>>;

static inline double INF_D() { return numeric_limits<double>::infinity(); }

// ---------------------------
// Instrumentation
// ---------------------------
struct Instrument {
    uint64_t relaxations = 0;
    uint64_t heap_ops = 0;
    void reset() { relaxations = heap_ops = 0; }
};

// ---------------------------
// Utilities & Graph generator
// ---------------------------
pair<AdjList, vector<Edge>> generate_sparse_directed_graph(int n, int m, double max_w = 100.0, optional<unsigned int> seed = nullopt) {
    std::mt19937_64 rng;
    if (seed.has_value()) rng.seed(seed.value());
    else rng.seed(std::random_device{}());

    uniform_real_distribution<double> wdist(1.0, max_w);
    uniform_real_distribution<double> unit(0.0, 1.0);

    AdjList graph(n);
    vector<Edge> edges;
    // weak backbone
    for (int i = 1; i < n; ++i) {
        uniform_int_distribution<int> r(0, i - 1);
        int u = r(rng);
        double w = wdist(rng);
        graph[u].emplace_back(i, w);
        edges.emplace_back(u, i, w);
    }
    int remaining = max(0, m - (n - 1));
    uniform_int_distribution<int> rn(0, max(0, n - 1));
    for (int i = 0; i < remaining; ++i) {
        int u = rn(rng);
        int v = rn(rng);
        double w = wdist(rng);
        graph[u].emplace_back(v, w);
        edges.emplace_back(u, v, w);
    }
    return {graph, edges};
}

// ---------------------------
// Dijkstra (standard)
// ---------------------------
unordered_map<Node, Weight> dijkstra(const AdjList &graph, Node source, Instrument *instr = nullptr) {
    if (!instr) instr = new Instrument(); // if nullptr, avoid using but create temp to keep code simple
    unordered_map<Node, Weight> dist;
    int n = (int)graph.size();
    dist.reserve(n * 2);
    for (int i = 0; i < n; ++i) dist[i] = INF_D();
    dist[source] = 0.0;

    // min-heap: pair<dist, node>
    priority_queue<pair<Weight, Node>, vector<pair<Weight, Node>>, greater<pair<Weight, Node>>> pq;
    pq.emplace(0.0, source);
    instr->heap_ops += 1;

    while (!pq.empty()) {
        auto [d_u, u] = pq.top(); pq.pop();
        instr->heap_ops += 1;
        if (d_u > dist[u]) continue;
        for (auto &e : graph[u]) {
            Node v = e.first;
            Weight w = e.second;
            instr->relaxations += 1;
            double alt = d_u + w;
            auto it = dist.find(v);
            double cur = (it != dist.end() ? it->second : INF_D());
            if (alt < cur) {
                dist[v] = alt;
                pq.emplace(alt, v);
                instr->heap_ops += 1;
            }
        }
    }
    return dist;
}

// ---------------------------
// DataStructure D (practical)
// ---------------------------
class DataStructureD {
public:
    DataStructureD(int M, double B_upper, optional<int> block_size = nullopt)
        : M_(max(1, M)), B_upper_(B_upper) {
        block_size_ = block_size.has_value() ? block_size.value() : max(1, M_ / 8);
    }

    void insert(Node v, Weight key) {
        auto it = best_.find(v);
        if (it == best_.end() || key < it->second) {
            best_[v] = key;
            heap_.emplace(key, v);
        }
    }

    // iterable of (v, key)
    void batch_prepend(const vector<pair<Node, Weight>>& items) {
        for (const auto &p : items) insert(p.first, p.second);
    }

    bool empty() {
        cleanup();
        return heap_.empty();
    }

    // pull -> (Bi, Si)
    pair<Weight, unordered_set<Node>> pull() {
        cleanup();
        if (heap_.empty()) throw out_of_range("pull from empty D");
        Weight Bi = heap_.top().first;
        unordered_set<Node> Si;
        while (!heap_.empty() && (int)Si.size() < block_size_) {
            auto pr = heap_.top(); heap_.pop();
            Weight key = pr.first;
            Node v = pr.second;
            auto it = best_.find(v);
            if (it != best_.end() && it->second == key) {
                Si.insert(v);
                best_.erase(it);
            }
        }
        return {Bi, Si};
    }

private:
    void cleanup() {
        while (!heap_.empty()) {
            auto pr = heap_.top();
            Node v = pr.second; Weight key = pr.first;
            auto it = best_.find(v);
            if (it == best_.end() || it->second != key) {
                heap_.pop();
            } else break;
        }
    }

    // min-heap (pair<key,node>) using greater comparator
    priority_queue<pair<Weight, Node>, vector<pair<Weight, Node>>, greater<pair<Weight, Node>>> heap_;
    unordered_map<Node, Weight> best_;
    int M_;
    double B_upper_;
    int block_size_;
};

// ---------------------------
// FIND_PIVOTS (bounded BF-like)
// ---------------------------
pair<unordered_set<Node>, unordered_set<Node>> find_pivots(
    const AdjList &graph,
    const unordered_map<Node, Weight> &dist,
    const unordered_set<Node> &S,
    double B,
    int n,
    int k_steps,
    int p_limit,
    Instrument *instr = nullptr
) {
    if (!instr) instr = new Instrument();

    // S_filtered = nodes in S with dist < B
    vector<Node> s_filtered;
    s_filtered.reserve(S.size());
    for (Node v : S) {
        auto it = dist.find(v);
        if (it != dist.end() && it->second < B) s_filtered.push_back(v);
    }

    unordered_set<Node> P;
    if (s_filtered.empty()) {
        // fallback: up to p_limit arbitrary samples from S
        if (!S.empty()) {
            int take = max(1, min((int)S.size(), p_limit));
            int count = 0;
            for (Node v : S) {
                P.insert(v);
                if (++count >= take) break;
            }
        }
    } else {
        sort(s_filtered.begin(), s_filtered.end(), [&](Node a, Node b) {
            double da = dist.at(a), db = dist.at(b);
            return da < db;
        });
        int take = max(1, min((int)s_filtered.size(), p_limit));
        for (int i = 0; i < take; ++i) P.insert(s_filtered[i]);
    }

    unordered_set<Node> source_frontier = (!P.empty() ? P : S);
    unordered_set<Node> discovered = source_frontier;
    unordered_set<Node> frontier = source_frontier;

    int steps = max(1, k_steps);
    for (int it_step = 0; it_step < steps; ++it_step) {
        if (frontier.empty()) break;
        unordered_set<Node> next_front;
        for (Node u : frontier) {
            double du = INF_D();
            auto itdu = dist.find(u);
            if (itdu != dist.end()) du = itdu->second;
            if (du >= B) continue;
            if ((size_t)u >= graph.size()) continue;
            for (const auto &pr : graph[u]) {
                Node v = pr.first; Weight w = pr.second;
                instr->relaxations += 1;
                double nd = du + w;
                if (nd < B && discovered.find(v) == discovered.end()) {
                    discovered.insert(v);
                    next_front.insert(v);
                }
            }
        }
        frontier.swap(next_front);
    }

    unordered_set<Node> W = discovered;
    if (P.empty() && !S.empty()) {
        P.insert(*S.begin());
    }
    return {P, W};
}

// ---------------------------
// BASECASE (limited Dijkstra)
 // ---------------------------
pair<Weight, unordered_set<Node>> basecase(
    const AdjList &graph,
    unordered_map<Node, Weight> &dist,
    double B,
    const unordered_set<Node> &S,
    int k,
    Instrument *instr = nullptr
) {
    if (!instr) instr = new Instrument();

    if (S.empty()) return {B, {}};

    // choose source x in S with smallest dist
    Node x = -1;
    double bestd = INF_D();
    for (Node v : S) {
        auto it = dist.find(v);
        double dv = (it != dist.end() ? it->second : INF_D());
        if (dv < bestd) { bestd = dv; x = v; }
    }
    if (x == -1) return {B, {}};

    priority_queue<pair<Weight, Node>, vector<pair<Weight, Node>>, greater<pair<Weight, Node>>> pq;
    double start_d = (dist.find(x) != dist.end()) ? dist[x] : INF_D();
    pq.emplace(start_d, x);
    instr->heap_ops += 1;

    unordered_set<Node> Uo;
    unordered_set<Node> visited;

    while (!pq.empty() && (int)Uo.size() < (k + 1)) {
        auto [d_u, u] = pq.top(); pq.pop();
        instr->heap_ops += 1;
        auto itu = dist.find(u);
        double currentDu = (itu != dist.end() ? itu->second : INF_D());
        if (d_u > currentDu) continue;
        if (Uo.find(u) == Uo.end()) Uo.insert(u);
        if ((size_t)u >= graph.size()) continue;
        for (auto &pr : graph[u]) {
            Node v = pr.first; Weight w = pr.second;
            instr->relaxations += 1;
            double newd = dist[u] + w;
            double dv = dist.find(v) != dist.end() ? dist[v] : INF_D();
            if (newd < dv && newd < B) {
                dist[v] = newd;
                pq.emplace(newd, v);
                instr->heap_ops += 1;
            }
        }
    }

    if ((int)Uo.size() <= k) {
        return {B, Uo};
    } else {
        vector<double> finite_dists;
        finite_dists.reserve(Uo.size());
        for (Node v : Uo) {
            auto it = dist.find(v);
            if (it != dist.end() && isfinite(it->second)) finite_dists.push_back(it->second);
        }
        if (finite_dists.empty()) return {B, {}};
        double maxd = *max_element(finite_dists.begin(), finite_dists.end());
        unordered_set<Node> U_filtered;
        for (Node v : Uo) {
            auto it = dist.find(v);
            if (it != dist.end() && it->second < maxd) U_filtered.insert(v);
        }
        return {maxd, U_filtered};
    }
}

// ---------------------------
// BMSSP (practical recursive)
// ---------------------------
pair<Weight, unordered_set<Node>> bmssp(
    const AdjList &graph,
    unordered_map<Node, Weight> &dist,
    const vector<Edge> &edges,
    int l,
    double B,
    const unordered_set<Node> &S,
    int n,
    Instrument *instr = nullptr
) {
    if (!instr) instr = new Instrument();

    int t_param, k_param;
    if (n <= 2) { t_param = 1; k_param = 2; }
    else {
        double ln = log(max(3, n));
        t_param = max(1, (int)round(pow(ln, 2.0 / 3.0)));
        k_param = max(2, (int)round(pow(ln, 1.0 / 3.0)));
    }

    if (l <= 0) {
        if (S.empty()) return {B, {}};
        return basecase(graph, dist, B, S, k_param, instr);
    }

    int p_limit = max(1, 1 << min(10, t_param));
    int k_steps = max(1, k_param);

    auto piv = find_pivots(graph, dist, S, B, n, k_steps, p_limit, instr);
    unordered_set<Node> P = move(piv.first);
    unordered_set<Node> W = move(piv.second);

    int M = 1 << max(0, (l - 1) * t_param);
    DataStructureD D(M, B, max(1, (int)min((int)max(1, (int)P.size()), 64)));
    for (Node x : P) {
        double dx = INF_D();
        auto it = dist.find(x);
        if (it != dist.end()) dx = it->second;
        D.insert(x, dx);
    }

    double B_prime_initial = B;
    if (!P.empty()) {
        double minp = INF_D();
        for (Node x : P) {
            auto it = dist.find(x);
            double dx = (it != dist.end() ? it->second : INF_D());
            if (dx < minp) minp = dx;
        }
        if (isfinite(minp)) B_prime_initial = minp;
    }

    unordered_set<Node> U;
    vector<double> B_prime_sub_values;

    int loop_guard = 0;
    int limit = k_param * (1 << (l * max(1, t_param)));

    while ((int)U.size() < limit && !D.empty()) {
        loop_guard++;
        if (loop_guard > 20000) break;
        Weight Bi;
        unordered_set<Node> Si;
        try {
            tie(Bi, Si) = D.pull();
        } catch (...) {
            break;
        }

        auto subres = bmssp(graph, dist, edges, l - 1, Bi, Si, n, instr);
        double B_prime_sub = subres.first;
        unordered_set<Node> Ui = move(subres.second);
        B_prime_sub_values.push_back(B_prime_sub);
        for (auto u : Ui) U.insert(u);

        // Prepare batch set
        vector<pair<Node, Weight>> K_for_batch;
        K_for_batch.reserve(64);

        for (Node u : Ui) {
            auto itu = dist.find(u);
            if (itu == dist.end()) continue;
            double du = itu->second;
            if (!isfinite(du)) continue;
            if ((size_t)u >= graph.size()) continue;
            for (auto &pr : graph[u]) {
                Node v = pr.first; Weight w_uv = pr.second;
                instr->relaxations += 1;
                double newd = du + w_uv;
                double dv = dist.find(v) != dist.end() ? dist[v] : INF_D();
                if (newd <= dv) {
                    dist[v] = newd;
                    if (Bi <= newd && newd < B) {
                        D.insert(v, newd);
                    } else if (B_prime_sub <= newd && newd < Bi) {
                        K_for_batch.emplace_back(v, newd);
                    }
                }
            }
        }

        for (Node x : Si) {
            auto itx = dist.find(x);
            double dx = (itx != dist.end() ? itx->second : INF_D());
            if (B_prime_sub <= dx && dx < Bi) {
                K_for_batch.emplace_back(x, dx);
            }
        }

        if (!K_for_batch.empty()) D.batch_prepend(K_for_batch);
    }

    double B_prime_final = B_prime_initial;
    if (!B_prime_sub_values.empty()) {
        double minsub = *min_element(B_prime_sub_values.begin(), B_prime_sub_values.end());
        B_prime_final = min(B_prime_initial, minsub);
    }

    unordered_set<Node> U_final = U;
    for (Node x : W) {
        auto it = dist.find(x);
        double dx = (it != dist.end() ? it->second : INF_D());
        if (dx < B_prime_final) U_final.insert(x);
    }

    return {B_prime_final, U_final};
}

// ---------------------------
// Test harness
// ---------------------------
struct TestResult {
    int n; int m; unsigned int seed;
    double dijkstra_time;
    uint64_t dijkstra_relax;
    double bmssp_time;
    uint64_t bmssp_relax;
    int dijkstra_reachable;
    int bmssp_reachable;
    double max_diff;
};

TestResult run_single_test(int n, int m, unsigned int seed = 0, Node source = 0) {
    cout << "Generating graph: n=" << n << ", m=" << m << ", seed=" << seed << "\n";
    auto gen = generate_sparse_directed_graph(n, m, 100.0, seed);
    AdjList graph = move(gen.first);
    vector<Edge> edges = move(gen.second);
    double avg_deg = 0.0;
    for (auto &adj : graph) avg_deg += (double)adj.size();
    avg_deg /= (double)max(1, n);
    cout << "Graph generated. avg out-degree ≈ " << fixed << setprecision(3) << avg_deg << "\n";

    // Dijkstra
    Instrument instr_dij;
    auto t0 = chrono::high_resolution_clock::now();
    auto dist_dij_map = dijkstra(graph, source, &instr_dij);
    auto t1 = chrono::high_resolution_clock::now();
    double dij_time = chrono::duration<double>(t1 - t0).count();
    int reachable_dij = 0;
    for (auto &kv : dist_dij_map) if (isfinite(kv.second)) ++reachable_dij;
    cout << "Dijkstra: time=" << dij_time << "s, relaxations=" << instr_dij.relaxations << ", heap_ops=" << instr_dij.heap_ops << ", reachable=" << reachable_dij << "\n";

    // BMSSP
    unordered_map<Node, Weight> dist_bm;
    dist_bm.reserve(n * 2);
    for (int i = 0; i < n; ++i) dist_bm[i] = INF_D();
    dist_bm[source] = 0.0;

    Instrument instr_bm;
    int l;
    if (n <= 2) l = 1;
    else {
        double ln = log(max(3, n));
        int t_guess = max(1, (int)round(pow(ln, 2.0 / 3.0)));
        l = max(1, (int)max(1.0, round(ln / (double)t_guess)));
    }
    cout << "BMSSP params: top-level l=" << l << "\n";

    auto t2 = chrono::high_resolution_clock::now();
    auto res = bmssp(graph, dist_bm, edges, l, INF_D(), unordered_set<Node>{source}, n, &instr_bm);
    auto t3 = chrono::high_resolution_clock::now();
    double bm_time = chrono::duration<double>(t3 - t2).count();
    int reachable_bm = 0;
    for (auto &kv : dist_bm) if (isfinite(kv.second)) ++reachable_bm;
    cout << "BMSSP: time=" << bm_time << "s, relaxations=" << instr_bm.relaxations << ", reachable=" << reachable_bm << ", B'=" << res.first << ", |U_final|=" << res.second.size() << "\n";

    // Compare distances on commonly reachable nodes
    vector<double> diffs;
    for (int v = 0; v < n; ++v) {
        double dv = INF_D();
        double db = INF_D();
        auto it1 = dist_dij_map.find(v);
        if (it1 != dist_dij_map.end()) dv = it1->second;
        auto it2 = dist_bm.find(v);
        if (it2 != dist_bm.end()) db = it2->second;
        if (isfinite(dv) && isfinite(db)) diffs.push_back(fabs(dv - db));
    }
    double max_diff = 0.0;
    if (!diffs.empty()) max_diff = *max_element(diffs.begin(), diffs.end());
    cout << "Distance agreement (max abs diff on commonly reachable nodes): " << scientific << setprecision(6) << max_diff << "\n";

    TestResult tr;
    tr.n = n; tr.m = m; tr.seed = seed;
    tr.dijkstra_time = dij_time; tr.dijkstra_relax = instr_dij.relaxations;
    tr.bmssp_time = bm_time; tr.bmssp_relax = instr_bm.relaxations;
    tr.dijkstra_reachable = reachable_dij; tr.bmssp_reachable = reachable_bm;
    tr.max_diff = max_diff;
    return tr;
}

// ---------------------------
// CLI main
// ---------------------------
int main(int argc, char** argv) {
    int n = 200000;
    int m = 800000;
    unsigned int seed = 0;
    for (int i = 1; i < argc; ++i) {
        string a = argv[i];
        if ((a == "-n" || a == "--nodes") && i + 1 < argc) { n = stoi(argv[++i]); }
        else if ((a == "-m" || a == "--edges") && i + 1 < argc) { m = stoi(argv[++i]); }
        else if ((a == "-s" || a == "--seed") && i + 1 < argc) { seed = (unsigned int)stoul(argv[++i]); }
    }

    // To keep runtime reasonable by default for C++ demo, clamp default n/m if user doesn't want huge run:
    if (n > 5000000) n = 5000000;
    if (m > 50000000) m = 50000000;

    try {
        run_single_test(n, m, seed, 0);
    } catch (const exception &ex) {
        cerr << "Error: " << ex.what() << endl;
        return 2;
    }
    return 0;
}
