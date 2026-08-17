#include <bits/stdc++.h>
using namespace std;

constexpr int W = 3; // lattice width
constexpr int MAX_N = 30; // how many terms to compute

const string CHECKPOINT_META = "checkpoint_meta.bin";
const string CHECKPOINT_LOG  = "checkpoint_log.bin";
using Col = uint8_t;
using Board = vector<Col>;

static inline void trim(Board& b) {
    while (!b.empty() && b.front() == 0) b.erase(b.begin());
    while (!b.empty() && b.back()  == 0) b.pop_back();
}

static inline Col mirrorRows(Col c) {
    Col r = 0;
    for (int i = 0; i < W; i++)
        if (c & (1 << i)) r |= (1 << (W - 1 - i));
    return r;
}

static Board canonical(Board b) {
    trim(b);
    if (b.empty()) return b;

    Board rev(b.rbegin(), b.rend());
    Board mir(b.size());
    for (size_t i = 0; i < b.size(); i++) mir[i] = mirrorRows(b[i]);
    Board revmir(mir.rbegin(), mir.rend());

    Board best = b;
    if (rev    < best) best = rev;
    if (mir    < best) best = mir;
    if (revmir < best) best = revmir;
    return best;
}

static inline __uint128_t packBoard(const Board& b) {
    __uint128_t key = 0;
    for (Col c : b) key = (key << W) | c;
    key |= (__uint128_t)1 << (W * b.size());
    return key;
}

struct U128Hash {
    size_t operator()(__uint128_t x) const noexcept {
        uint64_t hi = (uint64_t)(x >> 64), lo = (uint64_t)x;
        return std::hash<uint64_t>()(hi) ^ (std::hash<uint64_t>()(lo) + 0x9e3779b97f4a7c15ULL);
    }
};


constexpr int NUM_SHARDS = 64;

struct Shard {
    unordered_map<__uint128_t, int, U128Hash> map;
    mutex mtx;
};
vector<Shard> g_shards(NUM_SHARDS);
U128Hash g_hasher;

static inline Shard& shardFor(__uint128_t key) {
    return g_shards[g_hasher(key) % NUM_SHARDS];
}

static inline bool memoGet(__uint128_t key, int& out) {
    Shard& s = shardFor(key);
    lock_guard<mutex> lock(s.mtx);
    auto it = s.map.find(key);
    if (it == s.map.end()) return false;
    out = it->second;
    return true;
}

static inline void logEntry(__uint128_t key, int value);

static inline void memoPut(__uint128_t key, int value) {
    Shard& s = shardFor(key);
    lock_guard<mutex> lock(s.mtx);
    s.map[key] = value;
}

static inline void memoPutLogged(__uint128_t key, int value) {
    memoPut(key, value);
    logEntry(key, value);
}

static inline size_t memoSize() {
    size_t total = 0;
    for (auto& s : g_shards) total += s.map.size();
    return total;
}

vector<pair<__uint128_t, int>> g_log;
mutex g_log_mtx;
size_t g_log_flushed = 0;

static inline void logEntry(__uint128_t key, int value) {
    lock_guard<mutex> lock(g_log_mtx);
    g_log.push_back({key, value});
}

static void flushLog() {
    lock_guard<mutex> lock(g_log_mtx);
    if (g_log_flushed >= g_log.size()) return;
    ofstream out(CHECKPOINT_LOG, ios::binary | ios::app);
    for (size_t i = g_log_flushed; i < g_log.size(); i++) {
        uint64_t hi = (uint64_t)(g_log[i].first >> 64);
        uint64_t lo = (uint64_t)(g_log[i].first);
        int32_t  val = g_log[i].second;
        out.write(reinterpret_cast<const char*>(&hi), sizeof(hi));
        out.write(reinterpret_cast<const char*>(&lo), sizeof(lo));
        out.write(reinterpret_cast<const char*>(&val), sizeof(val));
    }
    g_log_flushed = g_log.size();
}

static void writeMeta(int lastN) {
    ofstream out(CHECKPOINT_META, ios::binary | ios::trunc);
    int32_t v = lastN;
    out.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

static int loadCheckpoint() {
    ifstream meta(CHECKPOINT_META, ios::binary);
    if (!meta) return 0;
    int32_t lastN = 0;
    meta.read(reinterpret_cast<char*>(&lastN), sizeof(lastN));
    if (!meta) return 0;

    ifstream log(CHECKPOINT_LOG, ios::binary);
    if (log) {
        uint64_t hi, lo;
        int32_t val;
        size_t count = 0;
        while (log.read(reinterpret_cast<char*>(&hi), sizeof(hi))) {
            log.read(reinterpret_cast<char*>(&lo), sizeof(lo));
            log.read(reinterpret_cast<char*>(&val), sizeof(val));
            if (!log) break;
            __uint128_t key = ((__uint128_t)hi << 64) | lo;
            memoPut(key, val);
            count++;
        }
        cout << "resumed from checkpoint: loaded " << count
             << " cached states, last completed n=" << lastN << "\n";
    }
    return lastN;
}

// i love bfs
static vector<Board> splitComponents(const Board& b) {
    int n = (int)b.size();
    vector<array<bool, 4>> visited(n); // W <= 4
    for (auto& row : visited) row.fill(false);

    vector<Board> comps;
    for (int c = 0; c < n; c++) {
        for (int r = 0; r < W; r++) {
            if (!(b[c] & (1 << r)) || visited[c][r]) continue;

            vector<pair<int,int>> verts;
            queue<pair<int,int>> q;
            q.push({c, r});
            visited[c][r] = true;
            int cmin = c, cmax = c;

            static const int dc[4] = {0, 0, 1, -1};
            static const int dr[4] = {1, -1, 0, 0};

            while (!q.empty()) {
                auto [cc, rr] = q.front(); q.pop();
                verts.push_back({cc, rr});
                cmin = min(cmin, cc); cmax = max(cmax, cc);
                for (int k = 0; k < 4; k++) {
                    int nc = cc + dc[k], nr = rr + dr[k];
                    if (nc < 0 || nc >= n || nr < 0 || nr >= W) continue;
                    if (!(b[nc] & (1 << nr)) || visited[nc][nr]) continue;
                    visited[nc][nr] = true;
                    q.push({nc, nr});
                }
            }

            Board comp(cmax - cmin + 1, 0);
            for (auto& [cc, rr] : verts) comp[cc - cmin] |= (1 << rr);
            comps.push_back(std::move(comp));
        }
    }
    return comps;
}

static Board applyMove(Board b, int col, int row) {
    b[col] &= ~(1 << row);
    if (row > 0)                       b[col] &= ~(1 << (row - 1));
    if (row < W - 1)                   b[col] &= ~(1 << (row + 1));
    if (col > 0)                       b[col - 1] &= ~(1 << row);
    if (col + 1 < (int)b.size())       b[col + 1] &= ~(1 << row);
    return b;
}

int grundy(Board b) {
    b = canonical(b);
    if (b.empty()) return 0;

    __uint128_t key = packBoard(b);
    int cached;
    if (memoGet(key, cached)) return cached;

    vector<int> reachable;
    int n = (int)b.size();
    for (int c = 0; c < n; c++) {
        for (int r = 0; r < W; r++) {
            if (!(b[c] & (1 << r))) continue;
            Board after = applyMove(b, c, r);
            int g = 0;
            for (Board& comp : splitComponents(after)) g ^= grundy(comp);
            reachable.push_back(g);
        }
    }

    sort(reachable.begin(), reachable.end());
    reachable.erase(unique(reachable.begin(), reachable.end()), reachable.end());
    int mex = 0;
    for (int v : reachable) {
        if (v == mex) mex++;
        else if (v > mex) break;
    }

    memoPutLogged(key, mex);
    return mex;
}

int grundyParallel(Board b, int numThreads) {
    b = canonical(b);
    if (b.empty()) return 0;

    __uint128_t key = packBoard(b);
    int cached;
    if (memoGet(key, cached)) return cached;

    vector<pair<int,int>> moves;
    int n = (int)b.size();
    for (int c = 0; c < n; c++)
        for (int r = 0; r < W; r++)
            if (b[c] & (1 << r)) moves.push_back({c, r});

    vector<int> results(moves.size());
    atomic<size_t> idx{0};

    auto worker = [&]() {
        size_t i;
        while ((i = idx.fetch_add(1)) < moves.size()) {
            auto [c, r] = moves[i];
            Board after = applyMove(b, c, r);
            int g = 0;
            for (Board& comp : splitComponents(after)) g ^= grundy(comp);
            results[i] = g;
        }
    };

    numThreads = max(1, numThreads);
    vector<thread> pool;
    for (int t = 0; t < numThreads; t++) pool.emplace_back(worker);
    for (auto& th : pool) th.join();

    sort(results.begin(), results.end());
    results.erase(unique(results.begin(), results.end()), results.end());
    int mex = 0;
    for (int v : results) {
        if (v == mex) mex++;
        else if (v > mex) break;
    }

    memoPutLogged(key, mex);
    return mex;
}

int main() {
    for (auto& s : g_shards) s.map.reserve(1 << 15);

    unsigned hw = thread::hardware_concurrency();
    int numThreads = hw ? (int)hw : 4;
    cout << "using " << numThreads << " threads for root-level parallelism\n";

    int lastN = loadCheckpoint();
    int startN = lastN + 1;
    if (lastN > 0) cout << "resuming at n=" << startN << "\n";

    Board full(max(0, lastN), (Col)((1 << W) - 1));
    for (int n = startN; n <= MAX_N; n++) {
        full.push_back((Col)((1 << W) - 1));
        auto t0 = chrono::steady_clock::now();
        int g = grundyParallel(full, numThreads);
        auto t1 = chrono::steady_clock::now();
        double secs = chrono::duration<double>(t1 - t0).count();
        cout << "n=" << n << "  G=" << g
             << "  memo=" << memoSize()
             << "  time=" << secs << "s\n";

        flushLog();
        writeMeta(n);
    }
    return 0;
}