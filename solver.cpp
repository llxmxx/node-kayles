#include <bits/stdc++.h>
using namespace std;

using col = uint8_t;
using board = vector<col>;

constexpr int W = 3; // lattice width: 3 or 4
constexpr int MAX_N = 30; // how many terms to compute

struct U128Hash{
    size_t operator()(__uint128_t x) const noexcept{
        uint64_t hi = (uint64_t)(x >> 64), lo = (uint64_t)x;
        return hash<uint64_t>()(hi) ^ (std::hash<uint64_t>()(lo) + 0x9e3779b97f4a7c15ULL);
    }
};

struct Shard{
    unordered_map<__uint128_t, int, U128Hash> map;
    mutex mtx;
};

constexpr int NUM_SHARDS = 64;

vector<Shard> g_shards(NUM_SHARDS);
U128Hash g_hasher;

static inline void trim(board &b){
    while(!b.empty() && b.front() == 0) b.erase(b.begin());
    while(!b.empty() && b.back() == 0) b.pop_back();
}

static inline col mirrorRows(col c){
    col r = 0;
    for(int i = 0; i < W; i++){
        if(c & (1 << i)) r |= (1 << (W-1-i));
    }
    return r;
}

static board canonical(board b){
    trim(b);
    if(b.empty()) return b;

    board rev(b.rbegin(), b.rend());
    board mir(b.size());
    for(size_t i = 0; i < b.size(); i++) mir[i] = mirrorRows(b[i]);
    board revmir(mir.rbegin(), mir.rend());

    board best = b;
    if(rev < best) best = rev;
    if(mir < best) best = mir;
    if(revmir < best) best = revmir;
    return best;
}

static inline __uint128_t packBoard(const board &b){
    __uint128_t key = 0;
    for(col c: b) key = (key << W) | c;
    key |= (__uint128_t)1 << (W*b.size());
    return key;
}

static inline Shard &shardFor(__uint128_t key){
    return g_shards[g_hasher(key) % NUM_SHARDS];
}

static inline bool memoGet(__uint128_t key, int &out){
    Shard &s = shardFor(key);
    lock_guard<mutex> lock(s.mtx);
    auto it = s.map.find(key);
    if(it == s.map.end()) return false;
    out = it->second;
    return true;
}

static inline void memoPut(__uint128_t key, int value){
    Shard &s = shardFor(key);
    lock_guard<mutex> lock(s.mtx);
    s.map[key] = value;
}

static inline size_t memoSize(){
    size_t total = 0;
    for(auto &s : g_shards) total += s.map.size();
    return total;
}

// i love bfs
static vector<board> splitComponents(const board &b){
    int n = (int)b.size();
    vector<array<bool, 4>> visited(n); // W<=4
    for(auto &row : visited) row.fill(false);

    vector<board> comps;
    for(int c = 0; c < n; c++){
        for(int r = 0; r < W; r++){
            if(!(b[c] & (1 << r)) || visited[c][r]) continue;

            vector<pair<int, int>> verts;
            queue<pair<int, int>> q;
            q.push({c, r});
            visited[c][r] = true;
            int cmin = c, cmax = c;

            static const int dc[4] = {0, 0, 1, -1};
            static const int dr[4] = {1, -1, 0, 0};

            while(!q.empty()){
                auto [cc, rr] = q.front();
                q.pop();
                verts.push_back({cc, rr});
                cmin = min(cmin, cc); cmax = max(cmax, cc);
                for(int k = 0; k < 4; k++){
                    int nc = cc+dc[k], nr = rr+dr[k];
                    if(nc < 0 || nc >= n || nr < 0 || nr >= W) continue;
                    if(!(b[nc] & (1 << nr)) || visited[nc][nr]) continue;
                    visited[nc][nr] = true;
                    q.push({nc, nr});
                }
            }
            board comp(cmax-cmin+1, 0);
            for(auto &[cc, rr] : verts) comp[cc - cmin] |= (1 << rr);
            comps.push_back(move(comp));
        }
    }
    return comps;
}

static board applyMove(board b, int col, int row){
    b[col] &= ~(1 << row);
    if(row > 0) b[col] &= ~(1 << (row-1));
    if(row < W-1) b[col] &= ~(1 << (row+1));
    if(col > 0) b[col-1] &= ~(1 << row);
    if(col+1 < (int)b.size()) b[col+1] &= ~(1 << row);
    return b;
}

int grundy(board b){
    b = canonical(b);
    if(b.empty()) return 0;

    __uint128_t key = packBoard(b);
    int cached;
    if(memoGet(key, cached)) return cached;

    vector<int> reachable;
    int n = (int)b.size();
    for(int c = 0; c < n; c++){
        for(int r = 0; r < W; r++){
            if(!(b[c] & (1 << r))) continue;
            board after = applyMove(b, c, r);
            int g = 0;
            for(board &comp : splitComponents(after)) g^=grundy(comp);
            reachable.push_back(g);
        }
    }
    sort(reachable.begin(), reachable.end());
    reachable.erase(unique(reachable.begin(), reachable.end()), reachable.end());
    int mex = 0;
    for(int v : reachable){
        if(v == mex) mex++;
        else if(v > mex) break;
    }
    memoPut(key, mex);
    return mex;
}

int grundyParallel(board b, int numThreads){
    b = canonical(b);
    if(b.empty()) return 0;

    __uint128_t key = packBoard(b);
    int cached;
    if(memoGet(key, cached)) return cached;

    vector<pair<int, int>> moves;
    int n = (int)b.size();
    for(int c = 0; c < n; c++){
        for(int r = 0; r < W; r++){
            if(b[c] & (1 << r)) moves.push_back({c, r});
        }
    }

    vector<int> results(moves.size());
    atomic<size_t> idx{0};

    auto worker = [&]() {
        size_t i;
        while((i = idx.fetch_add(1)) < moves.size()){
            auto [c, r] = moves[i];
            board after = applyMove(b, c, r);
            int g = 0;
            for(board &comp: splitComponents(after)) g^=grundy(comp);
            results[i] = g;
        }
    };

    numThreads = max(1, numThreads);
    vector<thread> pool;
    for(int t = 0; t < numThreads; t++){
        pool.emplace_back(worker);
    }
    for(auto &th: pool) th.join();

    sort(results.begin(), results.end());
    results.erase(unique(results.begin(), results.end()), results.end());
    int mex = 0;
    for(int v : results){
        if(v == mex) mex++;
        else if(v > mex) break;
    }
    memoPut(key, mex);
    return mex;
}

int main(){
    for(auto &s: g_shards) s.map.reserve(1 << 15);

    unsigned hw = thread::hardware_concurrency();
    int numThreads = hw ? (int)hw : 4;
    cout << "using " << numThreads << " threads for root-level parallelism\n";

    board full;
    for(int n = 1; n <= MAX_N; n++){
        full.push_back((col)((1 << W)-1));
        auto t0 = chrono::steady_clock::now();
        int g = grundyParallel(full, numThreads);
        auto t1 = chrono::steady_clock::now();
        double secs = chrono::duration<double>(t1-t0).count();
        cout << "n = " << n << " g = " << g << " memo = " << memoSize() << " time = " << secs << "s\n";
    }

    return 0;
}