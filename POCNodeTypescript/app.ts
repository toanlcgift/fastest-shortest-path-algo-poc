#!/usr/bin/env ts-node
/**
 * app.ts
 *
 * Full TypeScript port of the Python BMSSP practical implementation.
 * - Graph generator
 * - Instrumentation
 * - Dijkstra
 * - DataStructureD (heap + map, lazy deletion)
 * - findPivots (bounded BF-like)
 * - basecase (Dijkstra-like limited)
 * - Recursive bmssp
 * - Test harness + minimal CLI
 *
 * Usage:
 *   ts-node bmssp.ts -n 2000 -m 8000 -s 42
 *
 * Note: For big graphs, increase node memory/stack or use Node with --stack-size if needed.
 */

type NodeId = number;
type Weight = number;
type Edge = [NodeId, NodeId, Weight];
type Graph = Map<NodeId, Array<[NodeId, Weight]>>;

const INF = Number.POSITIVE_INFINITY;

/* ----------------------------
   Simple min-heap for (key,node) pairs
   ---------------------------- */
class MinHeap {
    private data: Array<{ key: number; node: number }> = [];

    size(): number {
        return this.data.length;
    }
    isEmpty(): boolean {
        return this.data.length === 0;
    }
    peek(): { key: number; node: number } | null {
        return this.data.length ? this.data[0] ?? null : null;
    }
    push(key: number, node: number) {
        this.data.push({ key, node });
        this.siftUp(this.data.length - 1);
    }
    pop(): { key: number; node: number } | null {
        if (!this.data.length) return null;
        const top = this.data[0];
        const last = this.data.pop()!;
        if (this.data.length) {
            this.data[0] = last;
            this.siftDown(0);
        }
        return top ?? null;
    }
    private siftUp(i: number) {
        while (i > 0) {
            const p = (i - 1) >> 1;
            if (this.data[i]?.key ?? 0 < (this.data[p] ? this.data[p].key : 0)) {
                const temp = this.data[i]!;
                this.data[i] = this.data[p]!;
                this.data[p] = temp;
                i = p;
            } else break;
        }
    }
    private siftDown(i: number) {
        const n = this.data.length;
        while (true) {
            const l = i * 2 + 1;
            const r = l + 1;
            let smallest = i;
            if (l < n && (this.data[l]?.key ?? 0) < (this.data[smallest]?.key ?? 0)) smallest = l;
            if (r < n && (this.data[r]?.key ?? 0) < (this.data[smallest]?.key ?? 0)) smallest = r;
            if (smallest === i) break;
            const temp = this.data[i]!;
            this.data[i] = this.data[smallest]!;
            this.data[smallest] = temp;
            i = smallest;
        }
    }
}

/* ----------------------------
   RNG (seedable) - mulberry32
   ---------------------------- */
function mulberry32(seed: number) {
    let t = seed >>> 0;
    return function () {
        t += 0x6D2B79F5;
        let r = Math.imul(t ^ (t >>> 15), t | 1);
        r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
        return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
    };
}

/* ----------------------------
   Graph generator
   ---------------------------- */
function generateSparseDirectedGraph(n: number, m: number, maxW = 100.0, seed?: number): [Graph, Edge[]] {
    const rand = seed !== undefined ? mulberry32(seed) : Math.random;
    const graph: Graph = new Map();
    for (let i = 0; i < n; i++) graph.set(i, []);
    const edges: Edge[] = [];

    // backbone to avoid isolated nodes
    for (let i = 1; i < n; i++) {
        const u = Math.floor(rand() * i);
        const w = rand() * (maxW - 1.0) + 1.0;
        graph.get(u)!.push([i, w]);
        edges.push([u, i, w]);
    }

    const remaining = Math.max(0, m - (n - 1));
    for (let i = 0; i < remaining; i++) {
        const u = Math.floor(rand() * n);
        const v = Math.floor(rand() * n);
        const w = rand() * (maxW - 1.0) + 1.0;
        graph.get(u)!.push([v, w]);
        edges.push([u, v, w]);
    }
    return [graph, edges];
}

/* ----------------------------
   Instrumentation
   ---------------------------- */
class Instrument {
    relaxations = 0;
    heapOps = 0;
    reset() {
        this.relaxations = 0;
        this.heapOps = 0;
    }
}

/* ----------------------------
   Dijkstra (standard)
   ---------------------------- */
function dijkstra(graph: Graph, source: NodeId, instr?: Instrument): Map<NodeId, number> {
    instr = instr ?? new Instrument();
    const dist: Map<NodeId, number> = new Map();
    for (const k of graph.keys()) dist.set(k, INF);
    dist.set(source, 0);

    const heap = new MinHeap();
    heap.push(0, source);
    instr.heapOps++;

    while (!heap.isEmpty()) {
        const top = heap.pop()!;
        const d_u = top.key;
        const u = top.node;
        instr.heapOps++;
        const cur = dist.get(u) ?? INF;
        if (d_u > cur) continue;

        for (const [v, w] of graph.get(u) ?? []) {
            instr.relaxations++;
            const alt = d_u + w;
            const dv = dist.get(v) ?? INF;
            if (alt < dv) {
                dist.set(v, alt);
                heap.push(alt, v);
                instr.heapOps++;
            }
        }
    }
    return dist;
}

/* ----------------------------
   DataStructureD (heap + map)
   ---------------------------- */
class DataStructureD {
    private heap = new MinHeap();
    private best: Map<NodeId, number> = new Map();
    private M: number;
    private BUpper: number;
    private blockSize: number;

    constructor(M: number, BUpper: number, blockSize?: number) {
        this.M = Math.max(1, M);
        this.BUpper = BUpper;
        this.blockSize = blockSize ?? Math.max(1, Math.floor(this.M / 8));
    }

    insert(v: NodeId, key: number) {
        const prev = this.best.get(v);
        if (prev === undefined || key < prev) {
            this.best.set(v, key);
            this.heap.push(key, v);
        }
    }

    batchPrepend(iterablePairs: Iterable<[NodeId, number]>) {
        for (const [v, key] of iterablePairs) this.insert(v, key);
    }

    private cleanup() {
        while (!this.heap.isEmpty()) {
            const top = this.heap.peek()!;
            const node = top.node;
            const key = top.key;
            const cur = this.best.get(node);
            if (cur === undefined || cur !== key) {
                this.heap.pop(); // stale
            } else break;
        }
    }

    empty(): boolean {
        this.cleanup();
        return this.heap.isEmpty();
    }

    pull(): [number, Set<NodeId>] {
        this.cleanup();
        if (this.heap.isEmpty()) throw new Error("pull from empty D");
        const Bi = this.heap.peek()!.key;
        const Si = new Set<NodeId>();
        while (!this.heap.isEmpty() && Si.size < this.blockSize) {
            const top = this.heap.pop()!;
            const key = top.key;
            const v = top.node;
            const cur = this.best.get(v);
            if (cur !== undefined && cur === key) {
                Si.add(v);
                this.best.delete(v);
            }
        }
        return [Bi, Si];
    }
}

/* ----------------------------
   findPivots (bounded BF-like)
   ---------------------------- */
function findPivots(
    graph: Graph,
    dist: Map<NodeId, number>,
    S: Set<NodeId>,
    B: number,
    n: number,
    k_steps: number,
    p_limit: number,
    instr?: Instrument
): [Set<NodeId>, Set<NodeId>] {
    instr = instr ?? new Instrument();

    const S_filtered: NodeId[] = [];
    for (const v of S) {
        const dv = dist.get(v) ?? INF;
        if (dv < B) S_filtered.push(v);
    }

    let P: Set<NodeId>;
    if (S_filtered.length === 0) {
        // fallback: choose up to p_limit arbitrary from S
        P = new Set<NodeId>();
        if (S.size) {
            let count = 0;
            for (const v of S) {
                P.add(v);
                if (++count >= Math.max(1, Math.min(S.size, p_limit))) break;
            }
        }
    } else {
        S_filtered.sort((a, b) => (dist.get(a) ?? INF) - (dist.get(b) ?? INF));
        const take = Math.max(1, Math.min(S_filtered.length, p_limit));
        P = new Set(S_filtered.slice(0, take));
    }

    const source_frontier = P.size ? new Set(P) : new Set(S);
    const discovered = new Set(source_frontier);
    let frontier = new Set(source_frontier);

    for (let step = 0; step < Math.max(1, k_steps); step++) {
        if (!frontier.size) break;
        const nextFront = new Set<NodeId>();
        for (const u of frontier) {
            const du = dist.get(u) ?? INF;
            if (du >= B) continue;
            for (const [v, w] of graph.get(u) ?? []) {
                instr.relaxations++;
                const nd = du + w;
                if (nd < B && !discovered.has(v)) {
                    discovered.add(v);
                    nextFront.add(v);
                }
            }
        }
        frontier = nextFront;
    }

    const W = new Set(discovered);
    if (!P.size && S.size) P.add(S.values().next().value ?? 0);
    return [P, W];
}

/* ----------------------------
   basecase (Dijkstra-like limited)
   ---------------------------- */
function basecase(
    graph: Graph,
    dist: Map<NodeId, number>,
    B: number,
    S: Set<NodeId>,
    k: number,
    instr?: Instrument
): [number, Set<NodeId>] {
    instr = instr ?? new Instrument();
    if (!S.size) return [B, new Set()];

    // choose x with smallest dist
    let x: NodeId | null = null;
    let bestd = INF;
    for (const v of S) {
        const dv = dist.get(v) ?? INF;
        if (dv < bestd) { bestd = dv; x = v; }
    }
    if (x === null) return [B, new Set()];

    const heap = new MinHeap();
    const start_d = dist.get(x) ?? INF;
    heap.push(start_d, x);
    instr.heapOps++;

    const Uo = new Set<NodeId>();

    while (!heap.isEmpty() && Uo.size < (k + 1)) {
        const top = heap.pop()!;
        instr.heapOps++;
        const d_u = top.key;
        const u = top.node;
        const cur = dist.get(u) ?? INF;
        if (d_u > cur) continue;
        if (!Uo.has(u)) Uo.add(u);

        for (const [v, w] of graph.get(u) ?? []) {
            instr.relaxations++;
            const newd = (dist.get(u) ?? INF) + w;
            const dv = dist.get(v) ?? INF;
            if (newd < dv && newd < B) {
                dist.set(v, newd);
                heap.push(newd, v);
                instr.heapOps++;
            }
        }
    }

    if (Uo.size <= k) return [B, Uo];

    const finiteDists: number[] = [];
    for (const v of Uo) {
        const dv = dist.get(v) ?? INF;
        if (Number.isFinite(dv)) finiteDists.push(dv);
    }
    if (!finiteDists.length) return [B, new Set()];
    const maxd = Math.max(...finiteDists);
    const Ufiltered = new Set(Array.from(Uo).filter(v => (dist.get(v) ?? INF) < maxd));
    return [maxd, Ufiltered];
}

/* ----------------------------
   bmssp (recursive)
   ---------------------------- */
function bmssp(
    graph: Graph,
    dist: Map<NodeId, number>,
    edges: Edge[],
    l: number,
    B: number,
    S: Set<NodeId>,
    n: number,
    instr?: Instrument
): [number, Set<NodeId>] {
    instr = instr ?? new Instrument();

    // parameter heuristics
    let t_param: number, k_param: number;
    if (n <= 2) { t_param = 1; k_param = 2; }
    else {
        const ln = Math.log(Math.max(3, n));
        t_param = Math.max(1, Math.round(Math.pow(ln, 2.0 / 3.0)));
        k_param = Math.max(2, Math.round(Math.pow(ln, 1.0 / 3.0)));
    }

    if (l <= 0) {
        if (!S.size) return [B, new Set()];
        return basecase(graph, dist, B, S, k_param, instr);
    }

    const p_limit = Math.max(1, 1 << Math.min(10, t_param));
    const k_steps = Math.max(1, k_param);

    const [P, W] = findPivots(graph, dist, S, B, n, k_steps, p_limit, instr);

    const M = 1 << Math.max(0, (l - 1) * t_param);
    const D = new DataStructureD(M, B, Math.max(1, Math.min(P.size || 1, 64)));

    for (const x of P) {
        D.insert(x, dist.get(x) ?? INF);
    }

    const B_prime_initial = (P.size ? Math.min(...Array.from(P).map(x => dist.get(x) ?? INF)) : B);

    const U = new Set<NodeId>();
    const B_prime_sub_values: number[] = [];

    let loop_guard = 0;
    const limit = k_param * (1 << (l * Math.max(1, t_param)));

    while (U.size < limit && !D.empty()) {
        loop_guard++;
        if (loop_guard > 20000) break;

        let Bi: number, Si: Set<NodeId>;
        try {
            [Bi, Si] = D.pull();
        } catch (e) {
            break;
        }

        const [B_prime_sub, Ui] = bmssp(graph, dist, edges, l - 1, Bi, Si, n, instr);
        B_prime_sub_values.push(B_prime_sub);
        for (const u of Ui) U.add(u);

        // K_for_batch as Set of tuples -> represent as Map to avoid duplicates
        const K_for_batchArr: Array<[NodeId, number]> = [];

        for (const u of Ui) {
            const du = dist.get(u) ?? INF;
            if (!Number.isFinite(du)) continue;
            for (const [v, w_uv] of graph.get(u) ?? []) {
                instr.relaxations++;
                const newd = du + w_uv;
                const dv = dist.get(v) ?? INF;
                if (newd <= dv) {
                    dist.set(v, newd);
                    if (Bi <= newd && newd < B) {
                        D.insert(v, newd);
                    } else if (B_prime_sub <= newd && newd < Bi) {
                        K_for_batchArr.push([v, newd]);
                    }
                }
            }
        }

        for (const x of Si) {
            const dx = dist.get(x) ?? INF;
            if (B_prime_sub <= dx && dx < Bi) K_for_batchArr.push([x, dx]);
        }

        if (K_for_batchArr.length) D.batchPrepend(K_for_batchArr);
    }

    const B_prime_final = (B_prime_sub_values.length ? Math.min(B_prime_initial, ...B_prime_sub_values) : B_prime_initial);

    const U_final = new Set(U);
    for (const x of W) {
        if ((dist.get(x) ?? INF) < B_prime_final) U_final.add(x);
    }
    return [B_prime_final, U_final];
}

/* ----------------------------
   Test harness (runSingleTest)
   ---------------------------- */
function runSingleTest(n: number, m: number, seed = 0, source = 0) {
    console.log(`Generating graph: n=${n}, m=${m}, seed=${seed}`);
    const [graph, edges] = generateSparseDirectedGraph(n, m, 100.0, seed);
    const avgDeg = Array.from(graph.values()).reduce((s, arr) => s + arr.length, 0) / n;
    console.log(`Graph generated. avg out-degree ≈ ${avgDeg.toFixed(3)}`);

    // Dijkstra
    const instr_dij = new Instrument();
    const t0 = Date.now();
    const dist_dij = dijkstra(graph, source, instr_dij);
    const t1 = Date.now();
    console.log(`Dijkstra: time=${((t1 - t0) / 1000).toFixed(6)}s, relaxations=${instr_dij.relaxations}, heap_ops=${instr_dij.heapOps}, reachable=${Array.from(dist_dij.values()).filter(v => Number.isFinite(v)).length}`);

    // BMSSP
    const dist_bm: Map<NodeId, number> = new Map();
    for (const k of graph.keys()) dist_bm.set(k, INF);
    dist_bm.set(source, 0);
    const instr_bm = new Instrument();
    let l: number;
    if (n <= 2) l = 1;
    else {
        const t_guess = Math.max(1, Math.round(Math.pow(Math.log(Math.max(3, n)), 2.0 / 3.0)));
        l = Math.max(1, Math.max(1, Math.round(Math.log(Math.max(3, n)) / t_guess)));
    }
    console.log(`BMSSP params: top-level l=${l}`);

    const t2 = Date.now();
    const [Bp, U_final] = bmssp(graph, dist_bm, edges, l, INF, new Set([source]), n, instr_bm);
    const t3 = Date.now();
    console.log(`BMSSP: time=${((t3 - t2) / 1000).toFixed(6)}s, relaxations=${instr_bm.relaxations}, reachable=${Array.from(dist_bm.values()).filter(v => Number.isFinite(v)).length}, B'=${Bp}, |U_final|=${U_final.size}`);

    const diffs: number[] = [];
    for (const v of graph.keys()) {
        const dv = dist_dij.get(v) ?? INF;
        const db = dist_bm.get(v) ?? INF;
        if (Number.isFinite(dv) && Number.isFinite(db)) diffs.push(Math.abs(dv - db));
    }
    const maxDiff = diffs.length ? Math.max(...diffs) : 0;
    console.log(`Distance agreement (max abs diff on commonly reachable nodes): ${maxDiff.toExponential(6)}`);
    return {
        n, m, seed,
        dijkstra_time: (t1 - t0) / 1000,
        dijkstra_relax: instr_dij.relaxations,
        bmssp_time: (t3 - t2) / 1000,
        bmssp_relax: instr_bm.relaxations,
        dijkstra_reachable: Array.from(dist_dij.values()).filter(v => Number.isFinite(v)).length,
        bmssp_reachable: Array.from(dist_bm.values()).filter(v => Number.isFinite(v)).length,
        max_diff: maxDiff
    };
}

/* ----------------------------
   CLI parsing
   ---------------------------- */
function parseArgs(argv: string[]) {
    let n = 200000, m = 800000, seed = 0;
    for (let i = 2; i < argv.length; i++) {
        const a = argv[i];
        if ((a === '-n' || a === '--nodes') && i + 1 < argv.length) { n = parseInt(argv[++i] ?? '', 10); }
        else if ((a === '-m' || a === '--edges') && i + 1 < argv.length) { m = parseInt(argv[++i] ?? '', 10); }
        else if ((a === '-s' || a === '--seed') && i + 1 < argv.length) { seed = parseInt(argv[++i] ?? '', 10); }
    }
    return { n, m, seed };
}

const { n, m, seed } = parseArgs(process.argv);
// For quicker local test, you may want to reduce defaults:
// const testDefaults = { n: 2000, m: 8000, seed: 42 };
runSingleTest(n, m, seed, 0);
