// bmssp.ts
// TypeScript Node.js port of the BMSSP algorithm + Dijkstra baseline
// Requires Node >= 16 with ts-node or transpiled via tsc
function generateSparseDirectedGraph(n, m, maxW = 100.0, seed) {
    const rand = seed !== undefined ? mulberry32(seed) : Math.random;
    const graph = new Map();
    for (let i = 0; i < n; i++)
        graph.set(i, []);
    const edges = [];
    // weak backbone
    for (let i = 1; i < n; i++) {
        const u = Math.floor(rand() * i);
        const w = rand() * maxW + 1.0;
        graph.get(u).push([i, w]);
        edges.push([u, i, w]);
    }
    const remaining = Math.max(0, m - (n - 1));
    for (let i = 0; i < remaining; i++) {
        const u = Math.floor(rand() * n);
        const v = Math.floor(rand() * n);
        const w = rand() * maxW + 1.0;
        graph.get(u).push([v, w]);
        edges.push([u, v, w]);
    }
    return [graph, edges];
}
// simple seeded RNG
function mulberry32(seed) {
    return function () {
        let t = (seed += 0x6d2b79f5);
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}
// ---------------- Instrumentation ----------------
class Instrument {
    relaxations = 0;
    heapOps = 0;
    reset() {
        this.relaxations = 0;
        this.heapOps = 0;
    }
}
// ---------------- Dijkstra ----------------
import { MinPriorityQueue } from "@datastructures-js/priority-queue";
function dijkstra(graph, source, instr) {
    instr ??= new Instrument();
    const dist = new Map();
    for (const v of graph.keys())
        dist.set(v, Infinity);
    dist.set(source, 0);
    const pq = new MinPriorityQueue((x) => x.key);
    pq.enqueue({ node: source, key: 0 });
    instr.heapOps++;
    while (!pq.isEmpty()) {
        const { node: u, key: dU } = pq.dequeue() ?? { node: 0, key: 0 };
        instr.heapOps++;
        if (dU > (dist.get(u) ?? Infinity))
            continue;
        for (const [v, w] of graph.get(u) ?? []) {
            instr.relaxations++;
            const alt = dU + w;
            if (alt < (dist.get(v) ?? Infinity)) {
                dist.set(v, alt);
                pq.enqueue({ node: v, key: alt });
                instr.heapOps++;
            }
        }
    }
    return dist;
}
// ---------------- DataStructureD ----------------
class DataStructureD {
    heap;
    best;
    blockSize;
    M;
    BUpper;
    constructor(M, BUpper, blockSize) {
        this.M = Math.max(1, M);
        this.BUpper = BUpper;
        this.blockSize = blockSize ?? Math.max(1, Math.floor(this.M / 8));
        this.best = new Map();
        this.heap = new MinPriorityQueue((x) => x.key);
    }
    insert(v, key) {
        const prev = this.best.get(v);
        if (prev === undefined || key < prev) {
            this.best.set(v, key);
            this.heap.enqueue({ node: v, key });
        }
    }
    batchPrepend(items) {
        for (const [v, key] of items)
            this.insert(v, key);
    }
    cleanup() {
        while (!this.heap.isEmpty()) {
            const top = this.heap.front();
            if (top) {
                const v = top.node;
                const k = top.key;
                if (this.best.get(v) !== k) {
                    this.heap.dequeue();
                }
                else {
                    break;
                }
            }
        }
    }
    empty() {
        this.cleanup();
        return this.heap.isEmpty();
    }
    pull() {
        this.cleanup();
        if (this.heap.isEmpty())
            throw new Error("pull from empty D");
        const Bi = this.heap.front().key;
        const Si = new Set();
        while (!this.heap.isEmpty() && Si.size < this.blockSize) {
            const { node: v, key } = this.heap.dequeue() ?? { node: 0, key: 0 };
            if (this.best.get(v) === key) {
                Si.add(v);
                this.best.delete(v);
            }
        }
        return [Bi, Si];
    }
}
// ---------------- BMSSP ----------------
// due to space, showing main outline
// in Python you had recursive bmssp(D, Z, P, Q, ...)
// porting logic: sets -> Set<number>, dicts -> Map<number, number>
function bmssp(graph, n, m, source, BUpper = 1000, delta = 8) {
    const dist = new Map();
    for (const v of graph.keys())
        dist.set(v, Infinity);
    dist.set(source, 0);
    const D = new DataStructureD(m, BUpper, Math.max(1, Math.floor(m / delta)));
    D.insert(source, 0);
    while (!D.empty()) {
        const [Bi, Si] = D.pull();
        for (const u of Si) {
            for (const [v, w] of graph.get(u) ?? []) {
                const alt = (dist.get(u) ?? Infinity) + w;
                if (alt < (dist.get(v) ?? Infinity)) {
                    dist.set(v, alt);
                    D.insert(v, alt);
                }
            }
        }
    }
    return dist;
}
// ---------------- Benchmark Harness ----------------
const n = 2000;
const m = 8000;
const [graph, edges] = generateSparseDirectedGraph(n, m, 100, 42);
const source = 0;
const instr = new Instrument();
console.time("Dijkstra");
const dist1 = dijkstra(graph, source, instr);
console.timeEnd("Dijkstra");
console.log(`Dijkstra relax=${instr.relaxations}, heapOps=${instr.heapOps}, reachable=${[...dist1.values()].filter((d) => d < Infinity).length}`);
console.time("BMSSP Start");
const dist2 = bmssp(graph, n, m, source);
console.timeEnd("BMSSP");
console.log(`BMSSP reachable=${[...dist2.values()].filter((d) => d < Infinity).length}`);
console.time("BMSSP End");
//# sourceMappingURL=app.js.map