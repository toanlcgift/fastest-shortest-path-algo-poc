// bmssp_full_impl.dart
// Dart 2.12+ (null-safety) single-file port of the Python BMSSP practical implementation.
//
// Provides: graph generator, Dijkstra, DataStructureD, FIND_PIVOTS, BASECASE, BMSSP recursion,
// instrumentation, and a simple test harness / CLI.

import 'dart:collection';
import 'dart:io';
import 'dart:math';

// Type aliases
typedef Node = int;
typedef Weight = double;
typedef Edge = List<dynamic>; // [u, v, w]
// Graph: Map<Node, List<Pair(Node, Weight)>> -- We'll use List<List<Object>> pairs
// But better typed: Map<int, List<_Neighbor>>

class _Neighbor {
  final int v;
  final double w;
  _Neighbor(this.v, this.w);
}

// ---------------------------
// Utilities & Graph generator
// ---------------------------
class GraphAndEdges {
  final Map<int, List<_Neighbor>> graph;
  final List<Edge> edges;
  GraphAndEdges(this.graph, this.edges);
}

GraphAndEdges generateSparseDirectedGraph(int n, int m, {double maxW = 100.0, int? seed}) {
  final rng = (seed != null) ? Random(seed) : Random();
  final graph = <int, List<_Neighbor>>{};
  for (var i = 0; i < n; ++i) graph[i] = <_Neighbor>[];
  final edges = <Edge>[];

  // weak backbone to avoid isolated nodes
  for (var i = 1; i < n; ++i) {
    var u = rng.nextInt(i);
    var w = rng.nextDouble() * (maxW - 1.0) + 1.0;
    graph[u]!.add(_Neighbor(i, w));
    edges.add([u, i, w]);
  }
  var remaining = max(0, m - (n - 1));
  for (var i = 0; i < remaining; ++i) {
    var u = rng.nextInt(n);
    var v = rng.nextInt(n);
    var w = rng.nextDouble() * (maxW - 1.0) + 1.0;
    graph[u]!.add(_Neighbor(v, w));
    edges.add([u, v, w]);
  }
  return GraphAndEdges(graph, edges);
}

// ---------------------------
// Instrumentation
// ---------------------------
class Instrument {
  int relaxations = 0;
  int heapOps = 0;
  void reset() {
    relaxations = 0;
    heapOps = 0;
  }
}

// ---------------------------
// Simple generic min-heap / priority queue
// ---------------------------
// We'll implement a simple binary heap keyed by double (priority).
class PriorityQueue<T> {
  final List<_PQEntry<T>> _data = [];
  int get length => _data.length;
  bool get isEmpty => _data.isEmpty;

  void add(T value, double priority) {
    _data.add(_PQEntry(value, priority));
    _siftUp(_data.length - 1);
  }

  // returns (value, priority)
  _PQEntry<T> pop() {
    if (_data.isEmpty) throw StateError('pop from empty priority queue');
    final top = _data.first;
    final last = _data.removeLast();
    if (_data.isNotEmpty) {
      _data[0] = last;
      _siftDown(0);
    }
    return top;
  }

  _PQEntry<T> peek() {
    if (_data.isEmpty) throw StateError('peek from empty priority queue');
    return _data.first;
  }

  void _siftUp(int idx) {
    while (idx > 0) {
      final parent = (idx - 1) >> 1;
      if (_data[idx].priority < _data[parent].priority) {
        final tmp = _data[idx];
        _data[idx] = _data[parent];
        _data[parent] = tmp;
        idx = parent;
      } else {
        break;
      }
    }
  }

  void _siftDown(int idx) {
    final n = _data.length;
    while (true) {
      final left = idx * 2 + 1;
      final right = left + 1;
      var smallest = idx;
      if (left < n && _data[left].priority < _data[smallest].priority) smallest = left;
      if (right < n && _data[right].priority < _data[smallest].priority) smallest = right;
      if (smallest == idx) break;
      final tmp = _data[idx];
      _data[idx] = _data[smallest];
      _data[smallest] = tmp;
      idx = smallest;
    }
  }
}

class _PQEntry<T> {
  final T value;
  final double priority;
  _PQEntry(this.value, this.priority);
}

// ---------------------------
// Dijkstra
// ---------------------------
Map<int, double> dijkstra(Map<int, List<_Neighbor>> graph, int source, [Instrument? instr]) {
  instr ??= Instrument();
  final dist = <int, double>{};
  for (final k in graph.keys) dist[k] = double.infinity;
  dist[source] = 0.0;
  final pq = PriorityQueue<int>();
  pq.add(source, 0.0);
  instr.heapOps++;

  while (!pq.isEmpty) {
    final entry = pq.pop();
    instr.heapOps++;
    final dU = entry.priority;
    final u = entry.value;
    final cur = dist[u] ?? double.infinity;
    if (dU > cur) continue;
    final adj = graph[u]!;
    for (final nb in adj) {
      instr.relaxations++;
      final v = nb.v;
      final w = nb.w;
      final alt = dU + w;
      final dv = dist[v] ?? double.infinity;
      if (alt < dv) {
        dist[v] = alt;
        pq.add(v, alt);
        instr.heapOps++;
      }
    }
  }
  return dist;
}

// ---------------------------
// DataStructure D (practical)
// ---------------------------
class DataStructureD {
  final PriorityQueue<int> _heap = PriorityQueue<int>();
  final Map<int, double> _best = {};
  final int M;
  final double BUpper;
  final int blockSize;

  DataStructureD(int M, double BUpper, {int? blockSize})
      : M = M < 1 ? 1 : M,
        BUpper = BUpper,
        blockSize = blockSize ?? max(1, (M < 1 ? 1 : M) ~/ 8);

  void insert(int v, double key) {
    final prev = _best[v];
    if (prev == null || key < prev) {
      _best[v] = key;
      _heap.add(v, key);
    }
  }

  void batchPrepend(Iterable<List<dynamic>> iterablePairs) {
    for (final p in iterablePairs) {
      final v = p[0] as int;
      final key = p[1] as double;
      insert(v, key);
    }
  }

  void _cleanup() {
    // We can't directly inspect heap internals, so emulate lazy drain:
    // Pop until top matches _best (this requires peeking).
    // Our PQ exposes peek via peek; catch error if empty.
    try {
      while (_heap.length > 0) {
        final top = _heap.peek();
        final v = top.value;
        final key = top.priority;
        final bestVal = _best[v];
        if (bestVal == null || bestVal != key) {
          _heap.pop();
        } else {
          break;
        }
      }
    } catch (e) {
      // empty, ignore
    }
  }

  bool isEmpty() {
    _cleanup();
    return _heap.isEmpty;
  }

  // pull returns Bi and Si set of up to blockSize nodes
  Map<String, dynamic> pull() {
    _cleanup();
    if (_heap.isEmpty) throw StateError('pull from empty D');
    final Bi = _heap.peek().priority;
    final Si = <int>{};
    while (!_heap.isEmpty && Si.length < blockSize) {
      final entry = _heap.pop();
      final key = entry.priority;
      final v = entry.value;
      final bestVal = _best[v];
      if (bestVal != null && bestVal == key) {
        Si.add(v);
        _best.remove(v);
      }
    }
    return {'Bi': Bi, 'Si': Si};
  }
}

// ---------------------------
// FIND_PIVOTS (bounded Bellman-Ford-like)
// ---------------------------
Map<String, Set<int>> findPivots(
    Map<int, List<_Neighbor>> graph,
    Map<int, double> dist,
    Set<int> S,
    double B,
    int n,
    int kSteps,
    int pLimit,
    [Instrument? instr]) {
  instr ??= Instrument();
  final sFiltered = S.where((v) => (dist[v] ?? double.infinity) < B).toList();

  Set<int> P;
  if (sFiltered.isEmpty) {
    if (S.isEmpty) {
      P = <int>{};
    } else {
      final take = max(1, min(S.length, pLimit));
      P = S.take(take).toSet();
    }
  } else {
    sFiltered.sort((a, b) => (dist[a] ?? double.infinity).compareTo(dist[b] ?? double.infinity));
    final take = max(1, min(sFiltered.length, pLimit));
    P = sFiltered.take(take).toSet();
  }

  final sourceFrontier = P.isNotEmpty ? P.toSet() : S.toSet();
  final discovered = sourceFrontier.toSet();
  var frontier = sourceFrontier.toSet();

  for (var step = 0; step < max(1, kSteps); ++step) {
    if (frontier.isEmpty) break;
    final nextFront = <int>{};
    for (final u in frontier) {
      final du = dist[u] ?? double.infinity;
      if (du >= B) continue;
      for (final nb in graph[u] ?? []) {
        instr.relaxations++;
        final nd = du + nb.w;
        if (nd < B && !discovered.contains(nb.v)) {
          discovered.add(nb.v);
          nextFront.add(nb.v);
        }
      }
    }
    frontier = nextFront;
  }

  final W = discovered.toSet();
  if (P.isEmpty && S.isNotEmpty) {
    P = {S.first};
  }
  return {'P': P, 'W': W};
}

// ---------------------------
// BASECASE (Dijkstra-like limited)
// ---------------------------
Map<String, dynamic> basecase(
    Map<int, List<_Neighbor>> graph, Map<int, double> dist, double B, Set<int> S, int k, [Instrument? instr]) {
  instr ??= Instrument();
  if (S.isEmpty) return {'Bprime': B, 'Uo': <int>{}};

  // choose x in S with smallest dist
  int x = S.first;
  double bestd = dist[x] ?? double.infinity;
  for (final v in S) {
    final dv = dist[v] ?? double.infinity;
    if (dv < bestd) {
      bestd = dv;
      x = v;
    }
  }

  final heap = PriorityQueue<int>();
  final startD = dist[x] ?? double.infinity;
  heap.add(x, startD);
  instr.heapOps++;

  final Uo = <int>{};

  while (!heap.isEmpty && Uo.length < (k + 1)) {
    final entry = heap.pop();
    instr.heapOps++;
    final dU = entry.priority;
    final u = entry.value;
    final cur = dist[u] ?? double.infinity;
    if (dU > cur) continue;
    if (!Uo.contains(u)) Uo.add(u);
    for (final nb in graph[u] ?? []) {
      instr.relaxations++;
      final newd = (dist[u] ?? double.infinity) + nb.w;
      final dv = dist[nb.v] ?? double.infinity;
      if (newd < dv && newd < B) {
        dist[nb.v] = newd;
        heap.add(nb.v, newd);
        instr.heapOps++;
      }
    }
  }

  if (Uo.length <= k) {
    return {'Bprime': B, 'Uo': Uo};
  } else {
    final finiteDists = Uo.where((v) => (dist[v] ?? double.infinity).isFinite).map((v) => dist[v]!).toList();
    if (finiteDists.isEmpty) return {'Bprime': B, 'Uo': <int>{}};
    final maxd = finiteDists.reduce(max);
    final uFiltered = Uo.where((v) => (dist[v] ?? double.infinity) < maxd).toSet();
    return {'Bprime': maxd, 'Uo': uFiltered};
  }
}

// ---------------------------
// BMSSP (practical recursive)
// ---------------------------
Map<String, dynamic> bmssp(
    Map<int, List<_Neighbor>> graph,
    Map<int, double> dist,
    List<Edge> edges,
    int l,
    double B,
    Set<int> S,
    int n,
    [Instrument? instr]) {
  instr ??= Instrument();

  int tParam, kParam;
  if (n <= 2) {
    tParam = 1;
    kParam = 2;
  } else {
    final ln = log(max(3, n));
    tParam = max(1, (ln.pow(2.0 / 3.0)).round());
    kParam = max(2, (ln.pow(1.0 / 3.0)).round());
  }

  if (l <= 0) {
    if (S.isEmpty) return {'Bprime': B, 'Ufinal': <int>{}};
    final bc = basecase(graph, dist, B, S, kParam, instr);
    return {'Bprime': bc['Bprime'] as double, 'Ufinal': bc['Uo'] as Set<int>};
  }

  final pLimit = max(1, 1 << min(10, tParam));
  final kSteps = max(1, kParam);

  final piv = findPivots(graph, dist, S, B, n, kSteps, pLimit, instr);
  var P = (piv['P'] as Set<int>).toSet();
  final W = (piv['W'] as Set<int>).toSet();

  final M = 1 << max(0, (l - 1) * tParam);
  final D = DataStructureD(M, B, blockSize: max(1, min(P.isEmpty ? 1 : P.length, 64)));

  for (final x in P) {
    final dx = dist[x] ?? double.infinity;
    D.insert(x, dx);
  }

  double bPrimeInitial = B;
  if (P.isNotEmpty) {
    double minv = double.infinity;
    for (final x in P) {
      final dx = dist[x] ?? double.infinity;
      if (dx < minv) minv = dx;
    }
    if (minv.isFinite) bPrimeInitial = minv;
  }

  final U = <int>{};
  final bPrimeSubs = <double>[];

  var loopGuard = 0;
  final limit = kParam * (1 << (l * max(1, tParam)));

  while (U.length < limit && !D.isEmpty()) {
    loopGuard++;
    if (loopGuard > 20000) break;
    Map<String, dynamic> pulled;
    try {
      pulled = D.pull();
    } catch (e) {
      break;
    }
    final Bi = pulled['Bi'] as double;
    final Si = pulled['Si'] as Set<int>;

    final sub = bmssp(graph, dist, edges, l - 1, Bi, Si, n, instr);
    final bPrimeSub = sub['Bprime'] as double;
    final Ui = (sub['Ufinal'] as Set<int>).toSet();
    bPrimeSubs.add(bPrimeSub);
    U.addAll(Ui);

    final batch = <List<dynamic>>[];

    for (final u in Ui) {
      final du = dist[u] ?? double.infinity;
      if (!du.isFinite) continue;
      for (final nb in graph[u] ?? []) {
        instr.relaxations++;
        final newd = du + nb.w;
        final dv = dist[nb.v] ?? double.infinity;
        if (newd <= dv) {
          dist[nb.v] = newd;
          if (Bi <= newd && newd < B) {
            D.insert(nb.v, newd);
          } else if (bPrimeSub <= newd && newd < Bi) {
            batch.add([nb.v, newd]);
          }
        }
      }
    }

    for (final x in Si) {
      final dx = dist[x] ?? double.infinity;
      if (bPrimeSub <= dx && dx < Bi) {
        batch.add([x, dx]);
      }
    }

    if (batch.isNotEmpty) D.batchPrepend(batch);
  }

  double bPrimeFinal = bPrimeInitial;
  if (bPrimeSubs.isNotEmpty) {
    final minSub = bPrimeSubs.reduce(min);
    bPrimeFinal = min(bPrimeInitial, minSub);
  }

  final uFinal = U.toSet();
  for (final x in W) {
    final dx = dist[x] ?? double.infinity;
    if (dx < bPrimeFinal) uFinal.add(x);
  }

  return {'Bprime': bPrimeFinal, 'Ufinal': uFinal};
}

// ---------------------------
// Test harness
// ---------------------------
class TestResult {
  int n = 0;
  int m = 0;
  int seed = 0;
  double dijkstraTime = 0.0;
  int dijkstraRelax = 0;
  double bmsspTime = 0.0;
  int bmsspRelax = 0;
  int dijkstraReachable = 0;
  int bmsspReachable = 0;
  double maxDiff = 0.0;
}

TestResult runSingleTest(int n, int m, {int seed = 0, int source = 0}) {
  print('Generating graph: n=$n, m=$m, seed=$seed');
  final ge = generateSparseDirectedGraph(n, m, seed: seed);
  final graph = ge.graph;
  final edges = ge.edges;
  final avgDeg = graph.values.fold(0, (p, e) => p + e.length) / n;
  print('Graph generated. avg out-degree ≈ ${avgDeg.toStringAsFixed(3)}');

  // Dijkstra timing
  final instrDij = Instrument();
  final t0 = DateTime.now().millisecondsSinceEpoch;
  final distDij = dijkstra(graph, source, instrDij);
  final t1 = DateTime.now().millisecondsSinceEpoch;
  final dijTime = (t1 - t0) / 1000.0;
  final dijReachable = distDij.values.where((v) => v.isFinite).length;
  print('Dijkstra: time=${dijTime.toStringAsFixed(6)}s, relaxations=${instrDij.relaxations}, heap_ops=${instrDij.heapOps}, reachable=$dijReachable');

  // BMSSP practical
  final distBm = <int, double>{};
  for (final k in graph.keys) distBm[k] = double.infinity;
  distBm[source] = 0.0;
  final instrBm = Instrument();

  int l;
  if (n <= 2) l = 1;
  else {
    final ln = log(max(3, n));
    final tGuess = max(1, (ln.pow(2.0 / 3.0)).round());
    l = max(1, max(1, (ln / tGuess).round()));
  }
  print('BMSSP params: top-level l=$l');

  final t2 = DateTime.now().millisecondsSinceEpoch;
  final res = bmssp(graph, distBm, edges, l, double.infinity, {source}, n, instrBm);
  final t3 = DateTime.now().millisecondsSinceEpoch;
  final bmTime = (t3 - t2) / 1000.0;
  final bmReachable = distBm.values.where((v) => v.isFinite).length;
  final Bp = res['Bprime'] as double;
  final Ufinal = res['Ufinal'] as Set<int>;
  print('BMSSP: time=${bmTime.toStringAsFixed(6)}s, relaxations=${instrBm.relaxations}, reachable=$bmReachable, B\'=$Bp, |U_final|=${Ufinal.length}');

  final diffs = <double>[];
  for (final v in graph.keys) {
    final dv = distDij[v] ?? double.infinity;
    final db = distBm[v] ?? double.infinity;
    if (dv.isFinite && db.isFinite) diffs.add((dv - db).abs());
  }
  final maxDiff = diffs.isEmpty ? 0.0 : diffs.reduce(max);
  print('Distance agreement (max abs diff on commonly reachable nodes): ${maxDiff.toStringAsExponential(6)}');

  final tr = TestResult()
    ..n = n
    ..m = m
    ..seed = seed
    ..dijkstraTime = dijTime
    ..dijkstraRelax = instrDij.relaxations
    ..bmsspTime = bmTime
    ..bmsspRelax = instrBm.relaxations
    ..dijkstraReachable = dijReachable
    ..bmsspReachable = bmReachable
    ..maxDiff = maxDiff;
  return tr;
}

// ---------------------------
// CLI main
// ---------------------------
void printUsage() {
  print('Usage: dart bmssp_full_impl.dart [-n nodes] [-m edges] [-s seed]');
}

void main(List<String> args) {
  int n = 200000;
  int m = 800000;
  int seed = 0;

  for (var i = 0; i < args.length; ++i) {
    final a = args[i];
    if ((a == '-n' || a == '--nodes') && i + 1 < args.length) {
      n = int.tryParse(args[++i]) ?? n;
    } else if ((a == '-m' || a == '--edges') && i + 1 < args.length) {
      m = int.tryParse(args[++i]) ?? m;
    } else if ((a == '-s' || a == '--seed') && i + 1 < args.length) {
      seed = int.tryParse(args[++i]) ?? seed;
    } else if (a == '-h' || a == '--help') {
      printUsage();
      return;
    }
  }

  // For interactive testing on small machines you may want to set smaller defaults:
  if (Platform.environment.containsKey('BMSSP_SMALL')) {
    n = 2000;
    m = 8000;
  }

  // run test
  try {
    runSingleTest(n, m, seed: seed, source: 0);
  } catch (e, st) {
    stderr.writeln('Error: $e\n$st');
    exit(2);
  }
}

// ---------------------------
// Extensions for math helpers
// ---------------------------
extension DoublePow on double {
  double pow(double p) => mathPow(this, p);
}
double mathPow(double a, double b) => powDouble(a, b);

double powDouble(double a, double b) {
  return math_pow(a, b).toDouble();
}

// Because dart:math.pow returns num and may be slow to cast, we use a wrapper
num math_pow(num a, num b) => pow(a, b);
