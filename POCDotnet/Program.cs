
public class Program
{
    public static void Main(string[] args)
    {
        var parser = new ArgumentParser();
        parser.AddArgument("-n", "--nodes", 200000, "nodes");
        parser.AddArgument("-m", "--edges", 800000, "edges");
        parser.AddArgument("-s", "--seed", 0, "seed");
        var parsedArgs = parser.Parse(args);

        int nodes = parser.GetValue<int>("-n");
        int edges = parser.GetValue<int>("-m");
        int seed = parser.GetValue<int>("-s");

        RunSingleTest(200000, 800000, 0);
    }

    public static void RunSingleTest(int n, int m, int seed)
    {
        Console.WriteLine($"Generating graph: n={n}, m={m}, seed={seed}");
        var (graph, edges) = GenerateSparseDirectedGraph(n, m, seed);
        double avgDeg = graph.Values.Average(adj => adj.Count);
        Console.WriteLine($"Graph generated. avg out-degree ≈ {avgDeg:F3}");

        var dist0 = graph.Keys.ToDictionary(v => v, v => double.PositiveInfinity);
        dist0[0] = 0.0;

        var instrDij = new Instrument();
        var t0 = DateTime.Now;
        var distDij = Dijkstra(graph, 0, instrDij);
        var t1 = DateTime.Now;
        Console.WriteLine($"Dijkstra: time={(t1 - t0).TotalSeconds:F6}s, relaxations={instrDij.Relaxations}, heap_ops={instrDij.HeapOps}, reachable={distDij.Values.Count(v => double.IsFinite(v))}");

        var distBm = graph.Keys.ToDictionary(v => v, v => double.PositiveInfinity);
        distBm[0] = 0.0;
        var instrBm = new Instrument();
        int l = n <= 2 ? 1 : Math.Max(1, (int)Math.Round(Math.Log(Math.Max(3, n)) / Math.Max(1, Math.Round(Math.Pow(Math.Log(Math.Max(3, n)), 2.0 / 3.0)))));
        Console.WriteLine($"BMSSP params: top-level l={l}");
        t0 = DateTime.Now;
        var (Bp, UFinal) = BMSSP(graph, distBm, edges, l, double.PositiveInfinity, new HashSet<int> { 0 }, n, instrBm);
        t1 = DateTime.Now;
        Console.WriteLine($"BMSSP: time={(t1 - t0).TotalSeconds:F6}s, relaxations={instrBm.Relaxations}, reachable={distBm.Values.Count(v => double.IsFinite(v))}, B'={Bp}, |U_final|={UFinal.Count}");

        var diffs = graph.Keys
            .Where(v => double.IsFinite(distDij[v]) && double.IsFinite(distBm[v]))
            .Select(v => Math.Abs(distDij[v] - distBm[v]))
            .ToList();
        double maxDiff = diffs.Any() ? diffs.Max() : 0.0;
        Console.WriteLine($"Distance agreement (max abs diff on commonly reachable nodes): {maxDiff:E6}");
    }

    public static (double, HashSet<int>) BMSSP(
        Dictionary<int, List<(int, double)>> graph,
        Dictionary<int, double> dist,
        List<(int, int, double)> edges,
        int l,
        double B,
        HashSet<int> S,
        int n,
        Instrument instr = null
    )
    {
        if (instr == null) instr = new Instrument();

        // Sensible parameter choices (heuristic approximations from the paper)  
        int tParam, kParam;
        if (n <= 2)
        {
            tParam = 1;
            kParam = 2;
        }
        else
        {
            tParam = Math.Max(1, (int)Math.Round(Math.Pow(Math.Log(Math.Max(3, n)), 2.0 / 3.0)));
            kParam = Math.Max(2, (int)Math.Round(Math.Pow(Math.Log(Math.Max(3, n)), 1.0 / 3.0)));
        }

        // Base case  
        if (l <= 0)
        {
            if (S.Count == 0) return (B, new HashSet<int>());
            return BaseCase(graph, dist, B, S, kParam, instr);
        }

        // FIND_PIVOTS: compute P, W  
        int pLimit = Math.Max(1, (int)Math.Pow(2, Math.Min(10, tParam)));
        int kSteps = Math.Max(1, kParam);
        var (P, W) = FindPivots(graph, dist, S, B, n, kSteps, pLimit, instr);

        // Data structure D initialization  
        int M = (int)Math.Pow(2, Math.Max(0, (l - 1) * tParam));
        var D = new DataStructureD(M, B, Math.Max(1, Math.Min(P.Count, 64)));

        foreach (var x in P)
        {
            D.Insert(x, dist.ContainsKey(x) ? dist[x] : double.PositiveInfinity);
        }

        double BPrimeInitial = P.Count > 0 ? P.Min(x => dist.ContainsKey(x) ? dist[x] : double.PositiveInfinity) : B;
        var U = new HashSet<int>();
        var BPrimeSubValues = new List<double>();

        int loopGuard = 0;
        int limit = kParam * (int)Math.Pow(2, l * Math.Max(1, tParam));

        while (U.Count < limit && !D.IsEmpty())
        {
            loopGuard++;
            if (loopGuard > 20000) break;

            double Bi;
            HashSet<int> Si;
            try
            {
                (Bi, Si) = D.Pull();
            }
            catch (InvalidOperationException)
            {
                break;
            }

            var (BPrimeSub, Ui) = BMSSP(graph, dist, edges, l - 1, Bi, Si, n, instr);
            BPrimeSubValues.Add(BPrimeSub);
            U.UnionWith(Ui);

            var KForBatch = new HashSet<(int, double)>();
            foreach (var u in Ui)
            {
                if (!dist.ContainsKey(u)) continue;
                double du = dist[u];
                foreach (var (v, wUv) in graph[u])
                {
                    instr.Relaxations++;
                    double newDist = du + wUv;
                    if (newDist <= (dist.ContainsKey(v) ? dist[v] : double.PositiveInfinity))
                    {
                        dist[v] = newDist;
                        if (Bi <= newDist && newDist < B)
                        {
                            D.Insert(v, newDist);
                        }
                        else if (BPrimeSub <= newDist && newDist < Bi)
                        {
                            KForBatch.Add((v, newDist));
                        }
                    }
                }
            }

            foreach (var x in Si)
            {
                if (dist.ContainsKey(x) && BPrimeSub <= dist[x] && dist[x] < Bi)
                {
                    KForBatch.Add((x, dist[x]));
                }
            }

            if (KForBatch.Count > 0)
            {
                D.BatchPrepend(KForBatch);
            }
        }

        double BPrimeFinal = BPrimeSubValues.Count > 0 ? Math.Min(BPrimeInitial, BPrimeSubValues.Min()) : BPrimeInitial;
        var UFinal = new HashSet<int>(U);

        foreach (var x in W)
        {
            if (dist.ContainsKey(x) && dist[x] < BPrimeFinal)
            {
                UFinal.Add(x);
            }
        }

        return (BPrimeFinal, UFinal);
    }

    public static (double, HashSet<int>) BaseCase(
        Dictionary<int, List<(int, double)>> graph,
        Dictionary<int, double> dist,
        double B,
        HashSet<int> S,
        int k,
        Instrument instr = null)
    {
        if (instr == null)
        {
            instr = new Instrument();
        }

        if (S == null || S.Count == 0)
        {
            return (B, new HashSet<int>());
        }

        // Choose source x in S with smallest dist  
        int x = S.OrderBy(v => dist.ContainsKey(v) ? dist[v] : double.PositiveInfinity).First();

        // Local heap  
        var heap = new SortedSet<(double, int)>(Comparer<(double, int)>.Create((a, b) =>
        {
            int cmp = a.Item1.CompareTo(b.Item1);
            return cmp != 0 ? cmp : a.Item2.CompareTo(b.Item2);
        }));

        double startD = dist.ContainsKey(x) ? dist[x] : double.PositiveInfinity;
        heap.Add((startD, x));
        instr.HeapOps++;

        var Uo = new HashSet<int>();
        var visited = new HashSet<int>();

        while (heap.Count > 0 && Uo.Count < (k + 1))
        {
            var (d_u, u) = heap.Min;
            heap.Remove(heap.Min);
            instr.HeapOps++;

            if (d_u > (dist.ContainsKey(u) ? dist[u] : double.PositiveInfinity))
            {
                continue;
            }

            // Mark 'u' complete for this basecase  
            if (!Uo.Contains(u))
            {
                Uo.Add(u);
            }

            if (graph.ContainsKey(u))
            {
                foreach (var (v, w) in graph[u])
                {
                    instr.Relaxations++;
                    double newD = (dist.ContainsKey(u) ? dist[u] : double.PositiveInfinity) + w;
                    if (newD < (dist.ContainsKey(v) ? dist[v] : double.PositiveInfinity) && newD < B)
                    {
                        dist[v] = newD;
                        heap.Add((newD, v));
                        instr.HeapOps++;
                    }
                }
            }
        }

        if (Uo.Count <= k)
        {
            return (B, Uo);
        }
        else
        {
            var finiteDists = Uo
                .Where(v => dist.ContainsKey(v) && !double.IsInfinity(dist[v]))
                .Select(v => dist[v])
                .ToList();

            if (finiteDists.Count == 0)
            {
                return (B, new HashSet<int>());
            }

            double maxD = finiteDists.Max();
            var UFiltered = new HashSet<int>(Uo.Where(v => dist.ContainsKey(v) && dist[v] < maxD));
            return (maxD, UFiltered);
        }
    }

    public static (HashSet<int>, HashSet<int>) FindPivots(
        Dictionary<int, List<(int, double)>> graph,
        Dictionary<int, double> dist,
        HashSet<int> S,
        double B,
        int n,
        int kSteps,
        int pLimit,
        Instrument? instrument = null
    )
    {
        if (instrument == null)
            instrument = new Instrument();
        // Filter S to those with dist < B  
        var SFiltered = S.Where(v => dist.GetValueOrDefault(v, double.PositiveInfinity) < B).ToList();

        // Choose pivots P — heuristic: smallest distances in SFiltered  
        HashSet<int> P;
        if (!SFiltered.Any())
        {
            // Fallback: choose up to pLimit arbitrary samples from S  
            P = S.Any() ? new HashSet<int>(S.Take(Math.Max(1, Math.Min(S.Count, pLimit)))) : new HashSet<int>();
        }
        else
        {
            SFiltered.Sort((v1, v2) => dist.GetValueOrDefault(v1, double.PositiveInfinity).CompareTo(dist.GetValueOrDefault(v2, double.PositiveInfinity)));
            P = new HashSet<int>(SFiltered.Take(Math.Max(1, Math.Min(SFiltered.Count, pLimit))));
        }

        // Bounded BF: start frontier from P (if P empty use S)  
        var sourceFrontier = P.Any() ? P : new HashSet<int>(S);
        var discovered = new HashSet<int>(sourceFrontier);
        var frontier = new HashSet<int>(sourceFrontier);

        // Perform relaxations but not set dist globally here; instead return W (discovered)  
        for (int i = 0; i < Math.Max(1, kSteps); i++)
        {
            if (!frontier.Any())
                break;

            var nextFront = new HashSet<int>();
            foreach (var u in frontier)
            {
                var du = dist.GetValueOrDefault(u, double.PositiveInfinity);
                if (du >= B)
                    continue;

                foreach (var (v, w) in graph.GetValueOrDefault(u, new List<(int, double)>()))
                {
                    var nd = du + w;
                    if (nd < B && !discovered.Contains(v))
                    {
                        discovered.Add(v);
                        nextFront.Add(v);
                    }
                }
            }
            frontier = nextFront;
        }

        var W = new HashSet<int>(discovered);

        // P must be small relative to S; ensure non-empty if S non-empty  
        if (!P.Any() && S.Any())
        {
            P = new HashSet<int> { S.First() };
        }

        return (P, W);
    }

    public static (Dictionary<int, List<(int, double)>>, List<(int, int, double)>) GenerateSparseDirectedGraph(int n, int m, int? seed = null)
    {
        var random = seed.HasValue ? new Random(seed.Value) : new Random();
        var graph = Enumerable.Range(0, n).ToDictionary(i => i, i => new List<(int, double)>());
        var edges = new List<(int, int, double)>();

        for (int i = 1; i < n; i++)
        {
            int u = random.Next(0, i);
            double w = random.NextDouble() * 100.0 + 1.0;
            graph[u].Add((i, w));
            edges.Add((u, i, w));
        }

        int remaining = Math.Max(0, m - (n - 1));
        for (int i = 0; i < remaining; i++)
        {
            int u = random.Next(n);
            int v = random.Next(n);
            double w = random.NextDouble() * 100.0 + 1.0;
            graph[u].Add((v, w));
            edges.Add((u, v, w));
        }

        return (graph, edges);
    }

    public static Dictionary<int, double> Dijkstra(Dictionary<int, List<(int, double)>> graph, int source, Instrument instr = null)
    {
        instr ??= new Instrument();
        var dist = graph.Keys.ToDictionary(v => v, v => double.PositiveInfinity);
        dist[source] = 0.0;
        var heap = new PriorityQueue<(double, int)>();
        heap.Enqueue((0.0, source));
        instr.HeapOps++;

        while (dist.Count < (source + 1))
        {
            Console.WriteLine("Dequeue Dijkstra heap.Count = " + heap.Count);
            var (dU, u) = heap.Dequeue();
            instr.HeapOps++;
            if (dU > dist[u]) continue;

            foreach (var (v, w) in graph[u])
            {
                instr.Relaxations++;
                double alt = dU + w;
                if (alt < dist.GetValueOrDefault(v, double.PositiveInfinity))
                {
                    dist[v] = alt;
                    heap.Enqueue((alt, v));
                    instr.HeapOps++;
                }
            }
        }

        return dist;
    }
}

public class Instrument
{
    public int Relaxations { get; set; }
    public int HeapOps { get; set; }

    public void Reset()
    {
        Relaxations = 0;
        HeapOps = 0;
    }
}

public class PriorityQueue<T> where T : IComparable<T>
{
    private readonly List<T> _elements = new List<T>();

    public int Count => _elements.Count;

    public void Enqueue(T item)
    {
        _elements.Add(item);
        _elements.Sort();
    }

    public T Dequeue()
    {
        if (_elements.Count == 0) throw new InvalidOperationException("Queue is empty.");
        var item = _elements[0];
        _elements.RemoveAt(0);
        return item;
    }
}

public class ArgumentParser
{
    private readonly Dictionary<string, (string, object)> _arguments = new Dictionary<string, (string, object)>();

    public void AddArgument(string shortName, string longName, object defaultValue, string description)
    {
        _arguments[shortName] = (longName, defaultValue);
    }

    public Dictionary<string, object> Parse(string[] args)
    {
        var parsedArgs = new Dictionary<string, object>();
        foreach (var arg in args)
        {
            if (_arguments.ContainsKey(arg))
            {
                var (longName, defaultValue) = _arguments[arg];
                parsedArgs[longName] = defaultValue;
            }
        }
        return parsedArgs;
    }

    public T GetValue<T>(string key)
    {
        return (T)_arguments[key].Item2;
    }
}

public class DataStructureD
{
    private List<(double Key, int Node)> heap;
    private Dictionary<int, double> best;
    private int M;
    private double BUpper;
    private int blockSize;

    public DataStructureD(int M, double BUpper, int? blockSize = null)
    {
        this.heap = new List<(double, int)>();
        this.best = new Dictionary<int, double>();
        this.M = Math.Max(1, M);
        this.BUpper = BUpper;
        this.blockSize = blockSize ?? Math.Max(1, this.M / 8);
    }

    public void Insert(int v, double key)
    {
        if (!best.TryGetValue(v, out var prev) || key < prev)
        {
            best[v] = key;
            heap.Add((key, v));
            heap.Sort((a, b) => a.Key.CompareTo(b.Key));
        }
    }

    public void BatchPrepend(IEnumerable<(int Node, double Key)> pairs)
    {
        foreach (var (v, key) in pairs)
        {
            Insert(v, key);
        }
    }

    private void Cleanup()
    {
        while (heap.Count > 0 && (!best.ContainsKey(heap[0].Node) || best[heap[0].Node] != heap[0].Key))
        {
            heap.RemoveAt(0);
        }
    }

    public bool IsEmpty()
    {
        Cleanup();
        return heap.Count == 0;
    }

    public (double Bi, HashSet<int> Si) Pull()
    {
        Cleanup();
        if (heap.Count == 0)
        {
            throw new InvalidOperationException("Pull from empty DataStructureD");
        }

        double Bi = heap[0].Key;
        var Si = new HashSet<int>();

        while (heap.Count > 0 && Si.Count < blockSize)
        {
            var (key, v) = heap[0];
            heap.RemoveAt(0);

            if (best.TryGetValue(v, out var bestKey) && bestKey == key)
            {
                Si.Add(v);
                best.Remove(v);
            }
        }

        return (Bi, Si);
    }
}
