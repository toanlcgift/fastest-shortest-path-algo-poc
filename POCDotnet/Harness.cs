using System.Diagnostics;

namespace POCDotnet
{
	/// <summary>
	/// Test harness (similar to run_single_test)
	/// </summary>
	public static class Harness
	{
		public static void RunSingleTest(int n, int m, int seed = 0, int source = 0)
		{
			Console.WriteLine($"Generating graph: n={n}, m={m}, seed={seed}");
			var (graph, edges) = GraphGenerator.GenerateSparseDirectedGraph(n, m, seed: seed);
			double avgDeg = graph.Values.Sum(adj => adj.Count) / (double)n;
			Console.WriteLine($"Graph generated. avg out-degree ≈ {avgDeg:F3}");

			var distDij = graph.Keys.ToDictionary(v => v, _ => double.PositiveInfinity);
			distDij[source] = 0.0;

			var instrDij = new Instrument();
			var sw = Stopwatch.StartNew();
			distDij = ShortestPath.Dijkstra(graph, source, instrDij);
			sw.Stop();
			Console.WriteLine($"Dijkstra: time={sw.Elapsed.TotalSeconds:F6}s, relaxations={instrDij.Relaxations}, heap_ops={instrDij.HeapOps}, reachable={distDij.Values.Count(v => !double.IsInfinity(v))}");

			var distBm = graph.Keys.ToDictionary(v => v, _ => double.PositiveInfinity);
			distBm[source] = 0.0;

			var instrBm = new Instrument();
			int l;
			if (n <= 2) l = 1;
			else
			{
				double ln = Math.Log(Math.Max(3, n));
				int tGuess = Math.Max(1, (int)Math.Round(Math.Pow(ln, 2.0 / 3.0)));
				l = Math.Max(1, (int)Math.Max(1, Math.Round(ln / tGuess)));
			}
			Console.WriteLine($"BMSSP params: top-level l={l}");

			sw.Restart();
			var (Bp, Ufinal) = BmsspRunner.Run(graph, distBm, edges, l, double.PositiveInfinity, new HashSet<int> { source }, n, instrBm);
			sw.Stop();
			Console.WriteLine($"BMSSP: time={sw.Elapsed.TotalSeconds:F6}s, relaxations={instrBm.Relaxations}, reachable={distBm.Values.Count(v => !double.IsInfinity(v))}, B'={Bp}, |U_final|={Ufinal.Count}");

			var diffs = new List<double>();
			foreach (var v in graph.Keys)
			{
				var dv = distDij[v];
				var db = distBm[v];
				if (!double.IsInfinity(dv) && !double.IsInfinity(db))
				{
					diffs.Add(Math.Abs(dv - db));
				}
			}
			double maxDiff = diffs.Count > 0 ? diffs.Max() : 0.0;
			Console.WriteLine($"Distance agreement (max abs diff on commonly reachable nodes): {maxDiff:E6}");
		}
	}
}
