using Edge = System.ValueTuple<int, int, double>;
using Graph = System.Collections.Generic.Dictionary<int, System.Collections.Generic.List<(int v, double w)>>;

namespace POCDotnet
{

	// ---------------------------
	// Utilities & Graph generator
	// ---------------------------
	public static class GraphGenerator
	{
		public static (Graph, List<Edge>) GenerateSparseDirectedGraph(int n, int m, double maxW = 100.0, int? seed = null)
		{
			var rand = seed.HasValue ? new Random(seed.Value) : new Random();
			var graph = new Graph();
			for (int i = 0; i < n; i++) graph[i] = new List<(int, double)>();
			var edges = new List<Edge>();

			// weak backbone to avoid isolated nodes
			for (int i = 1; i < n; i++)
			{
				int u = rand.Next(i);
				double w = rand.NextDouble() * maxW + 1.0;
				graph[u].Add((i, w));
				edges.Add((u, i, w));
			}

			int remaining = Math.Max(0, m - (n - 1));
			for (int i = 0; i < remaining; i++)
			{
				int u = rand.Next(n);
				int v = rand.Next(n);
				double w = rand.NextDouble() * maxW + 1.0;
				graph[u].Add((v, w));
				edges.Add((u, v, w));
			}
			return (graph, edges);
		}
	}
}
