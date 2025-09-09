using Graph = System.Collections.Generic.Dictionary<int, System.Collections.Generic.List<(int v, double w)>>;
using Node = System.Int32;

namespace POCDotnet
{


	// ---------------------------
	// BASECASE (Dijkstra-like limited)
	// ---------------------------
	public static class BaseCaseRunner
	{
		public static (double BPrime, HashSet<Node> Uo) Run(
		Graph graph,
			Dictionary<Node, double> dist,
		double B,
			HashSet<Node> S,
			int k,
			Instrument? instr = null)
		{
			instr ??= new Instrument();

			if (S.Count == 0) return (B, new HashSet<Node>());

			var x = S.MinBy(v => dist.TryGetValue(v, out var dv) ? dv : double.PositiveInfinity);
			var heap = new PriorityQueue<Node, double>();

			var startD = dist.TryGetValue(x, out var sd) ? sd : double.PositiveInfinity;
			heap.Enqueue(x, startD);
			instr.HeapOps++;

			var Uo = new HashSet<Node>();

			while (heap.Count > 0 && Uo.Count < (k + 1))
			{
				heap.TryDequeue(out var u, out var dU);
				instr.HeapOps++;
				if (dU > (dist.TryGetValue(u, out var du) ? du : double.PositiveInfinity)) continue;

				if (!Uo.Contains(u)) Uo.Add(u);

				foreach (var (v, w) in graph[u])
				{
					instr.Relaxations++;
					var newd = dist[u] + w;
					if (newd < (dist.TryGetValue(v, out var dv) ? dv : double.PositiveInfinity) && newd < B)
					{
						dist[v] = newd;
						heap.Enqueue(v, newd);
						instr.HeapOps++;
					}
				}
			}

			if (Uo.Count <= k)
			{
				return (B, Uo);
			}
			else
			{
				var finite = Uo.Where(v => !double.IsPositiveInfinity(dist[v])).Select(v => dist[v]).ToList();
				if (finite.Count == 0) return (B, new HashSet<Node>());
				var maxd = finite.Max();
				var Ufiltered = new HashSet<Node>(Uo.Where(v => dist[v] < maxd));
				return (maxd, Ufiltered);
			}
		}
	}
}
