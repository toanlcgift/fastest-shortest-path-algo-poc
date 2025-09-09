using Graph = System.Collections.Generic.Dictionary<int, System.Collections.Generic.List<(int v, double w)>>;
using Node = System.Int32;

namespace POCDotnet
{
	/// <summary>
	/// FIND_PIVOTS (practical bounded BF)
	/// </summary>
	public static class PivotFinder
	{
		public static (HashSet<Node> P, HashSet<Node> W) FindPivots(
		Graph graph,
			Dictionary<Node, double> dist,
			HashSet<Node> S,
			double B,
			int n,
			int kSteps,
			int pLimit,
			Instrument? instr = null)
		{
			instr ??= new Instrument();

			var sFiltered = S.Where(v => dist.TryGetValue(v, out var dv) && dv < B).ToList();

			HashSet<Node> P;
			if (sFiltered.Count == 0)
			{
				var take = Math.Max(1, Math.Min(S.Count, pLimit));
				P = new HashSet<Node>(S.Take(take));
			}
			else
			{
				sFiltered.Sort((a, b) => dist[a].CompareTo(dist[b]));
				P = new HashSet<Node>(sFiltered.Take(Math.Max(1, Math.Min(sFiltered.Count, pLimit))));
			}

			var sourceFrontier = P.Count > 0 ? new HashSet<Node>(P) : new HashSet<Node>(S);
			var discovered = new HashSet<Node>(sourceFrontier);
			var frontier = new HashSet<Node>(sourceFrontier);

			for (int step = 0; step < Math.Max(1, kSteps); step++)
			{
				if (frontier.Count == 0) break;
				var nextFront = new HashSet<Node>();
				foreach (var u in frontier)
				{
					var du = dist.TryGetValue(u, out var vdu) ? vdu : double.PositiveInfinity;
					if (du >= B) continue;
					foreach (var (v, w) in graph[u])
					{
						instr.Relaxations++;
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

			var W = new HashSet<Node>(discovered);
			if (P.Count == 0 && S.Count > 0)
			{
				P.Add(S.First());
			}
			return (P, W);
		}
	}
}
