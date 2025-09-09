using Edge = System.ValueTuple<int, int, double>;
using Graph = System.Collections.Generic.Dictionary<int, System.Collections.Generic.List<(int v, double w)>>;
using Node = System.Int32;

namespace POCDotnet
{


	// ---------------------------
	// BMSSP (practical recursive)
	// ---------------------------
	public static class BmsspRunner
	{
		private static (int tParam, int kParam) GetParams(int n)
		{
			if (n <= 2) return (1, 2);
			double ln = Math.Log(Math.Max(3, n));
			int t = Math.Max(1, (int)Math.Round(Math.Pow(ln, 2.0 / 3.0)));
			int k = Math.Max(2, (int)Math.Round(Math.Pow(ln, 1.0 / 3.0)));
			return (t, k);
		}

		public static (double BPrime, HashSet<Node> UFinal) Run(
		Graph graph,
			Dictionary<Node, double> dist,
			List<Edge> edges,
		int l,
		double B,
			HashSet<Node> S,
			int n,
			Instrument? instr = null)
		{
			instr ??= new Instrument();

			var (tParam, kParam) = GetParams(n);

			if (l <= 0)
			{
				if (S.Count == 0) return (B, new HashSet<Node>());
				return BaseCaseRunner.Run(graph, dist, B, S, kParam, instr);
			}

			int pLimit = Math.Max(1, 1 << Math.Min(10, tParam));
			int kSteps = Math.Max(1, kParam);

			var (P, W) = PivotFinder.FindPivots(graph, dist, S, B, n, kSteps, pLimit, instr);

			int M = 1 << Math.Max(0, (l - 1) * tParam);
			var D = new DataStructureD(M, B, blockSize: Math.Max(1, Math.Min(Math.Max(P.Count, 1), 64)));

			foreach (var x in P)
			{
				dist.TryGetValue(x, out var dx);
				D.Insert(x, dx);
			}

			double bPrimeInitial = P.Count > 0 ? P.Min(x => dist.TryGetValue(x, out var dx) ? dx : double.PositiveInfinity) : B;
			var U = new HashSet<Node>();
			var bPrimeSubs = new List<double>();

			int loopGuard = 0;
			int limit = kParam * (1 << (l * Math.Max(1, tParam)));

			while (U.Count < limit && !D.Empty())
			{
				loopGuard++;
				if (loopGuard > 20000) break;

				double Bi;
				HashSet<Node> Si;
				try
				{
					(Bi, Si) = D.Pull();
				}
				catch
				{
					break;
				}

				var (bPrimeSub, Ui) = Run(graph, dist, edges, l - 1, Bi, Si, n, instr);
				bPrimeSubs.Add(bPrimeSub);
				U.UnionWith(Ui);

				var batch = new HashSet<(Node v, double key)>();

				foreach (var u in Ui)
				{
					if (!dist.TryGetValue(u, out var du) || double.IsPositiveInfinity(du)) continue;

					foreach (var (v, wuv) in graph[u])
					{
						instr.Relaxations++;
						var newd = du + wuv;

						if (newd <= (dist.TryGetValue(v, out var dv) ? dv : double.PositiveInfinity))
						{
							dist[v] = newd;
							if (Bi <= newd && newd < B)
							{
								D.Insert(v, newd);
							}
							else if (bPrimeSub <= newd && newd < Bi)
							{
								batch.Add((v, newd));
							}
						}
					}
				}

				foreach (var x in Si)
				{
					var dx = dist.TryGetValue(x, out var dxx) ? dxx : double.PositiveInfinity;
					if (bPrimeSub <= dx && dx < Bi)
					{
						batch.Add((x, dx));
					}
				}

				if (batch.Count > 0) D.BatchPrepend(batch);
			}

			double bPrimeFinal = bPrimeSubs.Count > 0 ? Math.Min(bPrimeSubs.Prepend(bPrimeInitial).Min(), bPrimeInitial) : bPrimeInitial;

			var UFinal = new HashSet<Node>(U);
			foreach (var x in W)
			{
				if (dist.TryGetValue(x, out var dx) && dx < bPrimeFinal) UFinal.Add(x);
			}

			return (bPrimeFinal, UFinal);
		}
	}
}
