using Graph = System.Collections.Generic.Dictionary<int, System.Collections.Generic.List<(int v, double w)>>;
using Node = System.Int32;
using Weight = System.Double;

namespace POCDotnet
{
	// ---------------------------
	// Dijkstra
	// ---------------------------
	public static class ShortestPath
	{
		public static Dictionary<Node, Weight> Dijkstra(Graph graph, Node source, Instrument? instr = null)
		{
			instr ??= new Instrument();
			var dist = graph.Keys.ToDictionary(v => v, _ => double.PositiveInfinity);
			dist[source] = 0.0;

			var pq = new PriorityQueue<Node, double>();
			pq.Enqueue(source, 0.0);
			instr.HeapOps++;

			while (pq.Count > 0)
			{
				pq.TryDequeue(out var u, out var dU);
				instr.HeapOps++;
				if (dU > dist[u]) continue;

				foreach (var (v, w) in graph[u])
				{
					instr.Relaxations++;
					double alt = dU + w;
					if (alt < dist[v])
					{
						dist[v] = alt;
						pq.Enqueue(v, alt);
						instr.HeapOps++;
					}
				}
			}
			return dist;
		}
	}
}
