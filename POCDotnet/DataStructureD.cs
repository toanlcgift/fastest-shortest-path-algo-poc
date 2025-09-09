using Node = System.Int32;

namespace POCDotnet
{

	// ---------------------------
	// DataStructure D (practical)
	// ---------------------------
	public class DataStructureD
	{
		private readonly PriorityQueue<Node, double> _heap = new();
		private readonly Dictionary<Node, double> _best = new();
		private readonly int _blockSize;
		private readonly int _M;
		private readonly double _BUpper;

		public DataStructureD(int M, double BUpper, int? blockSize = null)
		{
			_M = Math.Max(1, M);
			_BUpper = BUpper;
			_blockSize = blockSize ?? Math.Max(1, _M / 8);
		}

		public void Insert(Node v, double key)
		{
			if (!_best.TryGetValue(v, out var prev) || key < prev)
			{
				_best[v] = key;
				_heap.Enqueue(v, key);
			}
		}

		public void BatchPrepend(IEnumerable<(Node v, double key)> items)
		{
			foreach (var (v, key) in items) Insert(v, key);
		}

		private void Cleanup()
		{
			// remove stale heap entries
			while (_heap.Count > 0)
			{
				if (_heap.TryPeek(out var v, out var key))
				{
					if (_best.TryGetValue(v, out var val) && val == key) break;
					_heap.Dequeue(); // stale
				}
			}
		}

		public bool Empty()
		{
			Cleanup();
			return _heap.Count == 0;
		}

		public (double Bi, HashSet<Node> Si) Pull()
		{
			Cleanup();
			if (_heap.Count == 0) throw new InvalidOperationException("pull from empty D");
			_heap.TryPeek(out _, out var Bi);
			var Si = new HashSet<Node>();
			while (_heap.Count > 0 && Si.Count < _blockSize)
			{
				if (_heap.TryDequeue(out var v, out var key))
				{
					if (_best.TryGetValue(v, out var val) && val == key)
					{
						Si.Add(v);
						_best.Remove(v);
					}
				}
			}
			return (Bi, Si);
		}
	}
}
