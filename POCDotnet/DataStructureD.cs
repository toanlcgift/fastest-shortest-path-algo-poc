namespace POCDotnet
{

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
}
