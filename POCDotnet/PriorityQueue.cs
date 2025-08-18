namespace POCDotnet
{
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
}
