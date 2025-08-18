namespace POCDotnet
{
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
}
