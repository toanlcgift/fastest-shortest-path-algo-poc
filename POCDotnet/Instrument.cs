namespace POCDotnet
{
	// ---------------------------
	// Instrumentation
	// ---------------------------
	public class Instrument
	{
		public long Relaxations { get; set; }
		public long HeapOps { get; set; }
		public void Reset() { Relaxations = 0; HeapOps = 0; }
	}
}
