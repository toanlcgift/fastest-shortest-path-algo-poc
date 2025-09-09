namespace POCDotnet
{
	public static class Program
	{
		public static void Main(string[] args)
		{
			int n = 200_000, m = 800_000, seed = 0;
			for (int i = 0; i < args.Length - 1; i++)
			{
				if (args[i] == "-n" || args[i] == "--nodes") int.TryParse(args[++i], out n);
				else if (args[i] == "-m" || args[i] == "--edges") int.TryParse(args[++i], out m);
				else if (args[i] == "-s" || args[i] == "--seed") int.TryParse(args[++i], out seed);
			}

			Harness.RunSingleTest(n, m, seed, source: 0);
		}
	}
}
