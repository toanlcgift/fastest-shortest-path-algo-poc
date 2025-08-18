namespace POCDotnet
{
    public class ArgumentParser
    {
        private readonly Dictionary<string, (string, object)> _arguments = new Dictionary<string, (string, object)>();

        public void AddArgument(string shortName, string longName, object defaultValue, string description)
        {
            _arguments[shortName] = (longName, defaultValue);
        }

        public Dictionary<string, object> Parse(string[] args)
        {
            var parsedArgs = new Dictionary<string, object>();
            foreach (var arg in args)
            {
                if (_arguments.ContainsKey(arg))
                {
                    var (longName, defaultValue) = _arguments[arg];
                    parsedArgs[longName] = defaultValue;
                }
            }
            return parsedArgs;
        }

        public T GetValue<T>(string key)
        {
            return (T)_arguments[key].Item2;
        }
    }
}
