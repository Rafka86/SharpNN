using System.Text;

namespace ReSharpNN {

  internal class Layer {
    internal float[] Input { get; }
    internal float[] Unit { get; }
    internal float[] Delta { get; }

    internal int Size => Unit.Length;
    internal int PureSize => Input.Length;
    internal bool HasBias { get; }

    private readonly UpdateFunction _function;
    
    internal Layer(int size, UpdateFunction updFunc, bool hasBias) {
      Input = new float[size];
      Unit = new float[size + (hasBias ? 1 : 0)];
      Delta = new float[size];

      _function = updFunc;
      
      HasBias = hasBias;
    }

    internal void Update(Layer preLayer, Connection connection) {
      for (var i = 0; i < Input.Length; i++) Input[i] = 0.0f;
      for (var i = 0; i < connection.PostLayerSize; i++)
        for (var j = 0; j < connection.PreLayerSize; j++)
          Input[i] += connection[i, j] * preLayer.Unit[j];

      var tmp = _function(Input);
      for (var i = 0; i < tmp.Length; i++)
        Unit[i] = tmp[i];
    }

    internal string Dump(int dumpLevel) {
      var sb = new StringBuilder();
      sb.Clear();

      sb.AppendLine("==================== Layer Infos ====================");
      sb.AppendLine($"Number of neurons              : {Unit.Length.ToString()}");
      //sb.AppendLine($"This layer is the output layer : {IsOutputLayer}");
      sb.AppendLine($"This layer's function          : {_function.Method.Name}");
      //sb.AppendLine($"This layer's delta             : {df.Method.Name}");

      if (dumpLevel > 1) {
        sb.AppendLine("-------------------- Activities ---------------------");
        for (var i = 0; i < Input.Length; i++)
          sb.AppendLine($"  Input[{i,4}] = {Input[i],-8: ##0.###;-##0.###}"
                      + $"\tUnit[{i,4}] = {Unit[i],-8: ##0.###;-##0.###}"
                      + $"\tDelta[{i,4}] = {Delta[i],-8: ##0.###;-##0.###}");
        sb.AppendLine("-----------------------------------------------------");
      }
      
      sb.AppendLine("=====================================================");
      
      return sb.ToString();
    }
  }

}