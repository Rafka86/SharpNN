using System.Text;

namespace ReSharpNN {

  internal class Connection {
    internal int PreLayerSize { get; }
    internal int PostLayerSize { get; }
    internal float[] Weight { get; }

    internal Connection(int preLayerSize, int postLayerSize, WeightsInitializeFunction initFunc) {
      PreLayerSize = preLayerSize;
      PostLayerSize = postLayerSize;

      Weight = initFunc(postLayerSize, preLayerSize);
    }

    internal float this[int i, int j] => Weight[PreLayerSize * i + j];
    
    internal string Dump(int dumpLevel) {
      var sb = new StringBuilder();
      sb.Clear();
      
      sb.AppendLine("================== Connection Infos =================");
      sb.AppendLine($"Number of pre layer's units  : {PreLayerSize.ToString()}");
      sb.AppendLine($"Number of post layer's units : {PostLayerSize.ToString()}");
      sb.AppendLine($"Number of weights            : {Weight.Length.ToString()}");

      if (dumpLevel > 1) {
        sb.AppendLine("---------------------- Weights ----------------------");
        for (var i = 0; i < PostLayerSize; i++)
          for (var j = 0; j < PreLayerSize; j++)
            sb.AppendLine($"  Weight[{i,4}, {j,4}] = {Weight[PreLayerSize * i + j]: ##0.###;-##0.###}");
        sb.AppendLine("-----------------------------------------------------");
      }
      
      sb.AppendLine("=====================================================");

      return sb.ToString();
    }
  }

}