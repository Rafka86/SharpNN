using System.Text;

namespace SharpNN {

  internal class Connection {
    private readonly Layer _preLayer;
    private readonly Layer _postLayer;
    internal int PreLayerSize => _preLayer.Size;
    internal int PostLayerSize => _postLayer.PureSize;

    private readonly float[] _weight;
    private readonly float[] _dw;

    internal Connection(Layer preLayer, Layer postLayer, WeightsInitializeFunction initFunc) {
      _preLayer = preLayer;
      _postLayer = postLayer;

      _weight = initFunc(PostLayerSize, PreLayerSize);
      _dw = new float[PreLayerSize * PostLayerSize];
    }

    internal float this[int i, int j] => _weight[PreLayerSize * i + j];

    internal void ClearDeltaW() {
      for (var i = 0; i < _dw.Length; i++)
        _dw[i] = 0.0f;
    }
    
    internal void CalculationDeltaW(Layer preLayer, Layer postLayer) {
      for (var i = 0; i < PostLayerSize; i++)
        for (var j = 0; j < PreLayerSize; j++)
          _dw[PreLayerSize * i + j] += postLayer.Delta[i] * preLayer.Unit[j];
    }

    internal void ApplyDeltaW(float learningRate) {
      for (var i = 0; i < _weight.Length; i++) {
        _weight[i] -= learningRate * _dw[i];
        _dw[i] = 0.0f;
      }
    }
    
    internal string Dump(int dumpLevel) {
      var sb = new StringBuilder();
      sb.Clear();
      
      sb.AppendLine("================== Connection Infos =================");
      sb.AppendLine($"Number of pre layer's units  : {PreLayerSize.ToString()}");
      sb.AppendLine($"Number of post layer's units : {PostLayerSize.ToString()}");
      sb.AppendLine($"Number of weights            : {_weight.Length.ToString()}");

      if (dumpLevel > 1) {
        sb.AppendLine("---------------------- Weights ----------------------");
        for (var i = 0; i < PostLayerSize; i++)
          for (var j = 0; j < PreLayerSize; j++)
            sb.AppendLine($"  Weight[{i,4}, {j,4}] = {_weight[PreLayerSize * i + j]: ##0.###;-##0.###}");
        sb.AppendLine("-----------------------------------------------------");
      }
      
      sb.AppendLine("=====================================================");

      return sb.ToString();
    }
  }

}