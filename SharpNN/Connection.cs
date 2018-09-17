using System.Text;
using System.Threading.Tasks;

using SharpMKL;

using SharpNN.Optimizer;

namespace SharpNN {

  internal class Connection {
    private readonly Layer _preLayer;
    private readonly Layer _postLayer;
    internal int PreLayerSize => _preLayer.Size;
    internal int PostLayerSize => _postLayer.PureSize;

    internal float[] Weight { get; private set; }
    private readonly float[] _dw;
    internal Optimizer.Optimizer Optimizer { get; set; } = new SGD();

    internal Connection(Layer preLayer, Layer postLayer, WeightsInitializeFunction initFunc) {
      _preLayer = preLayer;
      _postLayer = postLayer;

      Weight = initFunc(PostLayerSize, PreLayerSize);
      _dw = new float[PreLayerSize * PostLayerSize];
    }
    
    internal float this[int i, int j] {
      set => Weight[PreLayerSize * i + j] = value;
      get => Weight[PreLayerSize * i + j];
    }

    internal void ClearDeltaW() {
      Parallel.For(0, _dw.Length, i => { _dw[i] = 0.0f; });
    }
    
    internal void CalculationDeltaW() {
      Parallel.For(0, PostLayerSize, i => {
                                       for (var j = 0; j < PreLayerSize; j++)
                                         _dw[PreLayerSize * i + j] += _postLayer.Delta[i] * _preLayer.Unit[j];
                                     });
    }

    internal void CalculationBiasDeltaW() {
      Parallel.For(0, PostLayerSize,
                   i => {
                     _dw[PreLayerSize * i + PreLayerSize - 1]
                       += _postLayer.Delta[i] * _preLayer.Unit[_preLayer.Unit.Length - 1];
                   });
    }

    internal void ApplyDeltaW() {
      Optimizer.Update(Weight, _dw);
    }

    internal void PostProcess() {
      if (!Optimizer.NeedPostProcess) return;
      Weight = Optimizer.PostProcess();
    }
    
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
          for (var j = 0; j < PreLayerSize; j++) {
            sb.Append($"  Weight[{i,4}, {j,4}] = {Weight[PreLayerSize * i + j]: ##0.0##;-##0.0##}");
            sb.AppendLine($"\tdW[{i,4}, {j,4}] = {_dw[PreLayerSize * i + j]: ##0.0##;-##0.0##}");
          }
        sb.AppendLine("-----------------------------------------------------");
      }
      
      sb.AppendLine("=====================================================");

      return sb.ToString();
    }
  }

}