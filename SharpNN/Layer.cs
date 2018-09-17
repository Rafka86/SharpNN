using System.Text;
using System.Threading.Tasks;

using SharpMKL;

using static SharpNN.ActivateDiffFunctions;

namespace SharpNN {

  internal class Layer {
    internal float[] Input { get; }
    internal float[] Unit { get; }
    internal float[] Delta { get; }

    internal int Size => Unit.Length;
    internal int PureSize => Input.Length;
    internal bool HasBias { get; }

    internal ActivateFunction Function { get; }
    private readonly DiffFunction _diff;
    
    internal Layer(int size, ActivateFunction updFunc, bool hasBias) {
      Input = new float[size];
      Unit = new float[size + (hasBias ? 1 : 0)];
      Delta = new float[size];

      Function = updFunc;
      _diff = ChooseDiffFunction();
      
      HasBias = hasBias;
      if (HasBias) Unit[Input.Length] = 1.0f;

      DiffFunction ChooseDiffFunction() {
        switch (updFunc.Method.Name) {
          case "Identity": return DiffIdentity;
          case "Tanh": return DiffTanh;
          case "Sigmoid": return DiffSigmoid;
          case "ReLU": return DiffReLU;
          case "LeakyReLU": return DiffLeakyReLU;
          default: return null;
        }
      }
    }
    
    internal void Update(Layer preLayer, Connection connection) {
      Blas2.gemv(BlasLayout.RowMajor, BlasTranspose.NoTrans,
                 connection.PostLayerSize, connection.PreLayerSize, 1.0f, connection.Weight, connection.PreLayerSize,
                 preLayer.Unit, 1, 0.0f, Input, 1);

      Blas1.copy(PureSize, Function(Input), 1, Unit, 1);
    }

    internal void CalculationOutputLayerDelta(float[] teacher) {
      Parallel.For(0, Delta.Length, i => { Delta[i] = Unit[i] - teacher[i]; });
    }

    internal void CalculationDelta(Layer postLayer, Connection outConnection) {
      var diff = _diff(Unit);
      Parallel.For(0, Delta.Length, i => {
                                      Delta[i] = 0.0f;
                                      for (var k = 0; k < postLayer.Delta.Length; k++)
                                        Delta[i] += postLayer.Delta[k] * outConnection[k, i];
                                      Delta[i] *= diff[i];
                                    });
    }

    internal string Dump(int dumpLevel) {
      var sb = new StringBuilder();
      sb.Clear();

      sb.AppendLine("==================== Layer Infos ====================");
      sb.AppendLine($"Number of neurons              : {Unit.Length.ToString()}");
      //sb.AppendLine($"This layer is the output layer : {IsOutputLayer}");
      sb.AppendLine($"This layer's function          : {Function.Method.Name}");
      sb.AppendLine($"This layer's delta             : {_diff?.Method.Name ?? "None"}");

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