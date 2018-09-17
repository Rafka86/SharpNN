using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace SharpNN {

  public partial class Network {
    private readonly List<Layer> _layers = new List<Layer>();
    private readonly List<Connection> _connections = new List<Connection>();
    private Layer _inputLayer;
    private Layer _outputLayer;

    private ErrorFunction _error;
    
    public int LayersCount => _layers.Count;
    public int ConnectionsCount => _connections.Count;

    public void SetInputs(float[] input) {
      for (var i = 0; i < input.Length; i++)
        _inputLayer.Unit[i] = input[i];
    }
    
    public void ForwardPropagation() {
      for (var i = 1; i < _layers.Count; i++) {
        var preLayer = _layers[i - 1];
        var connection = _connections[i - 1];
        var layer = _layers[i];
        layer.Update(preLayer, connection);
      }
    }

    public void BackPropagation(float[] teacher) {
      _outputLayer.CalculationOutputLayerDelta(teacher);

      for (var i = LayersCount - 2; i >= 0; i--) {
        var preLayer = _layers[i];
        var connection = _connections[i];
        var postLayer = _layers[i + 1];
        
        connection.CalculationDeltaW();
        preLayer.CalculationDelta(postLayer, connection);
      }
    }

    internal void CopyWeightValues() {
      var from = _connections[ConnectionsCount - 2];
      var target = _connections[ConnectionsCount - 1];

      for (var i = 0; i < from.PostLayerSize; i++)
        for (var j = 0; j < from.PreLayerSize - 1; j++)
          target[j, i] = from[i, j];
    }
    
    internal void PartialBackPropagation() {
      var teacherLayer = _layers[LayersCount - 3];
      _outputLayer.CalculationOutputLayerDelta(teacherLayer.Unit.AsSpan(0, teacherLayer.PureSize).ToArray());

      var lastIndex = LayersCount - 1;
      _connections[lastIndex - 1].CalculationBiasDeltaW();
      _layers[lastIndex - 1].CalculationDelta(_layers[lastIndex], _connections[lastIndex - 1]);
      _connections[lastIndex - 2].CalculationDeltaW();
    }

    internal void SetOptimizer(Optimizer.Optimizer optimizer) {
      foreach (var connection in _connections)
        connection.Optimizer = optimizer.Clone();
    }

    internal void FinishLearning() {
      foreach (var connection in _connections)
        connection.PostProcess();
    }
    
    public void ClearDeltaW() {
      foreach (var connection in _connections)
        connection.ClearDeltaW();
    }

    public void UpdateWeights() {
      foreach (var connection in _connections) {
        connection.ApplyDeltaW();
      }
    }
    
    internal void UpdatePartialWeights() {
      _connections[ConnectionsCount - 1].ApplyDeltaW();
      _connections[ConnectionsCount - 2].ApplyDeltaW();
    }
    
    public float[] Output => _outputLayer.Unit;

    public float[] Input => _inputLayer.Unit.AsSpan(0, _inputLayer.PureSize).ToArray();

    public float Error(float[] teacher) => _error(_outputLayer.Unit, teacher);

    internal float Error() => _error(_outputLayer.Unit,
                                   _layers[LayersCount - 3].Unit.AsSpan(0, _layers[LayersCount - 3].PureSize).ToArray());

    public bool CheckStatus() {
      var checkRelNums = LayersCount - 1 == ConnectionsCount;
      Console.WriteLine($"Check the numbers of Layers and Connections. : {checkRelNums}");
      Console.WriteLine($"# of Layers : {LayersCount}");
      Console.WriteLine($"# of Connections : {ConnectionsCount}");

      var checkRelErrAct = true;
      if (_error.Method.Name == "CrossEntropy") checkRelErrAct = _outputLayer.Function.Method.Name == "Softmax";
      Console.WriteLine($"Check the relation between error function and output layer activate function. : {checkRelErrAct}");
      Console.WriteLine($"Error function : {_error.Method.Name}");
      Console.WriteLine($"Activate function of the output layer : {_outputLayer.Function.Method.Name}");

      return checkRelNums & checkRelErrAct;
    }

    public void Print(string filePath = null, int dumpLevel = 1) {
      if (filePath == null) Console.WriteLine(Dump(dumpLevel));
      else using (var sw = new StreamWriter(filePath)) sw.WriteLine(Dump(dumpLevel));
    }
    
    public string Dump(int dumpLevel = 1) {
      var sb = new StringBuilder();
      sb.Clear();
      
      sb.AppendLine("=================== Network Infos ===================");
      sb.AppendLine($"Number of layers : {_layers.Count.ToString()}");
      sb.AppendLine($"       Layers[{0,3}] : {_layers[0].Unit.Length.ToString()}(units) {_layers[0].HasBias.ToString()}(has bias)");
      for (var i = 1; i < _layers.Count; i++) {
        sb.AppendLine($"  Connections[{i - 1,3}] : {_connections[i - 1].PreLayerSize.ToString()}(pre neurons) {_connections[i - 1].PostLayerSize.ToString()}(post neurons)");
        sb.AppendLine($"       Layers[{i,3}] : {_layers[i].Unit.Length.ToString()}(neurons) {_layers[i].HasBias.ToString()}(has bias)");
      }
      sb.AppendLine("=====================================================");
      if (dumpLevel <= 0) return sb.ToString();

      sb.AppendLine();
      sb.AppendLine(">> Layer 0");
      sb.AppendLine(_layers[0].Dump(dumpLevel));
      for (var i = 1; i < _layers.Count; i++) {
        sb.AppendLine($">> Connection Layer {i - 1} -> Layer {i}");
        sb.AppendLine(_connections[i - 1].Dump(dumpLevel));
        sb.AppendLine($">> Layer {i}");
        sb.AppendLine(_layers[i].Dump(dumpLevel));
      }

      return sb.ToString();
    }

    public override string ToString() => Dump();
  }

}