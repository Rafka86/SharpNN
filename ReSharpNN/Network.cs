using System;
using System.Collections.Generic;
using System.Text;

using static System.Console;

using static ReSharpNN.UpdateFunctions;
using static ReSharpNN.WeightsInitializeFunctions;

namespace ReSharpNN {

  public class Network {
    private readonly List<Layer> _layers = new List<Layer>();
    private readonly List<Connection> _connections = new List<Connection>();
    private Layer _inputLayer;
    private Layer _outputLayer;

    private ErrorFunction _error;
    
    public int LayersCount => _layers.Count;
    public int ConnectionsCount => _connections.Count;

    public static class Factory {
      private static Network _network;

      public static void New() => _network = new Network();

      public static void SetErrorFunction(ErrorFunction errFunc) => _network._error = errFunc;
      
      public static void AddLayer(int size, UpdateFunction function = null, bool bias = true)
        => _network._layers.Add(new Layer(size, function ?? ReLU, bias));

      public static void AddConnection(WeightsInitializeFunction initFunction = null) {
        if (_network._layers.Count < 2) throw new ApplicationException("Not enough layers.");
        if (_network._layers.Count - 1 == _network._connections.Count)
          throw new ApplicationException("There are enough connections.");
        
        var lastIndex = _network._layers.Count - 1;
        var preLayer = _network._layers[lastIndex - 1];
        var postLayer = _network._layers[lastIndex];
        _network._connections.Add(new Connection(preLayer, postLayer, initFunction ?? He));
      }
      
      public static Network Create() {
        if (_network._error == null) throw new ArgumentException("Error function is not defined.");
        _network._inputLayer = _network._layers[0];
        _network._outputLayer = _network._layers[_network.LayersCount - 1];
        return _network;
      }

      public static void Create(out Network network) => network = Create();
    }

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
        
        connection.CalculationDeltaW(preLayer, postLayer);
        preLayer.CalculationDelta(postLayer, connection);
      }
    }

    internal void ClearDeltaW() {
      foreach (var connection in _connections)
        connection.ClearDeltaW();
    }

    internal void UpdateWeights(float learningRate) {
      foreach (var connection in _connections) {
        connection.ApplyDeltaW(learningRate);
      }
    }
    
    public float[] Output => _outputLayer.Unit;

    public float Error(float[] teacher) => _error(_outputLayer.Unit, teacher);

    public bool CheckStatus() {
      var checkRelNums = LayersCount - 1 == ConnectionsCount;
      WriteLine($"Check the numbers of Layers and Connections. : {checkRelNums}");
      WriteLine($"# of Layers : {LayersCount}");
      WriteLine($"# of Connections : {ConnectionsCount}");

      return checkRelNums;
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