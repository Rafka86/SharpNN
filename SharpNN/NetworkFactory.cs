using System;

using static SharpNN.ActivateFunctions;
using static SharpNN.ErrorFunctions;
using static SharpNN.WeightsInitializeFunctions;

namespace SharpNN {

  public partial class Network {
    public static class Factory {
      private static Network _network;

      public static void New() => _network = new Network();

      public static void SetErrorFunction(ErrorFunction errFunc) => _network._error = errFunc;
      
      public static void AddLayer(int size, ActivateFunction function = null, bool bias = true)
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
        var res = _network;
        _network = null;
        return res;
      }

      public static void Create(out Network network) => network = Create();

      public static void DockIn(Network network) => _network = network;

      public static Network PartialNetwork(int hiddenLayerNum) {
        if (_network == null) throw new ApplicationException("No network in the Factory.");
        var pNet = new Network {_inputLayer = _network._inputLayer, _error = MeanSquared};
        pNet._layers.Add(_network._layers[0]);

        for (var i = 1; i <= hiddenLayerNum; i++) {
          pNet._connections.Add(_network._connections[i - 1]);
          pNet._layers.Add(_network._layers[i]);
        }

        var outputLayerBase = pNet._layers[pNet.LayersCount - 2];
        pNet._outputLayer = new Layer(outputLayerBase.PureSize, outputLayerBase.Function, false);
        pNet._connections.Add(new Connection(pNet._layers[pNet.LayersCount - 1], pNet._outputLayer, Uniform));
        pNet._layers.Add(pNet._outputLayer);

        return pNet;
      }

      public static void OutOfDock() {
        _network = null;
      }
    }
  }

}