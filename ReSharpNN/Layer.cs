namespace ReSharpNN {

  internal class Layer {
    internal float[] Input;
    internal float[] Unit;
    internal float[] Delta;

    internal Layer(int size, bool hasBias = false) {
      Input = new float[size];
      Unit = new float[size];
      Delta = new float[size];
    }
  }

}