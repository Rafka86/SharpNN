namespace SharpNN.DataSet {

  public struct Datum {
    public float[] Input { get; }
    public float[] Output { get; }

    public Datum(float[] input, float[] output) {
      Input = input;
      Output = output;
    }
  }

}