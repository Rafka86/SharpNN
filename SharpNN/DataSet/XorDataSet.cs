namespace SharpNN.DataSet {

  public class XorDataSet : DataSet {
    public XorDataSet() {
      TrainingDataList.Add(new Datum(new[] {0.0f, 0.0f}, new[] {0.0f}));
      TrainingDataList.Add(new Datum(new[] {0.0f, 1.0f}, new[] {1.0f}));
      TrainingDataList.Add(new Datum(new[] {1.0f, 0.0f}, new[] {1.0f}));
      TrainingDataList.Add(new Datum(new[] {1.0f, 1.0f}, new[] {0.0f}));
      TestDataList.Add(new Datum(new[] {0.0f, 0.0f}, new[] {0.0f}));
      TestDataList.Add(new Datum(new[] {0.0f, 1.0f}, new[] {1.0f}));
      TestDataList.Add(new Datum(new[] {1.0f, 0.0f}, new[] {1.0f}));
      TestDataList.Add(new Datum(new[] {1.0f, 1.0f}, new[] {0.0f}));

      InputDataSize = 2;
      OutputDataSize = 1;
    }
  }

}