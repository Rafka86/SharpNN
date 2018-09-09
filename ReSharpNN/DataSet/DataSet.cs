using System.Collections.Generic;

namespace ReSharpNN.DataSet {

  public abstract class DataSet {
    protected List<Datum> TrainingDataList { get; } = new List<Datum>();
    protected List<Datum> TestDataList { get; } = new List<Datum>();
    public int TrainingDataSize => TrainingDataList.Count;
    public int TestDataSize => TestDataList.Count;

    public virtual IEnumerator<Datum> TrainingData() {
      foreach (var datum in TrainingDataList) yield return datum;
    }
    public virtual IEnumerator<Datum> TestData() {
      foreach (var datum in TestDataList) yield return datum;
    }
  }

}