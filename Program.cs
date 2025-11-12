using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace DeteksiHoaxKonoha
{
    public class ModelInput
    {
        [LoadColumn(0)] public string? Title { get; set; }
        [LoadColumn(1)] public string? Text { get; set; }
        [LoadColumn(2)] public string? Subject { get; set; }
        [LoadColumn(3)] public string? Date { get; set; }
        [LoadColumn(4)] public bool Label { get; set; } 
    }

    public class ModelOutput
    {
        [ColumnName("PredictedLabel")] public bool PredictedLabel { get; set; }
        public float Probability { get; set; }
        public float Score { get; set; }
    }

    class Program
    {
        private static string TRUE_DATASET_PATH = "true.csv";
        private static string FAKE_DATASET_PATH = "fake.csv";
        private static string COMBINED_DATASET_PATH = "combined.csv";
        private static string MODEL_PATH = "model_hoax.zip";

        static void Main(string[] args)
        {
            var mlContext = new MLContext(seed: 42);

            Console.WriteLine("## 1. Memuat dan Memproses Data (Biner)...");

            // Gabungkan dataset jika belum ada
            if (!File.Exists(COMBINED_DATASET_PATH))
                CombineDatasets();

            var loader = mlContext.Data.CreateTextLoader(
                columns: new[]
                {
                    new TextLoader.Column("Title", DataKind.String, 0),
                    new TextLoader.Column("Text", DataKind.String, 1),
                    new TextLoader.Column("Subject", DataKind.String, 2),
                    new TextLoader.Column("Date", DataKind.String, 3),
                    new TextLoader.Column("Label", DataKind.Boolean, 4)
                },
                hasHeader: true,
                separatorChar: ','
            );

            var data = loader.Load(COMBINED_DATASET_PATH);

            
            var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);
            var trainData = split.TrainSet;
            var testData = split.TestSet;

            
            var testEnum = mlContext.Data.CreateEnumerable<ModelInput>(testData, reuseRowObject: false).ToList();
            int pos = testEnum.Count(x => x.Label);
            int neg = testEnum.Count(x => !x.Label);
            Console.WriteLine($"--- Diagnostik Data ---");
            Console.WriteLine($"Total Data: {testEnum.Count}");
            Console.WriteLine($"Positif (hoax): {pos}");
            Console.WriteLine($"Negatif (benar): {neg}");
            Console.WriteLine("-----------------------");

            Console.WriteLine("\n## 2. Mendefinisikan Pipeline dan Melatih Model (Biner)...");

            var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", "Text")
                .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
                    labelColumnName: "Label", featureColumnName: "Features"))
                .Append(mlContext.Transforms.CopyColumns("PredictedLabel", "PredictedLabel"));

            var model = pipeline.Fit(trainData);

            Console.WriteLine("Pelatihan model selesai.");

            Console.WriteLine("\n## 3. Mengevaluasi Model...");
            try
            {
                var predictions = model.Transform(testData);
                var metrics = mlContext.BinaryClassification.Evaluate(predictions, labelColumnName: "Label");

                Console.WriteLine($"Akurasi: {metrics.Accuracy:P2}");
                Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:P2}");
                Console.WriteLine($"F1 Score: {metrics.F1Score:P2}");
            }
            catch (ArgumentOutOfRangeException)
            {
                Console.WriteLine("⚠️ Tidak dapat menghitung AUC: hanya satu kelas di data uji.");
            }

            
            mlContext.Model.Save(model, trainData.Schema, MODEL_PATH);
            Console.WriteLine($"\nModel disimpan ke: {MODEL_PATH}");

            Console.WriteLine("\n## 4. Uji Prediksi Contoh...");
            TestPrediction(mlContext, model, "Ular berkepala 5 ditemukan di desa Konoha");
            TestPrediction(mlContext, model, "Pemerintah Konoha membuka program beasiswa baru tahun ini");
        }

        
        private static void CombineDatasets()
        {
            Console.WriteLine("Menggabungkan true.csv dan fake.csv...");

            var trueLines = File.ReadAllLines(TRUE_DATASET_PATH).Skip(1);
            var fakeLines = File.ReadAllLines(FAKE_DATASET_PATH).Skip(1);

            using (var writer = new StreamWriter(COMBINED_DATASET_PATH))
            {
                writer.WriteLine("title,text,subject,date,label");

                void WriteSafe(string line, string label)
                {
                    
                    var parts = line.Split(',');
                    if (parts.Length < 4)
                        return; 

                    
                    for (int i = 0; i < 4; i++)
                    {
                        if (!parts[i].StartsWith("\""))
                            parts[i] = "\"" + parts[i].Replace("\"", "\"\"") + "\"";
                    }

                    writer.WriteLine(string.Join(",", parts[0], parts[1], parts[2], parts[3], label));
                }

                foreach (var line in trueLines)
                    WriteSafe(line, "false");

                foreach (var line in fakeLines)
                    WriteSafe(line, "true");
            }

            Console.WriteLine($"✅ File {COMBINED_DATASET_PATH} berhasil dibuat dengan format aman.");
        }


        private static void TestPrediction(MLContext mlContext, ITransformer model, string text)
        {
            var engine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);

            var sample = new ModelInput { Text = text };
            var result = engine.Predict(sample);

            Console.WriteLine($"Teks: {text}");
            Console.WriteLine($"Prediksi: {(result.PredictedLabel ? "HOAX" : "BENAR")}");
            Console.WriteLine($"Probabilitas: {result.Probability:P2}");
            Console.WriteLine("-----------------------------------");
        }
    }
}
