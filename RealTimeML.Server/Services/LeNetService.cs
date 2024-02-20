using System.Diagnostics;
using Tensorflow;
using Tensorflow.Keras.Engine;
using Tensorflow.NumPy;
using static Tensorflow.KerasApi;

namespace RealTimeML.Server.Services
{
    public class LeNetService
    {
        const float learningRate = 0.01f;
        const int batchSize = 128;
        const int epochs = 20;

        private Sequential baseModel;
        private Sequential predictModel;


        private NDArray trainImages;
        private NDArray trainLabels;

        private NDArray testImages;
        private NDArray testLabels;

        public LeNetService() 
        { 

        }

        public async Task Train()
        {
            PrepareData();

            PrepareModel();

            // Train the model
            {
                var stopwatch = new Stopwatch();
                Console.WriteLine("Starting training...");
                stopwatch.Start();

                baseModel.fit(trainImages, trainLabels, batchSize, epochs);

                stopwatch.Stop();
                Console.WriteLine($"Took {stopwatch.ElapsedMilliseconds / 1000} seconds");
            }

            baseModel.evaluate(testImages, testLabels);
            
        }

        public void PrepareData() 
        {
            // Load the MNIST dataset
            var data = keras.datasets.mnist.load_data();

            (trainImages, trainLabels) = data.Train;
            (testImages, testLabels) = data.Test;

            // Pre-process it, turning them all into doubles between 0-1 instead of bytes between 0-255. We do this even
            // though we receive data from the user as raw bytes (0-255), neural networks work better with doubles than
            // integer

            trainImages /= 255.0f;
            testImages /= 255.0f;



            trainImages = np.expand_dims(trainImages, -1);  // Eğitim veri seti için
            testImages = np.expand_dims(testImages, -1);    // Test veri seti için


        }
        public void PrepareModel()
        {
            baseModel = keras.Sequential();
            baseModel.add(keras.layers.Input((28, 28, 1)));
            baseModel.add(keras.layers.Conv2D(6, kernel_size: (5, 5), padding: "VALID", activation: "relu"));
            baseModel.add(keras.layers.MaxPooling2D(pool_size: (2, 2), strides: (2, 2), padding: "same"));
            baseModel.add(keras.layers.Conv2D(16, kernel_size: (5, 5), padding: "VALID", activation: "relu"));
            baseModel.add(keras.layers.MaxPooling2D(pool_size: (2, 2), strides: (2, 2), padding: "same"));
            baseModel.add(keras.layers.Flatten());
            baseModel.add(keras.layers.Dense(120, activation: "relu"));
            baseModel.add(keras.layers.Dense(84, activation: "relu"));
            baseModel.add(keras.layers.Dense(10));

            baseModel.compile(
                loss: keras.losses.SparseCategoricalCrossentropy(from_logits: true),
                optimizer: keras.optimizers.Adam(learningRate),
                metrics: new string[] { "accuracy" }
            );
        }
        public int Predict(byte[] data)
        {
            
            if (data.Length != 28 * 28)
            {
                Console.Error.WriteLine("Data must be exactly 28x28 (784) bytes. Truncating or padding to match.");
                Array.Resize(ref data, 28 * 28);
            }

            // Reshape our input data into 1x (28x28) arrays (AKA one 1x28x28 tensor, AKA a 3D vector)
            var array = np.array(data);
            array = array.reshape((1, 28, 28,1));
            array = array / 255.0f;

            // Create a copy of the same model, but with a Softmax layer at the end to get outputs.
            if (predictModel == null)
            {
                predictModel = keras.Sequential();
                predictModel.add(baseModel);
                predictModel.add(keras.layers.Softmax(-1));
                // predictionModel.compile();
            }

            Tensors predictions = predictModel.predict(array);

            var prediction = predictions[0].numpy().ToArray<float>();

            // En yüksek olasılığa sahip sınıfın indeksini bulma
            int predictedClassIndex = Array.IndexOf(prediction, prediction.Max());

            // Sınıf indeksi rakamı temsil ediyor
            return predictedClassIndex;
        }
        public void Test()
        {
            int correctPredictions = 0;
            int totalPredictions = (int)testImages.shape[0]; // Toplam test görüntüsü sayısı

            for (int i = 0; i < totalPredictions; i++)
            {
                byte[] imageData = testImages[i].BufferToArray(); // Test görüntüsünü al
                int actualLabel = (int)testLabels[i]; // Gerçek etiket

                // Test görüntüsünü modelinize vererek tahmin edilen sınıfı alın
                int predictedLabel = Predict(imageData);

                // Tahmin edilen sınıf ile gerçek etiketi karşılaştırın
                if (predictedLabel == actualLabel)
                {
                    correctPredictions++;
                }
            }

            // Doğruluk oranını hesaplayın
            double accuracy = (double)correctPredictions / totalPredictions;

            Console.WriteLine($"Accuracy: {accuracy * 100}%");
        }



    }
}
