using Microsoft.AspNetCore.Mvc;
using RealTimeML.Server.Services;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace RealTimeML.Server.Controllers
{
    [Route("/Predict")]
    public class PredictionController : Controller
    {
        private readonly LeNetService _leNetService;

        public PredictionController(LeNetService leNetService)
        {
            _leNetService = leNetService;
        }

        [HttpPost("/LeNet/Train")]
        public async Task<IActionResult> LeNet()
        {
            await _leNetService.Train();
            return Ok();
        }
        [HttpPost("/LeNet/Predict")]
        public async Task<IActionResult> LeNet(IFormFile formFile)
        {
            if (formFile.Length > 0)
            {
                using (var memoryStream = new MemoryStream())
                {
                    await formFile.CopyToAsync(memoryStream);
                    byte[] fileBytes = memoryStream.ToArray();

                    // Byte dizisini Predict fonksiyonuna iletme
                    byte[] preparedImage = PrepareImageForLeNet(fileBytes);
                    int prediction = _leNetService.Predict(preparedImage);

                    // Prediction sonucunu geri döndürme
                    return Ok(prediction);
                }
            }
            else
            {
                return BadRequest("Dosya boş olamaz.");
            }
        }
        public static byte[] PrepareImageForLeNet(byte[] imageBytes)
        {
            // Görüntüyü byte dizisinden ImageSharp görüntüsüne yükle
            using (Image<L8> image = Image.Load<L8>(imageBytes))
            {
                // Görüntüyü 28x28 piksele yeniden boyutlandır
                image.Mutate(x => x.Resize(28, 28));

                byte[] pixelData = new byte[28 * 28];

                for (int y = 0; y < 28; y++)
                {
                    for (int x = 0; x < 28; x++)
                    {
                        // Her bir pikselin yoğunluğunu al (L8 formatı zaten gri tonlamalıdır)
                        pixelData[y * 28 + x] = image[x, y].PackedValue;
                    }
                }

                return pixelData;
            }
        }
    }
}
