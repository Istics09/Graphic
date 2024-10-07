using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System;
using System.Drawing;
using System.IO;
using System.Xml.Linq;

static void Main()
{
    string annotationsDir = "Photos/Annotations"; 
    string imagesDir = "Photos/images"; 
    string testImageDir = "test";  

    Console.WriteLine("Képek feldolgozása elkezdődött...");
    //ProcessTestImage(testImageDir, imagesDir, annotationsDir);
    DetectMissingHole(imagesDir, testImageDir);
    Console.WriteLine("Képek feldolgozása befejeződött.");
}

static void DetectMissingHole(string imagesDir, string testImageDir)
{
    // Tesztkép betöltése
    string testImagePath = Directory.GetFiles(testImageDir, "*.jpg")[0];
    Image<Bgr, Byte> img = new Image<Bgr, Byte>(testImagePath);

    // Szürkeárnyalatos és bináris képek létrehozása
    var gray = img.Convert<Gray, byte>()
                    .SmoothGaussian(5)
                    .ThresholdBinaryInv(new Gray(40), new Gray(255));
    CvInvoke.Imwrite("_gray.png", gray);

    // Csukás (Closing) művelet alkalmazása
    Mat kernelCircle = CvInvoke.GetStructuringElement(ElementShape.Ellipse, new Size(5, 5), new Point(-1, -1));
    Mat closedImage = new Mat();
    CvInvoke.MorphologyEx(gray, closedImage, MorphOp.Close, kernelCircle, new Point(-1, -1), 1, BorderType.Default, new MCvScalar(0));
    CvInvoke.Imwrite("closed_image.png", closedImage);

    // Szürkeárnyalatos kép használata a Hough Circle detektáláshoz
    CircleF[] circles = CvInvoke.HoughCircles(
        closedImage,              // Szürkeárnyalatos képet használunk
        HoughModes.Gradient,      // Hough Circle módszer
        2.5,                      // Akkumulátor felbontása
        50.0,                     // Minimum távolság a körök között
        1000.0,                   // Canny küszöb
        30.0,                     // A kör középpontjának küszöbértéke
        15,                       // Minimum sugár
        30                        // Maximum sugár
    );

    // Körök kirajzolása a színes képen
    Image<Bgr, byte> outputImage = img.Clone();

    // Csak a fekete kör, fehér közepű köröket vizsgáljuk
    foreach (var circle in circles)
    {
        Point center = new Point((int)circle.Center.X, (int)circle.Center.Y);
        int radius = (int)circle.Radius;

        // Kör területének vizsgálata (körvonal fekete-e)
        Rectangle boundingBox = new Rectangle(center.X - radius, center.Y - radius, radius * 2, radius * 2);
        Mat circleROI = new Mat(closedImage, boundingBox);
        Image<Gray, byte> circleArea = circleROI.ToImage<Gray, byte>();

        // Fekete külső rész (minimális fekete pixelek aránya)
        int totalPixelCount = boundingBox.Width * boundingBox.Height;
        int blackPixelCount = totalPixelCount - CvInvoke.CountNonZero(circleArea);  // Számoljuk a fekete pixeleket

        double blackPixelRatio = (double)blackPixelCount / totalPixelCount;

        // Középpont vizsgálata (fehér-e a közép)
        if (center.X >= 0 && center.X < closedImage.Width && center.Y >= 0 && center.Y < closedImage.Height)
        {
            // Pixelérték lekérése a szürkeárnyalatos képből
            byte centerIntensity = circleArea.Data[center.Y - boundingBox.Y, center.X - boundingBox.X, 0]; // Pixel érték a középpontban

            if (centerIntensity == 255 && blackPixelRatio > 0.7)  // Ha a középpont fehér és a külső részek fekete
            {
                // Piros kör rajzolása
                CvInvoke.Circle(outputImage, center, radius, new MCvScalar(0, 0, 255), 2);
            }
        }
    }

    // Eredmény mentése
    CvInvoke.Imwrite("A.png", outputImage);
    Console.WriteLine("Eredmény elmentve: A.png");
}



Main();


