using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
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
    // Tesztkép betöltése a "test" mappából
    string testImagePath = Directory.GetFiles(testImageDir, "*.jpg")[0];
    Image<Bgr, Byte> testImg = new Image<Bgr, Byte>(testImagePath);
    // Kép betöltése
    Image<Bgr, Byte> img = new Image<Bgr, Byte>(testImagePath);
    string testImageName = Path.GetFileNameWithoutExtension(testImagePath);
    Console.WriteLine($"Teszt kép betöltve: {Path.GetFileName(testImagePath)}");

    // Kép előfeldolgozása
    Image<Gray, byte> thr = new(img.Size);
    Image<Gray, byte> gray = new(img.Size);

    CvInvoke.CvtColor(img, gray, ColorConversion.Bgr2Gray);
    CvInvoke.Imwrite("_gray.png", gray);

    CvInvoke.Threshold(gray, thr, 35, 255, ThresholdType.Binary);
    CvInvoke.Imwrite("_thr.png", thr);

    // Struktúraelem létrehozása (kör alakú kernel)
    Mat kernel = CvInvoke.GetStructuringElement(ElementShape.Ellipse, new Size(7, 7), new Point(-1, -1));

    // Első dilatáció a fehér területek megnövelésére
    Image<Gray, byte> dilated = new Image<Gray, byte>(thr.Size);
    CvInvoke.Dilate(thr, dilated, kernel, new Point(-1, -1), 2, BorderType.Default, new MCvScalar(0));


}


Main();


