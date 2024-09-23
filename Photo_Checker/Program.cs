using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;
using System.Drawing;
using Emgu.CV.Util;

Console.WriteLine("asd: ");
string name = Console.ReadLine();
CompareImages(name);

static void CompareImages(string name)
{
    // Képek betöltése
    Image<Bgr, Byte> img1 = new Image<Bgr, Byte>("test1/01.jpg");
    Image<Bgr, Byte> img2 = new Image<Bgr, Byte>("test1/" + name + ".jpg");

  
    Image<Bgr, Byte> diff1 = img1.AbsDiff(img2);  
    Image<Bgr, Byte> diff2 = img2.AbsDiff(img1);  


    Image<Gray, Byte> grayDiff1 = diff1.Convert<Gray, Byte>();
    Image<Gray, Byte> grayDiff2 = diff2.Convert<Gray, Byte>();

 
    Image<Gray, Byte> thresholdImg1 = grayDiff1.ThresholdBinary(new Gray(40), new Gray(255));
    Image<Gray, Byte> thresholdImg2 = grayDiff2.ThresholdBinary(new Gray(40), new Gray(255));

    // Kontúrok keresése a különbség képben az első képen
    VectorOfVectorOfPoint contours1 = new VectorOfVectorOfPoint();
    Mat hierarchy1 = new Mat();
    CvInvoke.FindContours(thresholdImg1, contours1, hierarchy1, RetrType.External, ChainApproxMethod.ChainApproxSimple);

    // Kontúrok keresése a különbség képben a második képen
    VectorOfVectorOfPoint contours2 = new VectorOfVectorOfPoint();
    Mat hierarchy2 = new Mat();
    CvInvoke.FindContours(thresholdImg2, contours2, hierarchy2, RetrType.External, ChainApproxMethod.ChainApproxSimple);

    // Kontúrok köré kis piros karikák rajzolása a második képen (csak ami a másodikon hiányzik és az elsőn megvan)
    for (int i = 0; i < contours2.Size; i++)
    {
        // Csak akkor rajzoljunk az első képen levő hibát, ha a különbség csak ott található meg.
        CircleF circle = CvInvoke.MinEnclosingCircle(contours2[i]);
        CvInvoke.Circle(img2, new Point((int)circle.Center.X, (int)circle.Center.Y), 5, new MCvScalar(0, 0, 255), 2); // piros kör a második képen
    }

    img2.Save("marked_image.jpg");  
}

