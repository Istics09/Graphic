using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using System;
using System.Drawing;
using System.IO;
using System.Xml.Linq;

static void Main()
{
    // Az annotációk és a képek mappáinak megadása
    string annotationsDir = "Photos/Annotations";  // Az XML fájlok mappája
    string imagesDir = "Photos/images";  // A többi képek mappája
    string testImageDir = "test";  // A teszt kép mappája

    Console.WriteLine("Képek feldolgozása elkezdődött...");
    ProcessTestImage(testImageDir, imagesDir, annotationsDir);
    Console.WriteLine("Képek feldolgozása befejeződött.");
}

static void ProcessTestImage(string testImageDir, string imagesDir, string annotationsDir)
{
    // Tesztkép betöltése a "test" mappából
    string testImagePath = Directory.GetFiles(testImageDir, "*.jpg")[0];
    Image<Bgr, Byte> testImg = new Image<Bgr, Byte>(testImagePath);
    string testImageName = Path.GetFileNameWithoutExtension(testImagePath);
    Console.WriteLine($"Teszt kép betöltve: {Path.GetFileName(testImagePath)}");

    // Betöltjük az összes többi képet a "Photos/images" mappából
    var imageFiles = Directory.GetFiles(imagesDir, "*.jpg", SearchOption.AllDirectories);

    bool similarImageFound = false;

    // Képek összehasonlítása a tesztképpel
    foreach (var imagePath in imageFiles)
    {
        Image<Bgr, Byte> img = new Image<Bgr, Byte>(imagePath);

        // Ellenőrizzük, hogy a képek mérete egyezik-e, ha nem, átméretezzük
        if (testImg.Size != img.Size)
        {
            CvInvoke.Resize(img, img, testImg.Size);  // Átméretezzük a képet, hogy ugyanakkora legyen, mint a tesztkép
        }

        // Hasonlóság számítása (pl. abszolút különbség)
        Mat diff = new Mat();
        CvInvoke.AbsDiff(testImg, img, diff);

        // A különbség összegzése (minél kisebb, annál hasonlóbb)
        MCvScalar sumDiff = CvInvoke.Sum(diff);  // Sum visszatér egy MCvScalar objektumot
        double totalDiff = sumDiff.V0 + sumDiff.V1 + sumDiff.V2;  // RGB csatornák összegzése

        // Ha a különbség kicsi, hasonló a kép
        if (totalDiff < 100000) // Threshold érték a hasonlóságra
        {
            Console.WriteLine($"Hasonló kép találva: {Path.GetFileName(imagePath)}");

            // Képnév alapján megkeressük az XML fájlt
            string imageName = Path.GetFileNameWithoutExtension(imagePath);
            string xmlFilePath = FindXmlForImage(imageName, annotationsDir);

            if (xmlFilePath != null)
            {
                Console.WriteLine($"XML fájl megtalálva: {xmlFilePath}");

                // XML fájl alapján megjelöljük a hibás területeket a tesztképen
                MarkErrorsFromAnnotations(testImg, xmlFilePath);

                // A megjelölt kép mentése
                string outputImagePath = Path.Combine("marked_images", testImageName + "_marked.jpg");
                Directory.CreateDirectory("marked_images");
                testImg.Save(outputImagePath);

                Console.WriteLine($"Feldolgozva és mentve: {outputImagePath}");
                similarImageFound = true;
                break;  // Ha találunk egy hasonló képet és feldolgoztuk, kiléphetünk a ciklusból
            }
        }
    }

    // Ha nincs hasonló kép az adatbázisban, próbáljunk meg hibát találni
    if (!similarImageFound)
    {
        Console.WriteLine("Nem találtunk hasonló képet az adatbázisban, hibafelismerés folyamatban...");

        // Példa: Missing hole felismerése
        DetectMissingHole(testImg);

        // Példa: Short felismerése
        DetectShort(testImg);

        // A megjelölt kép mentése
        string outputImagePath = Path.Combine("marked_images", testImageName + "_detected_marked.jpg");
        Directory.CreateDirectory("marked_images");
        testImg.Save(outputImagePath);

        Console.WriteLine($"Hiba felismerve és mentve: {outputImagePath}");
    }
}

static void DetectMissingHole(Image<Bgr, Byte> img)
{
    // Kép előfeldolgozása (szürkeárnyalat, zajszűrés)
    Image<Gray, Byte> grayImg = img.Convert<Gray, Byte>();
    CvInvoke.GaussianBlur(grayImg, grayImg, new Size(5, 5), 1.5);

    // Körök keresése (például hiányzó furat)
    CircleF[] circles = CvInvoke.HoughCircles(grayImg, HoughModes.Gradient, 1, 20, 100, 30, 5, 50);

    foreach (var circle in circles)
    {
        // Ha találunk egy kört, jelöljük meg pirossal
        CvInvoke.Circle(img, new Point((int)circle.Center.X, (int)circle.Center.Y), (int)circle.Radius, new MCvScalar(0, 0, 255), 2);
        CvInvoke.PutText(img, "Missing hole", new Point((int)circle.Center.X - 20, (int)circle.Center.Y - 10),
                         Emgu.CV.CvEnum.FontFace.HersheySimplex, 0.5, new MCvScalar(0, 0, 255), 2);
    }
}

static void DetectShort(Image<Bgr, Byte> img)
{
    // Kép éldetektálás (rövidzárlat keresése)
    Image<Gray, Byte> grayImg = img.Convert<Gray, Byte>();
    Image<Gray, Byte> edges = grayImg.Canny(50, 150);

    // Kontúrok keresése az élek alapján
    var contours = new Emgu.CV.Util.VectorOfVectorOfPoint();
    CvInvoke.FindContours(edges, contours, null, Emgu.CV.CvEnum.RetrType.External, Emgu.CV.CvEnum.ChainApproxMethod.ChainApproxSimple);

    // Végigmegyünk az összes kontúron
    for (int i = 0; i < contours.Size; i++)
    {
        // Lekérjük az adott kontúrt
        var contour = contours[i];

        // Bounding box kiszámítása az adott kontúrra
        Rectangle rect = CvInvoke.BoundingRectangle(contour);

        // Ha a bounding box nagyobb egy bizonyos méretnél, akkor jelöljük meg
        if (rect.Width > 10 && rect.Height > 10)  // Csak nagyobb területeket nézünk
        {
            // Jelöljük meg pirossal
            CvInvoke.Rectangle(img, rect, new MCvScalar(0, 0, 255), 2);
            CvInvoke.PutText(img, "Short", new Point(rect.X, rect.Y - 10),
                             Emgu.CV.CvEnum.FontFace.HersheySimplex, 0.5, new MCvScalar(0, 0, 255), 2);
        }
    }
}

static string FindXmlForImage(string imageName, string annotationsDirectory)
{
    Console.WriteLine($"XML fájl keresése a képhez: {imageName}");
    // Megkeressük a megfelelő XML fájlt az annotations mappában és almappáiban
    foreach (string xmlFilePath in Directory.GetFiles(annotationsDirectory, "*.xml", SearchOption.AllDirectories))
    {
        XDocument xmlDoc = XDocument.Load(xmlFilePath);
        string xmlImageName = xmlDoc.Element("annotation")?.Element("filename")?.Value;

        if (xmlImageName != null && Path.GetFileNameWithoutExtension(xmlImageName) == imageName)
        {
            Console.WriteLine($"XML fájl megtalálva: {xmlFilePath}");
            return xmlFilePath;  // Visszatérünk az XML fájl elérési útvonalával
        }
    }
    Console.WriteLine($"Nem található XML fájl a képhez: {imageName}");
    return null;  // Ha nincs meg a képhez tartozó XML fájl
}

static void MarkErrorsFromAnnotations(Image<Bgr, Byte> img, string xmlFilePath)
{
    // XML fájl betöltése
    XDocument xmlDoc = XDocument.Load(xmlFilePath);

    // Végigmegyünk az XML-ben található hibákon
    foreach (var obj in xmlDoc.Descendants("object"))
    {
        // Hiba típusának beolvasása
        string defectType = obj.Element("name")?.Value;

        // Bounding box koordináták az XML fájlból
        int xmin = int.Parse(obj.Element("bndbox").Element("xmin").Value);
        int ymin = int.Parse(obj.Element("bndbox").Element("ymin").Value);
        int xmax = int.Parse(obj.Element("bndbox").Element("xmax").Value);
        int ymax = int.Parse(obj.Element("bndbox").Element("ymax").Value);

        // Bounding box megrajzolása a képen piros színnel
        CvInvoke.Rectangle(img, new Rectangle(xmin, ymin, xmax - xmin, ymax - ymin), new MCvScalar(0, 0, 255), 2);

        // Hiba típusának kiírása a bounding box fölé
        CvInvoke.PutText(img, defectType, new Point(xmin, ymin - 10), Emgu.CV.CvEnum.FontFace.HersheySimplex, 0.5, new MCvScalar(0, 0, 255), 2);
    }
}

Main();
