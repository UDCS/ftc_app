package com.qualcomm.ftcrobotcontroller.opmodes;

import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.hardware.Camera;
import android.os.AsyncTask;
import android.os.Debug;
import android.os.Environment;
import android.util.Log;
import android.widget.FrameLayout;
import android.widget.ImageView;

import com.qualcomm.ftcrobotcontroller.CameraPreview;
import com.qualcomm.ftcrobotcontroller.FtcRobotControllerActivity;
import com.qualcomm.ftcrobotcontroller.R;
import com.qualcomm.robotcore.eventloop.opmode.LinearOpMode;
import com.qualcomm.robotcore.hardware.DcMotor;
import com.qualcomm.robotcore.hardware.Servo;
import com.qualcomm.robotcore.util.Range;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.android.Utils;
import org.opencv.video.Video;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;

import java.io.*;
import java.util.List;
import java.util.Scanner;

import org.apache.commons.io.FileUtils;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.FileSplit;
import org.canova.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

public class VisionGather extends LinearOpMode{

    final static int LOOKLEFT = 0;
    final static int LOOKFORWARD = 1;
    final static int LOOKRIGHT = 2;
    final static int DONERIGHT = 3;
    final static int DONELEFT = 4;
    static int lookState = 0;

    DcMotor Lmotor, Rmotor;
    Servo  ServoA1, ServoA2;

    /* Camera fields */
    private Camera camera;
    public CameraPreview preview;
    public Bitmap image;
    private int width;
    private int height;
    private YuvImage yuvImage = null;
    private int looped = 0;
    private String data;


    static Mat cvOrigImage;
    static Mat cvBWImage;
    static Mat cvColorImage;
    static Mat cvWorkingImage;
    static ImageView cvLayout;
    static Bitmap cvBitmap;
    static Mat kernel1; // for image processing
    static MatOfPoint2f toPoints; // for flow
    static MatOfPoint2f fromPoints; // for flow
    static Mat cvLastImage;               // for flow
    static Boolean CVImageProcessingFlag;
    static long lastTime;   // last time points were grabbed.
    static long currentTime;   // time when points were grabbed.
    static int FLOW_STATE;
    static long nextTime;
    static double[] globalSum;

    static int leftVal;

    @Override
    public void runOpMode() throws InterruptedException{
        Lmotor = hardwareMap.dcMotor.get("m1");
        Rmotor = hardwareMap.dcMotor.get("m2");
        ServoA1 = hardwareMap.servo.get("s1");
        ServoA2 = hardwareMap.servo.get("s2");

        Rmotor.setDirection(DcMotor.Direction.REVERSE);

        /* Camera stuff */
        camera = Camera.open(0);
        camera.setPreviewCallback(previewCallback);

        camera.setDisplayOrientation(90);
        Camera.Parameters parameters = camera.getParameters();
        parameters.setPreviewSize(240, 160);
        List<Camera.Size> x = parameters.getSupportedPreviewSizes();
        List<Integer> y = parameters.getSupportedPreviewFormats();
        camera.setParameters(parameters);
        data = parameters.flatten();

        ((FtcRobotControllerActivity) hardwareMap.appContext).initPreview(camera, this, previewCallback);
        cvLayout = ((FtcRobotControllerActivity) hardwareMap.appContext).cvLayout;
        new OpenCVSetup().execute();
        cvBitmap = Bitmap.createBitmap(160, 240, Bitmap.Config.ARGB_8888);
        CVImageProcessingFlag = false;
        nextTime = System.currentTimeMillis();
        globalSum = new double[3];

        waitForStart();
        while(opModeIsActive()){

//            doSingleJoystickDrive();
            doGoToTheLightDrive(leftVal);
            ServoA2.setPosition(0.6);

            switch(lookState){
                case LOOKLEFT:
                    ServoA1.setPosition(0.55);
                    break;

                case DONELEFT:
                    if(System.currentTimeMillis() > nextTime)
                        lookState = LOOKRIGHT;
                    break;

                case LOOKRIGHT:
                    ServoA1.setPosition(0.55);
                    break;

                case DONERIGHT:
                    if(System.currentTimeMillis() > nextTime)
                        lookState = LOOKLEFT;
                    break;

                case LOOKFORWARD:
                    ServoA1.setPosition(0.55);
                    break;
            }

            if(gamepad1.x)
                lookState = LOOKLEFT;
            if(gamepad1.y)
                lookState = LOOKFORWARD;
            if(gamepad1.b)
                lookState = LOOKRIGHT;

            telemetry.addData("left", globalSum[0]);
            telemetry.addData("middle", globalSum[1]);
            telemetry.addData("right", globalSum[2]);
            telemetry.addData("preference", largestVal(globalSum[0], globalSum[1], globalSum[2]));

            waitForNextHardwareCycle();

        }
        Lmotor.setPower(0);
        Rmotor.setPower(0);
    }

    private String largestVal(double l, double c, double r){
        if(l > c && l > r){
            return "left";
        }else if(r > c)
            return "right";
        else
            return "center";
    }


    /**
     * Mode for using one joystick
     */
    private void doGoToTheLightDrive(int turnLeft) {

        int left = turnLeft;
        telemetry.addData("LEFT:", left + " and " + turnLeft);

        float speed = -0.5f;
        float turn  = -left / 480.0f;
        float l_left_drive_power
                = (float) scale_motor_power(speed - turn);
        float l_right_drive_power
                = (float) scale_motor_power(speed + turn);
        telemetry.addData("s, t = ", "" + speed + ", " + turn);

        set_drive_power(l_left_drive_power, l_right_drive_power);


    }



    /* Camera Stuff */
    private Camera.PreviewCallback previewCallback = new Camera.PreviewCallback() {
        public void onPreviewFrame(byte[] data, Camera camera)
        {
            Camera.Parameters parameters = camera.getParameters();
            width = parameters.getPreviewSize().width;
            height = parameters.getPreviewSize().height;
            yuvImage = new YuvImage(data, ImageFormat.NV21, width, height, null);
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            yuvImage.compressToJpeg(new Rect(0, 0, width, height), 100, out);
            byte[] imageBytes = out.toByteArray();
            image = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
            int W = image.getWidth();
            int H = image.getHeight();

            Utils.bitmapToMat(image, cvOrigImage);
            new OpenCVProcess().execute();

            if(cvLayout != null && cvColorImage != null) {
                Utils.matToBitmap(cvColorImage, cvBitmap);
                cvLayout.setImageBitmap(cvBitmap);
            }
        }
    };


    /**
     * Method to set up the CV data structures that we will be using
     *
     * OpenCV stuff can't be done on the UI (main) thread. Must be done on a background thread.
     */
    public static class OpenCVSetup extends AsyncTask<Void, Void, Void> {
        @Override
        protected Void doInBackground(Void... params) {
            cvOrigImage = new Mat(160, 240, CvType.CV_8UC4);
            cvBWImage   = new Mat(160, 240, CvType.CV_8UC1);
            cvColorImage   = new Mat(240, 160, CvType.CV_8UC4);
            cvLastImage   = new Mat(240, 160, CvType.CV_8UC1);
            cvWorkingImage   = new Mat(240, 160, CvType.CV_8UC4);
            toPoints   = initPoints(cvWorkingImage.width(), cvWorkingImage.height(), 10);
            fromPoints = initPoints(cvWorkingImage.width(), cvWorkingImage.height(), 20);

            // Build a kernel
            int N = 3;
            kernel1 = new Mat(new Size(N, N), CvType.CV_64F);
            double[] values = {-1,0,1,-1,0,1,-1,0,1};
            for(int i = 0; i < N; i++){
                for(int j = 0; j < N; j++){
                    double entry = values[i*N + j];
                    kernel1.put(i, j, entry);
                }
            }

            return null;
        }
    }


    /**
     * Method for processing the image using OpenCV methods
     *
     * OpenCV stuff can't be done on the UI (main) thread. Must be done on a background thread.
     */
    public static class OpenCVProcess extends AsyncTask<Void, Void, Void> {
        @Override
        protected Void doInBackground(Void... params) {
            // Orient the image properly
            if(CVImageProcessingFlag == true || lookState == DONELEFT || lookState == DONERIGHT) {
                cvColorImage = null;
                return null;
            }
            CVImageProcessingFlag = true;

            Imgproc.cvtColor(cvOrigImage, cvBWImage, Imgproc.COLOR_BGR2GRAY);
            Imgproc.equalizeHist(cvBWImage, cvBWImage);

            Core.transpose(cvBWImage, cvBWImage);
            Core.flip(cvBWImage, cvBWImage, 1);

            // Compute the flow
            MatOfByte flowStatus = new MatOfByte();
            MatOfFloat err = new MatOfFloat();
            Video.calcOpticalFlowPyrLK(cvLastImage, cvBWImage, fromPoints, toPoints, flowStatus, err);

            Imgproc.cvtColor(cvBWImage, cvWorkingImage, Imgproc.COLOR_GRAY2BGRA);
            Point[] fpArray = fromPoints.toArray();
            Point[] tpArray = toPoints.toArray();
            float[] eArray = err.toArray();
            byte[]  b = flowStatus.toArray();

            // Bins to hold segment lengths
            ArrayList<ArrayList<Integer>> bins = new ArrayList<ArrayList<Integer>> ();
            int numBins = 3;
            for(int i = 0; i < numBins; i++)
                bins.add(new ArrayList<Integer>());

            for(int i = 0; i < fpArray.length; i++){
                if(b[i] == 1 && dist(fpArray[i], tpArray[i]) < 20) {
                    // draw the segment to the screen
                    Imgproc.line(cvWorkingImage, fpArray[i], tpArray[i], new Scalar(255, 0, 255));
                    // Place the segment into the appropriate bin
                    int binNumber = (int)(fpArray[i].x * numBins) / cvWorkingImage.cols();
                    bins.get(binNumber).add((int)dist(fpArray[i], tpArray[i]));
                }
            }

            // Now compute how to turn based on the bin
            int maxi = -1;
            double maxSum = -1;
            for(int i = 0; i < numBins; i++) {
                int numEntries = bins.get(i).size();
                double sum = 0;
                for (int j = 0; j < numEntries; j++) {
                    sum += bins.get(i).get(j);
                }
                if (sum > 0)
                    sum /= numEntries;
                globalSum[i] = sum;

                if (sum > maxi) {
                    maxSum = sum;
                    maxi = i;
                }
            }

            leftVal = 80 + (1-maxi) * 60;
            Log.d("leftVal", "" + leftVal);

            cvColorImage = cvWorkingImage;

            cvLastImage = cvBWImage.clone();

            if(lookState == LOOKLEFT) lookState = DONELEFT;
            if(lookState == LOOKRIGHT) lookState = DONERIGHT;
            nextTime = System.currentTimeMillis() + 500;

            CVImageProcessingFlag = false;

            return null;
        }
    }

    private static double dist(Point p1, Point p2){
        return Math.sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y));
    }

    /**
     * Create a grid of points for following with our flow
     *
     * @param width Width of image
     * @param height Height of image
     * @param spacing Side of grid square. A point goes in the center of each grid square.
     *                This quantity should divide both width and height.
     */
    private static MatOfPoint2f initPoints(int width, int height, int spacing) {
        Point[] ps = new Point[(width /spacing) * (height/spacing)];

        int counter = 0;

        for (int x = spacing/2; x < width; x += spacing) {
            for (int y = spacing/2; y < height; y += spacing) {
                ps[counter++] = new Point(x, y);
            }
        }

        return new MatOfPoint2f(ps);
    }


    /**
     * Mode for using one joystick
     */
    private void doSingleJoystickDrive() {
        float speed = -gamepad1.left_stick_y;
        float turn  = gamepad1.right_stick_x / 5;
        float l_left_drive_power
                = (float) scale_motor_power(speed - turn);
        float l_right_drive_power
                = (float) scale_motor_power(speed + turn);
        telemetry.addData("s, t = ", "" + speed + ", " + turn);

        set_drive_power(l_left_drive_power, l_right_drive_power);
    }

    /**
     *
     * @param left power for left wheels
     * @param right power for right wheels
     */
    void set_drive_power (double left, double right)
    {
        Lmotor.setPower(left);
        Rmotor.setPower(right);


    }



    private class SaveImageTask extends AsyncTask<byte[], Void, Void> {

        @Override
        protected Void doInBackground(byte[]... data) {
            FileOutputStream outStream = null;

            // Write to SD Card
            try {
//				File sdCard = Environment.getExternalStorageDirectory();
//				File dir = new File (sdCard.getAbsolutePath() + "/camtest");
                File dir = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES), "DataCollector");
                dir.mkdirs();

                String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
                String fileName = String.format("%s.jpg", timeStamp);
                File outFile = new File(dir, fileName);

                outStream = new FileOutputStream(outFile);
                outStream.write(data[0]);
                outStream.flush();
                outStream.close();

                telemetry.addData("Camera:", "onPictureTaken - wrote bytes: " + data.length + " to " + outFile.getAbsolutePath());
                Log.d("VisionGather", "onPictureTaken - wrote bytes: " + data.length + " to " + outFile.getAbsolutePath());
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            } finally {
            }
            return null;
        }

    }



    double scale_motor_power (double p_power)
    {
        //
        // Assume no scaling.
        //
        double l_scale = 0.0f;

        //
        // Ensure the values are legal.
        //
        double l_power = Range.clip(p_power, -1, 1);

        double[] l_array =
                { 0.00, 0.05, 0.09, 0.10, 0.12
                        , 0.15, 0.18, 0.24, 0.30, 0.36
                        , 0.43, 0.50, 0.60, 0.72, 0.85
                        , 1.00, 1.00
                };

        //
        // Get the corresponding index for the specified argument/parameter.
        //
        int l_index = (int) (l_power * 16.0);
        if (l_index < 0)
        {
            l_index = -l_index;
        }
        else if (l_index > 16)
        {
            l_index = 16;
        }

        if (l_power < 0)
        {
            l_scale = -l_array[l_index];
        }
        else
        {
            l_scale = l_array[l_index];
        }

        return l_scale;

    }



    public void CameraLoop() {
        camera.takePicture(shutterCallback, rawCallback, jpegCallback);
        //  camera.takePicture(null, null, null, null);

        telemetry.addData("Image:", "Image taken");

//    telemetry.addData("Image", Integer.toString(looped) + " images taken");
        Log.d("DEBUG:", data);
    }

    Camera.ShutterCallback shutterCallback = new Camera.ShutterCallback() {
        public void onShutter() {
            //			 Log.d(TAG, "onShutter'd");
        }
    };

    Camera.PictureCallback rawCallback = new Camera.PictureCallback() {
        public void onPictureTaken(byte[] data, Camera camera) {
            //			 Log.d(TAG, "onPictureTaken - raw");
        }
    };

    Camera.PictureCallback jpegCallback = new Camera.PictureCallback() {
        public void onPictureTaken(byte[] data, Camera camera) {
            new SaveImageTask().execute(data);
            //resetCam();
//            Log.d("VisionGather", "onPictureTaken - jpeg");
        }
    };

/*



    /**
     * Method for processing the image using OpenCV methods
     *
     * OpenCV stuff can't be done on the UI (main) thread. Must be done on a background thread.
     *
public static class OpenCVProcess extends AsyncTask<Void, Void, Void> {
    @Override
    protected Void doInBackground(Void... params) {
        Imgproc.cvtColor(cvOrigImage, cvBWImage, Imgproc.COLOR_BGR2GRAY);
        Imgproc.equalizeHist(cvBWImage, cvBWImage);
        Core.transpose(cvBWImage, cvBWImage);
        Core.flip(cvBWImage, cvBWImage, 1);
        Imgproc.filter2D(cvBWImage, cvBWImage, -1, kernel1);
        Imgproc.cvtColor(cvBWImage, cvColorImage, Imgproc.COLOR_GRAY2BGRA);
        return null;
    }
}




            int windowSize = 3;

            int[][] brightnessPS = new int[H][W];

            telemetry.addData("Width", W);
            telemetry.addData("Height", H);
            for(int i = 0; i < H; i++){
                for(int j = 0; j < W; j++){
                    int pixel = image.getPixel(j, i);
                    int brightness = (pixel & 0xFF) + ((pixel >> 8) & 0xFF) + ((pixel >> 16) & 0xFF);
                    brightnessPS[i][j] = brightness;
                    if(i > 0)
                        brightnessPS[i][j] += brightnessPS[i-1][j];
                    if(j > 0)
                        brightnessPS[i][j] += brightnessPS[i][j-1];
                    if(i > 0 && j > 0)
                        brightnessPS[i][j] -= brightnessPS[i-1][j-1];
                }
            }
            int maxi = 0;
            int maxj = 0;
            int maxBrightness = 0;
            for(int i = 1; i < H - windowSize; i++) {
                for (int j = 1; j < W - windowSize; j++) {
                    int brightHere = brightnessPS[i + windowSize][j + windowSize] -
                            brightnessPS[i][j + windowSize] -
                            brightnessPS[i + windowSize][j] +
                            brightnessPS[i][j];
                    if (brightHere > maxBrightness) {
                        maxi = i;
                        maxj = j;
                        maxBrightness = brightHere;
                    }

                }
            }
            telemetry.addData("maxi", maxi);
            telemetry.addData("maxj", maxj);
            //telemetry.addData("r", Math.random());
 */

    /**
     *
     */
    void processModel(MultiLayerNetwork model, String path)
            throws IOException, InterruptedException {
        //create array of strings called labels, read from the subdirectories of the directory below
        List<String> labels = Arrays.asList("circle", "fillCircle", "fillSquare", "line", "square");

        System.out.println("Predicting");

		/*
		//traverse dataset to get each label
		List<String> labels = new ArrayList<>();
		for(File f : new File(path).listFiles()) {
			labels.add(f.getName());
		}
		 */

        // Instantiating RecordReader. Specify height and width of images.
        RecordReader recordReader = new ImageRecordReader(28, 28, true, labels);

        // Point to data path.
        recordReader.initialize(new FileSplit(new File(path)));
        DataSetIterator iter = new RecordReaderDataSetIterator(recordReader, 784, labels.size());

        Resources res = hardwareMap.appContext.getResources();

        //Load network configuration from disk, if needed
        if(model == null){
            MultiLayerConfiguration confFromJson = MultiLayerConfiguration.fromJson(
                    resToString(res.openRawResource(R.raw.neuralnetconf)));

            //Load parameters from disk:
            INDArray newParams;
            DataInputStream dis = null;
            try {
                dis = new DataInputStream(res.openRawResource(R.raw.neuralnetcoeff));
                newParams = Nd4j.read(dis);
            } finally {
                if (dis != null) {
                    dis.close();
                }
            }

            //Create a MultiLayerNetwork from the saved configuration and parameters
            model = new MultiLayerNetwork(confFromJson);
            model.init();
            model.setParameters(newParams);
        }

        //iter = new MnistDataSetIterator(64, false, 12345);

        Evaluation eval = new Evaluation();
        while(iter.hasNext()){
            DataSet next = iter.next();
            next.normalize();
            INDArray predict = model.output(next.getFeatureMatrix());
            eval.eval(next.getLabels(), predict);
        }

        System.out.println(eval.stats());
	}

    private String resToString(InputStream in) {
        return new Scanner(in).useDelimiter("\\A").next();
    }

}
