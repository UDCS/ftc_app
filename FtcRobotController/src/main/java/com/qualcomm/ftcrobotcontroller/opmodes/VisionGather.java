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

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.io.StringWriter;
import java.io.UnsupportedEncodingException;
import java.io.Writer;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

public class VisionGather extends LinearOpMode{

    final static int LOOKLEFT = 0;
    final static int LOOKFORWARD = 1;
    final static int LOOKRIGHT = 2;
    final static int TURNING = 3;
    final static int NOT_TURNING = 4;
    static int turnState = 4;
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

    // Neural Net fields
    MyModel model = null;
    float predictionHistory = 3.0f;
    float MEMORY = 0.3f;
    int   QSIZE  = 11;
    LinkedList<Integer> predictionQueue;

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
    static long nextTime;
    static double[] globalSum;

    /* Adjustable fields */
    float predictAdjust = 0.0f;
    float ratio = 10f;
    static enum Field {
        PREDICTADJUST, RATIO, MEMORY
    }
    Field curField = Field.MEMORY;
    boolean pressed = false;


    @Override
    public void runOpMode() throws InterruptedException{
        Lmotor = hardwareMap.dcMotor.get("m1");
        Rmotor = hardwareMap.dcMotor.get("m2");
        ServoA1 = hardwareMap.servo.get("s1");
        ServoA2 = hardwareMap.servo.get("s2");

        Lmotor.setDirection(DcMotor.Direction.REVERSE);

        /* Camera stuff */
        camera = Camera.open(0);
        camera.setPreviewCallback(previewCallback);

        camera.setDisplayOrientation(90);
        Camera.Parameters parameters = camera.getParameters();
        parameters.setPreviewSize(240, 160);
        //parameters.setPreviewSize(1280, 720);
        List<Camera.Size> x = parameters.getSupportedPreviewSizes();
        List<Integer> y = parameters.getSupportedPreviewFormats();
        camera.setParameters(parameters);
        data = parameters.flatten();

        // Set up camera and OpenCV
        ((FtcRobotControllerActivity) hardwareMap.appContext).initPreview(camera, this, previewCallback);
        cvLayout = ((FtcRobotControllerActivity) hardwareMap.appContext).cvLayout;
        new OpenCVSetup().execute();
        cvBitmap = Bitmap.createBitmap(160, 240, Bitmap.Config.ARGB_8888);
        CVImageProcessingFlag = false;
        nextTime = System.currentTimeMillis();
        globalSum = new double[3];

        // Set up the neural network
        model = new MyModel(new int[] {160, 600, 3000, 6});
        model.read(R.raw.coeffplain);
        model.setMeanAndDev(R.raw.meanvarcsv);
        predictionQueue = new LinkedList<Integer>();

        waitForStart();
        while(opModeIsActive()){

            //doSingleJoystickDrive();
            //doGoToTheLightDrive();
            doNeuralNetDrive();
            ServoA2.setPosition(0.6);
            ServoA1.setPosition(0.55);

            if(gamepad1.x)
//                lookState = LOOKLEFT;
                curField = Field.MEMORY;
            if(gamepad1.y)
//                lookState = LOOKFORWARD;
                curField = Field.RATIO;
            if(gamepad1.b)
                curField = Field.PREDICTADJUST;
//                lookState = LOOKRIGHT;

            int inc = 0;
            if(!pressed) {
                if (gamepad1.dpad_up) {
                    inc = 1;
                    pressed = true;
                }else if (gamepad1.dpad_down) {
                    inc = -1;
                    pressed = true;
                }
            } else
            if(!gamepad1.dpad_up && !gamepad1.dpad_down)
                pressed = false;

            switch(curField){
                case MEMORY:
                    MEMORY += inc * 0.05f;
                    break;
                case RATIO:
                    ratio += inc * 1f;
                    break;
                case PREDICTADJUST:
                    predictAdjust += inc * 0.1f;
                    break;
                default:
                    break;
            }

            telemetry.addData("memory:", MEMORY);
            telemetry.addData("ratio:", ratio);
            telemetry.addData("predictAdjust:", predictAdjust);

            telemetry.addData("current field:", curField.toString());

            waitForNextHardwareCycle();

        }
        Lmotor.setPower(0);
        Rmotor.setPower(0);
        if (camera != null) {
            camera.stopPreview();
            camera.release();
            camera = null;
        }

    }


    /**
     * Mode for using one joystick
     */
    private void doNeuralNetDrive() {
        telemetry.addData("history:", "" + predictionHistory);

        float speed = 0.3f;
        float rightBias  = (predictionHistory - predictAdjust)/ratio;
        telemetry.addData("rightBias:", "" + rightBias);
        float l_left_drive_power = speed + rightBias;
        //= (float) scale_motor_power(speed + rightBias);
        float l_right_drive_power = speed - rightBias;
        //= (float) scale_motor_power(speed - rightBias);
        //telemetry.addData("s, t = ", "" + speed + ", " + turn);
        telemetry.addData("Powers: ", "" + l_left_drive_power + " : " + l_right_drive_power);
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
            telemetry.addData("model", model == null ? "null" : model.hello());
        }
    };


    /**
     * Method to set up the CV data structures that we will be using
     *
     * OpenCV stuff can't be done on the UI (main) thread. Must be done on a background thread.
     */
    public class OpenCVSetup extends AsyncTask<Void, Void, Void> {
        @Override
        protected Void doInBackground(Void... params) {
            cvOrigImage = new Mat(160, 240, CvType.CV_8UC4);
            cvBWImage   = new Mat(160, 240, CvType.CV_8UC1);
            cvColorImage   = new Mat(240, 160, CvType.CV_8UC4);
            cvLastImage   = new Mat(240, 160, CvType.CV_8UC1);
            cvWorkingImage   = new Mat(240, 160, CvType.CV_8UC4);
            toPoints   = initPoints(cvWorkingImage.width(), cvWorkingImage.height(), 40);
            fromPoints = initPoints(cvWorkingImage.width(), cvWorkingImage.height(), 40);

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
    public class OpenCVProcess extends AsyncTask<Void, Void, Void> {
        @Override
        protected Void doInBackground(Void... params) {
            // Orient the image properly
            Imgproc.cvtColor(cvOrigImage, cvBWImage, Imgproc.COLOR_BGR2GRAY);
            Imgproc.equalizeHist(cvBWImage, cvBWImage);

            // Rotate the image
            Core.transpose(cvBWImage, cvBWImage);
            Core.flip(cvBWImage, cvBWImage, 1);
            Imgproc.filter2D(cvBWImage, cvBWImage, -1, kernel1);

            // Compute the column sums
            int cols = cvBWImage.cols();
            float[] featureVector = new float[cols];
            for(int j = 0; j < cols; j++){
                float sum = 0.0f;
                for(int i = 0; i < cvBWImage.rows(); i++)
                    sum += cvBWImage.get(i, j)[0];
                featureVector[j] = sum * 213.0f/240.0f;
            }

            Imgproc.cvtColor(cvBWImage, cvColorImage, Imgproc.COLOR_GRAY2BGRA);
            int prediction = model.predict(featureVector);
            Imgproc.putText(cvColorImage, String.valueOf(prediction),
                    new Point(20, 120),
                    Core.FONT_HERSHEY_TRIPLEX, 2.0, new Scalar(0, 255, 0));

            String[] predString = {"Stop", "Hard Left", "Left", "Forward", "Right", "Hard Right"};
            telemetry.addData("prediction:", predString[prediction]);
            if(prediction != 0)
                predictionHistory = MEMORY * predictionHistory + (1-MEMORY) * prediction;

            // Manage the prediction queue
            if(predictionQueue != null) {
                predictionQueue.add(prediction);
                if (predictionQueue.size() >= QSIZE) {
                    String telstr = "";
                    int l = 0, f = 0, r = 0;
                    for (int p : predictionQueue) {
                        if (p == 1 || p == 2) l++;
                        if (p == 3) f++;
                        if (p == 4 || p == 5) r++;
                        telstr = telstr + p;
                    }
                    int thresh = 2;
                    if (l > f + thresh && l > r + thresh)
                        predictionHistory = 2;
                    else if (r > f + thresh && r > l + thresh)
                        predictionHistory = 4;
                    else
                        predictionHistory = 3;

                    telemetry.addData("Q:", telstr);
                    predictionQueue.removeFirst();
                }
            }


            return null;
        }
    }

    private static double xdist(Point p1, Point p2){
        return Math.abs(p1.x - p2.x);
    }

    private static double dist(Point p1, Point p2){
        return Math.sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
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


    public class MyModel{
        private float[][][] weights;
        private float[][]   biases;
        private float[]      means;  // For each input node
        private float[]    stddevs;  // For each input node
        private int      numInputs;  // size of the input layer

        int numLayers = 0;
        String helloString = "0";
        private boolean initialized = false;

        /**
         * Constructor, takes an array of the sizes of all node levels, including in and out.
         *
         * @param sizes
         */
        public MyModel(int[] sizes){
            numLayers = sizes.length;
            numInputs = sizes[0];

            weights = new float[numLayers-1][][];
            for(int i = 0; i < numLayers-1; i++){
                weights[i] = new float[sizes[i+1]][sizes[i]];
            }

            biases = new float[numLayers][];
            for(int i = 1; i < numLayers; i++)
                biases[i] = new float[sizes[i]];

            means   = new float[numInputs];
            stddevs = new float[numInputs];
            helloString += ".i";
        }

        /**
         * Read the plain coefficient file and fill the weights
         * @param rawid  An ID for the weights resource in the raw file
         */
        public void read(int rawid){
            Resources res = hardwareMap.appContext.getResources();
            try {
                DataInputStream dis = new DataInputStream(res.openRawResource(rawid));
                for(int level = 0; level < numLayers-1; level++){
                    for(int i = 0; i < weights[level].length; i++){
                        for(int j = 0; j < weights[level][i].length; j++){
                            weights[level][i][j] = dis.readFloat();
                        }
                    }
                    for(int i = 0; i < weights[level].length; i++){
                        biases[level+1][i] = dis.readFloat();
                    }

                }
                helloString += ".r";
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        /**
         * Read the CSV file containing means and standard deviations
         * @param rawid  An ID for the CSV resource in the raw file
         */
        public void setMeanAndDev(int rawid){
            Resources res = hardwareMap.appContext.getResources();
            try {
                BufferedReader reader = new BufferedReader(new InputStreamReader(
                        res.openRawResource(rawid)));
                String line = reader.readLine();
                String[] m = line.split(",");
                line = reader.readLine();
                String[] d = line.split(",");
                for(int i = 0; i < numInputs; i++){
                    means[i] = Float.valueOf(m[i]);
                    stddevs[i] = Float.valueOf(d[i]);
                }
                helloString += ".ms";
                initialized = true;
            } catch (IOException e) {
                e.printStackTrace();
            }

        }


        /**
         * Reads data from a csv file of image data and predicts the value.
         *
         * @param filename
         */
        public void verify(String filename){
            float[] ins = new float[numInputs];

            try {
                FileReader f = new FileReader(filename);
                BufferedReader reader = new BufferedReader(f);
                for(int t = 0; t < 500; t++){
                    String line = reader.readLine();
                    String[] m = line.split(",");
                    for(int i = 0; i < numInputs; i++){
                        ins[i] = Float.valueOf(m[i]);
                    }
                    //for(int i = numInputs - 10; i < numInputs; i++)
                    //	System.out.print(ins[i] + " ");
                    int prediction = predict(ins);
                    System.out.println(prediction);
                }

            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }


        }

        /**
         * Takes an array for input into the network, and returns the predicted class
         *
         * @param inputs
         * @return the index of the predicted class
         */
        public int predict(float[] inputs){
            // Initialize propagation values
            float[][] props = new float[numLayers][];
            props[0] = inputs;
            // Normalize input
            for(int i = 0; i < numInputs; i++)
                inputs[i] = (inputs[i] - means[i]) / stddevs[i];
            for(int i = 1; i < numLayers; i++)
                props[i] = new float[weights[i-1].length];

            // Propagate
            for(int layer = 1; layer < numLayers; layer++){
                for(int i = 0; i < weights[layer-1].length; i++){
                    props[layer][i] = biases[layer][i];
                    for(int j = 0; j < weights[layer-1][i].length; j++)
                        props[layer][i] += props[layer-1][j] * weights[layer-1][i][j];
                    props[layer][i] = (float) Math.tanh(props[layer][i]);
                }
            }

            // Select the greatest value on the output layer
            float max = props[numLayers - 1][0];
            int maxidx = 0;
            for(int i = 1; i < weights[numLayers - 2].length; i++){
                if(props[numLayers - 1][i] > max){
                    max = props[numLayers - 1][i];
                    maxidx = i;
                }
            }
            return maxidx;
        }


        /**
         * Returns a string indicating finished-ness statuses
         *
         * @return String where 0="started", i="initialized", r="read weights", ms = "read means and stddevs"
         */
        public String hello(){
            return helloString;
        }

        /**
         * Tells whether the model is finished with initialization
         *
         * @return True when all files have been read and processed
         */
        public boolean isInitialized(){
            return initialized;
        }
    }


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
     <<<<<<< HEAD
     *

     void processModel(){
     MultiLayerConfiguration confFromJson = null;


     Resources res = hardwareMap.appContext.getResources();
     InputStream is = res.openRawResource(R.raw.conf);

     Writer writer = new StringWriter();
     char[] buffer = new char[1024];
     try {
     Reader reader = new BufferedReader(new InputStreamReader(is, "UTF-8"));
     int n;
     while ((n = reader.read(buffer)) != -1) {
     writer.write(buffer, 0, n);
     }
     } catch (UnsupportedEncodingException e) {
     e.printStackTrace();
     } catch (IOException e) {
     e.printStackTrace();
     } finally {
     try {
     is.close();
     } catch (IOException e) {
     e.printStackTrace();
     }
     }

     String jsonString = writer.toString();
     //confFromJson = MultiLayerConfiguration.fromJson(jsonString);

     //Load parameters from disk:
     JavaNDArray newParams = null;
     float[] x = new float[1917606];
     try {
     DataInputStream dis = new DataInputStream(res.openRawResource(R.raw.coeffplain));
     for(int i = 0; i < 1917606; i++)
     x[i] = dis.readFloat();
     newParams = new JavaNDArray(x);

     } catch (FileNotFoundException e) {
     e.printStackTrace();
     } catch (IOException e) {
     e.printStackTrace();
     }

     //Create a MultiLayerNetwork from the saved configuration and parameters
     MultiLayerNetwork model = new MultiLayerNetwork(confFromJson);
     model.init();
     model.setParameters(newParams);
     }



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
     double distance = dist(fpArray[i], tpArray[i]);
     if(b[i] == 1 && distance > -1 && distance < 30) {
     // draw the segment to the screen
     Imgproc.line(cvWorkingImage, fpArray[i], tpArray[i], new Scalar(255, 0, 255));
     // Place the segment into the appropriate bin
     int binNumber = (int)(fpArray[i].x * numBins) / cvWorkingImage.cols();
     bins.get(binNumber).add((int)xdist(fpArray[i], tpArray[i]));
     }
     }

     // Now compute how to turn based on the bins
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
     }

     // Now set leftVal for turning. 80 = straight, less = left (maybe...)
     targetXval = 80;

     if(turnState == NOT_TURNING) {
     double factorThresh = 1.2;
     double minThresh = 2;
     if (globalSum[0] > minThresh && globalSum[0] > factorThresh * globalSum[1] && globalSum[0] > factorThresh * globalSum[2])
     targetXval = 140;
     if (globalSum[2] > minThresh && globalSum[2] > factorThresh * globalSum[1] && globalSum[2] > factorThresh * globalSum[0])
     targetXval = 20;
     }

     if(targetXval != 80)
     turnState = TURNING;
     else
     turnState = NOT_TURNING;

     Log.d("leftVal", "" + targetXval);

     cvColorImage = cvWorkingImage;
     cvLastImage = cvBWImage.clone();

     CVImageProcessingFlag = false;



     telemetry.addData("left", globalSum[0]);
     telemetry.addData("middle", globalSum[1]);
     telemetry.addData("right", globalSum[2]);
     telemetry.addData("longestLines", largestVal(globalSum[0], globalSum[1], globalSum[2]));





     /**
     * Mode for using one joystick
     *
     private void doGoToTheLightDrive() {
     telemetry.addData("targetXval:", "" + targetXval);

     float speed = 0.3f;
     float rightBias  = (targetXval - 80.0f)/600.0f;
     float l_left_drive_power
     = (float) scale_motor_power(speed + rightBias);
     float l_right_drive_power
     = (float) scale_motor_power(speed - rightBias);
     //telemetry.addData("s, t = ", "" + speed + ", " + turn);
     telemetry.addData("Powers: ", "" + l_left_drive_power + ":" + l_right_drive_power);
     set_drive_power(l_left_drive_power, l_right_drive_power);


     }




     private String largestVal(double l, double c, double r){
     if(l > c && l > r){
     return "left";
     }else if(r > c)
     return "right";
     else
     return "center";
     }



     */

}
