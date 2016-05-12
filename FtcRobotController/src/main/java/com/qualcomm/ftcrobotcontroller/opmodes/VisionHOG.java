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

public class VisionHOG extends VisionGather {
    private static ArrayList<Mat> kernels;


    @Override
    public void runOpMode() throws InterruptedException {
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
        model = new MyModel(new int[]{576, 600, 3000, 6});
        model.read(R.raw.coeffplainhog);
        model.setMeanAndDev(R.raw.meanvarhog);
        predictionQueue = new LinkedList<Integer>();

        waitForStart();
        while (opModeIsActive()) {

            doNeuralNetDrive();

            ServoA2.setPosition(0.6);
            ServoA1.setPosition(0.55);

            if (gamepad1.x)
//                lookState = LOOKLEFT;
                curField = Field.MEMORY;
            if (gamepad1.y)
//                lookState = LOOKFORWARD;
                curField = Field.RATIO;
            if (gamepad1.b)
                curField = Field.PREDICTADJUST;
//                lookState = LOOKRIGHT;

            int inc = 0;
            if (!pressed) {
                if (gamepad1.dpad_up) {
                    inc = 1;
                    pressed = true;
                } else if (gamepad1.dpad_down) {
                    inc = -1;
                    pressed = true;
                }
            } else if (!gamepad1.dpad_up && !gamepad1.dpad_down)
                pressed = false;

            switch (curField) {
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
        float rightBias = (predictionHistory - predictAdjust) / ratio;
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
        public void onPreviewFrame(byte[] data, Camera camera) {
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

            if (cvLayout != null && cvColorImage != null) {
                Utils.matToBitmap(cvColorImage, cvBitmap);
                cvLayout.setImageBitmap(cvBitmap);
            }
            telemetry.addData("model", model == null ? "null" : model.hello());
        }
    };


    /**
     * Method to set up the CV data structures that we will be using
     * <p/>
     * OpenCV stuff can't be done on the UI (main) thread. Must be done on a background thread.
     */
    public class OpenCVSetup extends AsyncTask<Void, Void, Void> {
        @Override
        protected Void doInBackground(Void... params) {
            cvOrigImage = new Mat(160, 240, CvType.CV_8UC4);
            cvBWImage = new Mat(160, 240, CvType.CV_8UC1);
            cvColorImage = new Mat(240, 160, CvType.CV_8UC4);
            cvLastImage = new Mat(240, 160, CvType.CV_8UC1);
            cvWorkingImage = new Mat(240, 160, CvType.CV_8UC4);

            // Build a kernel
            int N = 3;
            kernels = new ArrayList<Mat>();
            int angleStep = 30;
            for(double a = 0; a < 180; a += angleStep){
                Mat kernel = new Mat(new Size(N, N), CvType.CV_64F);
                double angle = a * Math.PI/180.0;
                for(int i = 0; i < N; i++){
                    for(int j = 0; j < N; j++){
                        //double entry = values[i*N + j];
                        double entry = Math.sin(Math.PI/2.0*((j-(N-1.0)/2)*Math.cos(angle)
                                + (i - (N-1.0)/2)*Math.sin(angle)));
                        kernel.put(i, j, entry);
                        //System.out.printf("%.3f ", entry);
                    }
                    System.out.println("");
                }
                System.out.println("");
                kernels.add(kernel);
            }

            return null;
        }
    }


    /**
     * Method for processing the image using OpenCV methods
     * <p/>
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

            float[] featureArray = new float[576];
            for(Mat kernel : kernels){
                Mat im = new Mat();
                Imgproc.filter2D(cvBWImage, im, -1, kernel);
                int windowSize = 20;
                // Get feature vector
                int idx = 0;
                for(int j = 0; j < im.cols(); j += windowSize){
                    for(int i = 0; i < im.rows(); i += windowSize){
                        Scalar sum = Core.sumElems(im.submat(i, i + windowSize - 1, j, j + windowSize - 1));
                        featureArray[idx] = (float)(sum.val[0]);
                        idx++;
                    }
                }
            }


            Imgproc.cvtColor(cvBWImage, cvColorImage, Imgproc.COLOR_GRAY2BGRA);
            int prediction = model.predict(featureArray);

            Imgproc.putText(cvColorImage, String.valueOf(prediction),
                    new Point(20, 120),
                    Core.FONT_HERSHEY_TRIPLEX, 2.0, new Scalar(0, 255, 0));

            String[] predString = {"Stop", "Hard Left", "Left", "Forward", "Right", "Hard Right"};
            telemetry.addData("prediction:", predString[prediction]);
            if (prediction != 0)
                predictionHistory = MEMORY * predictionHistory + (1 - MEMORY) * prediction;
            /*
            // Manage the prediction queue
            if (predictionQueue != null) {
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
            */


            return null;
        }
    }

}
