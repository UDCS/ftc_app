package com.qualcomm.ftcrobotcontroller.opmodes;

import android.graphics.Bitmap;
import android.graphics.YuvImage;
import android.hardware.Camera;
import android.os.AsyncTask;
import android.os.Environment;
import android.util.Log;

import com.qualcomm.ftcrobotcontroller.CameraPreview;
import com.qualcomm.ftcrobotcontroller.FtcRobotControllerActivity;
import com.qualcomm.robotcore.eventloop.opmode.LinearOpMode;
import com.qualcomm.robotcore.hardware.DcMotor;
import com.qualcomm.robotcore.hardware.Servo;
import com.qualcomm.robotcore.util.Range;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

public class VisionGather extends LinearOpMode{

    final int LOOKLEFT = 0;
    final int LOOKFORWARD = 1;
    final int LOOKRIGHT = 2;
    int lookState = 1;

    DcMotor Lmotor, Rmotor;
    Servo  ServoA1, ServoA2;

    long lastTime;
    long curTime;
    int cameraTimer;

    /* Camera fields */
    private Camera camera;
    public CameraPreview preview;
    public Bitmap image;
    private int width;
    private int height;
    private YuvImage yuvImage = null;
    private int looped = 0;
    private String data;

    @Override
    public void runOpMode() throws InterruptedException{
        Lmotor = hardwareMap.dcMotor.get("m1");
        Rmotor = hardwareMap.dcMotor.get("m2");
        ServoA1 = hardwareMap.servo.get("s1");
        ServoA2 = hardwareMap.servo.get("s2");

        Rmotor.setDirection(DcMotor.Direction.REVERSE);

/* Camera stuff */
        camera = ((FtcRobotControllerActivity)hardwareMap.appContext).camera;
        camera.setPreviewCallback(previewCallback);


        Camera.Parameters parameters = camera.getParameters();
        data = parameters.flatten();

        ((FtcRobotControllerActivity) hardwareMap.appContext).initPreview(camera, this, previewCallback);



        waitForStart();
        while(opModeIsActive()){
            updateTimer();
            telemetry.addData("Time:", "Updated, " + cameraTimer);
            if(cameraTimer > 10000) {
                telemetry.addData("Camera:", "Taking pic");
                CameraLoop();
                cameraTimer = 0;
            }

            doSingleJoystickDrive();
            ServoA2.setPosition(0.5);

            switch(lookState){
                case LOOKLEFT:
                    ServoA1.setPosition(0.4);
                    break;

                case LOOKFORWARD:
                    ServoA1.setPosition(0.5);
                    break;

                case LOOKRIGHT:
                    ServoA1.setPosition(0.6);
                    break;
            }

            if(gamepad1.x)
                lookState = LOOKLEFT;
            if(gamepad1.y)
                lookState = LOOKFORWARD;
            if(gamepad1.b)
                lookState = LOOKRIGHT;


            waitForNextHardwareCycle();

        }
        Lmotor.setPower(0);
        Rmotor.setPower(0);
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

    private void doNeuralNetworkDrive(int directions){

        float speed = 0f;
        float turn  = 0f;


        if(directions == STRAIGHT)
            speed = -0.25f;
        else if (directions == TURN_RIGHT) {
            speed = -0.1f;
            turn = 0.2f;
        }else if (directions == TURN_LEFT) {
            speed = -0.1f;
            turn = -0.2f;
        }
        float l_left_drive_power
                = (float) scale_motor_power(speed - turn);
        float l_right_drive_power
                = (float) scale_motor_power(speed + turn);
        telemetry.addData("s, t = ", "" + speed + ", " + turn);

        set_drive_power(l_left_drive_power, l_right_drive_power);
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

    /**
     *
     * @param left power for left wheels
     * @param right power for right wheels
     */
    void set_drive_power (double left, double right)
    {
        Lmotor.setPower(left);
        Rmotor.setPower(right);

        telemetry.addData("Motors: ", Lmotor.getPower()+"   " + Rmotor.getPower());


    }

    private void updateTimer() {
        lastTime = curTime;
        curTime = System.currentTimeMillis();
        cameraTimer += curTime - lastTime;
    }



    /* Camera Stuff */


    private Camera.PreviewCallback previewCallback = new Camera.PreviewCallback() {
        public void onPreviewFrame(byte[] data, Camera camera)
        {
//            Camera.Parameters parameters = camera.getParameters();
//            width = parameters.getPreviewSize().width;
//            height = parameters.getPreviewSize().height;
//            yuvImage = new YuvImage(data, ImageFormat.NV21, width, height, null);
//            looped += 1;
        }
    };

//private void convertImage() {
//    ByteArrayOutputStream out = new ByteArrayOutputStream();
//    yuvImage.compressToJpeg(new Rect(0, 0, width, height), 0, out);
//    byte[] imageBytes = out.toByteArray();
//    image = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
//}


    public void CameraLoop() {
//      camera.takePicture(shutterCallback, rawCallback, jpegCallback);

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
//            resetCam();
//            Log.d("VisionGather", "onPictureTaken - jpeg");
        }
    };

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

    /**
     *
     */
    void processModel(MultiLayerNetwork model){

	}

}
