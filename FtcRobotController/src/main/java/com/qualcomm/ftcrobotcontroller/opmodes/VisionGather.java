package com.qualcomm.ftcrobotcontroller.opmodes;

import com.qualcomm.robotcore.eventloop.opmode.LinearOpMode;
import com.qualcomm.robotcore.hardware.DcMotor;
import com.qualcomm.robotcore.hardware.Servo;
import com.qualcomm.robotcore.util.Range;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

public class VisionGather extends LinearOpMode{

    final int LOOKLEFT = 0;
    final int LOOKFORWARD = 1;
    final int LOOKRIGHT = 2;
    int lookState = 1;

    DcMotor Lmotor, Rmotor;
    Servo  ServoA1, ServoA2;

    @Override
    public void runOpMode() throws InterruptedException{
        Lmotor = hardwareMap.dcMotor.get("m1");
        Rmotor = hardwareMap.dcMotor.get("m2");
        ServoA1 = hardwareMap.servo.get("s1");
        ServoA2 = hardwareMap.servo.get("s2");

        Rmotor.setDirection(DcMotor.Direction.REVERSE);



        waitForStart();
        while(opModeIsActive()){
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

    /**
     *
     */
    void processModel(MultiLayerNetwork model){

    }
}