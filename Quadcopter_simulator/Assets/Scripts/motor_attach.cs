using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class motor_attach : MonoBehaviour
{
    //motors
    public GameObject motor_FL;
    public GameObject motor_FR;
    public GameObject motor_BL;
    public GameObject motor_BR;

    Vector3 pos_motor_FL = new Vector3(0.5f,0f,0.4f);
    Vector3 pos_motor_FR = new Vector3(0.5f,0f,-0.4f);
    Vector3 pos_motor_BL = new Vector3(-0.5f,0f,0.4f);
    Vector3 pos_motor_BR = new Vector3(-0.5f,0f,-0.4f);


    // Start is called before the first frame update
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {
        motor_FL.transform.position = Quadcopter_physic.angle_quad * pos_motor_FL + Quadcopter_physic.movement;
        motor_FR.transform.position = Quadcopter_physic.angle_quad * pos_motor_FR + Quadcopter_physic.movement;
        motor_BL.transform.position = Quadcopter_physic.angle_quad * pos_motor_BL + Quadcopter_physic.movement;
        motor_BR.transform.position = Quadcopter_physic.angle_quad * pos_motor_BR + Quadcopter_physic.movement;
    }
}
