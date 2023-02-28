using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Quadcopter_physic : MonoBehaviour
{

    //gravity
    public float mass = 1;
    Vector3 gravity = new Vector3(0f,-9.81f,0f);

    //thrust in N
    public float max_thrust = 20f;
    public float thrust_FL_pc = 0.5f;
    public float thrust_FR_pc = 0.5f;
    public float thrust_BL_pc = 0.5f;
    public float thrust_BR_pc = 0.5f;
    float thrust_FL;
    float thrust_FR;
    float thrust_BL;
    float thrust_BR;
    float thrust_add;
    
    Vector3 thrust;

    public static Quaternion angle_quad;

    //movement
    public static Vector3 movement_accel;
    public static Vector3 movement_vit;
    public static Vector3 movement = new Vector3(0f, 1f, 0f);

    //rotation
    Quaternion rotation;
    Vector3 rotation_axis_accel;
    Vector3 rotation_axis_vit;
    Vector3 rotation_axis;
    float radius = 1f;


    // Start is called before the first frame update
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {
        //set angle of quad using quaternion
        angle_quad = Quaternion.Euler(transform.localEulerAngles);

        
        
        //update thrust
        thrust_FL = thrust_FL_pc * (max_thrust/4);
        thrust_FR = thrust_FR_pc * (max_thrust/4);
        thrust_BL = thrust_BL_pc * (max_thrust/4);
        thrust_BR = thrust_BR_pc * (max_thrust/4);

        thrust_add = thrust_BL + thrust_BR + thrust_FL + thrust_FR;
        thrust = new Vector3(0f, thrust_add ,0f); 

        //thrust to absolute base using quaternion
        
        thrust = angle_quad * thrust;
        

        //update movement_accel
        movement_accel = gravity + thrust;
        
        
        //convert accel 3dvector into position 3dvector
        movement_vit = movement_vit + movement_accel * Time.deltaTime;
        movement = movement + movement_vit * Time.deltaTime;

        //apply movement vector
        
        transform.position = movement;

        //rotation x
        rotation_axis_accel.x = thrust_BR + thrust_FR - thrust_BL - thrust_FL;
        rotation_axis_accel.z = thrust_FL + thrust_FR - thrust_BL - thrust_BR;
        //technical thing
        rotation_axis_accel.y = thrust_FR + thrust_BL - thrust_FL - thrust_BR;
        
        rotation_axis_accel = (rotation_axis_accel*radius)/mass;
        rotation_axis_vit = rotation_axis_vit + rotation_axis_accel * Time.deltaTime;
        rotation_axis = rotation_axis + rotation_axis_vit * Time.deltaTime;
        
        rotation = Quaternion.Euler(rotation_axis * Mathf.Rad2Deg);
        
        //Debug.Log(rotation);
        transform.rotation  = rotation;
    }
}