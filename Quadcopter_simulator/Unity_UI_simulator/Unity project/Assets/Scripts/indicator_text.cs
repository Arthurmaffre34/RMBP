using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;


public class indicator_text : MonoBehaviour
{
    //create vector position, speed, accel
    Vector3 position;
    Vector3 speed;
    Vector3 accel;

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        //refresh values (position, speed, accel) from Quadcopter_physics
        gameObject.GetComponent<TMP_Text>().text = $"Accel: {Quadcopter_physic.movement_accel}\nSpeed: {Quadcopter_physic.movement_vit}\nPos: {Quadcopter_physic.movement}";
    }
}
