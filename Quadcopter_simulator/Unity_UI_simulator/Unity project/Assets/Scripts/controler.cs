using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.IO;
using System.Text;


public class Networking
{
    private string IP;
    private int PORT;

    private TcpListener listener = null;
    private TcpClient client = null;
    private NetworkStream ns = null;
    string msg;

    public int connection = 0;
    
    public Networking(string IP, int PORT)
    {
        this.IP = IP;
        this.PORT = PORT;
    }

    public void Start_server()
    {
        listener = new TcpListener(Dns.GetHostEntry(IP).AddressList[1], PORT);
        listener.Start();
        Debug.Log("is listening");
    }

    public void Stop_server()
    {
        return;
    }

    public void Accept_Client()
    {
        if (listener.Pending())
        {
            client = listener.AcceptTcpClient();
            Debug.Log("Connected");
            connection = 1;
            ns = client.GetStream();
        }

        if (connection == 0) {
            return;
        }

    }

    public (string, string) ReceiveData()
    {
        if (client == null)
        {
            return(null, null);
        }
        
        

        if ((ns != null) && (ns.DataAvailable))
        {
            byte[] buffer = new byte[1024];
            int bytesRead = ns.Read(buffer, 0, buffer.Length);

            string message_type = System.Text.Encoding.ASCII.GetString(buffer, 0, 1);
            string message = System.Text.Encoding.ASCII.GetString(buffer, 4, bytesRead -4);
            return (message_type, message);
            
        }
        else
        {
            return(null, null);
        }
    }

    public void SendData(Quadcopter_physic quadcopterPhysic)
    {
        if (ns != null)
        {
            //data format pos_x,pos_y,pos_z,vit_x,vit_y,vit_z,accel_x,accel_y,accel_x,rot_x,rot_y,rot_z,rotvit_x,rotvit_y,rotvit_z,rotaccel_x,rotaccel_y,rotaccel_z
            string mess = string.Join(",", Quadcopter_physic.movement.x.ToString(), Quadcopter_physic.movement.y.ToString(), Quadcopter_physic.movement.z.ToString(),
            Quadcopter_physic.movement_vit.x.ToString(), Quadcopter_physic.movement_vit.y.ToString(), Quadcopter_physic.movement_vit.z.ToString(),
            Quadcopter_physic.movement_accel.x.ToString(), Quadcopter_physic.movement_accel.y.ToString(), Quadcopter_physic.movement_accel.z.ToString(),
            Quadcopter_physic.rotation_axis.x.ToString(), Quadcopter_physic.rotation_axis.y.ToString(), Quadcopter_physic.rotation_axis.z.ToString(),
            Quadcopter_physic.rotation_axis_vit.x.ToString(), Quadcopter_physic.rotation_axis_vit.y.ToString(), Quadcopter_physic.rotation_axis_vit.z.ToString(),
            Quadcopter_physic.rotation_axis_accel.x.ToString(), Quadcopter_physic.rotation_axis_accel.y.ToString(), Quadcopter_physic.rotation_axis_accel.z.ToString());
            

            string message = string.Format("{0}{1:000}{2}", "3", mess.Length, mess);
            byte[] data = Encoding.ASCII.GetBytes(message);
            ns.Write(data, 0, data.Length);
            ns.Flush();
        }

    }

}


public class controler : MonoBehaviour
{
    Networking server = null;
    string message_type;
    string message;
    private Quadcopter_physic quadcopterPhysic;

    // Start is called before the first frame update
    void Awake()
    {
        quadcopterPhysic = FindObjectOfType<Quadcopter_physic>();
        server = new Networking("localhost", 5697);

        server.Start_server();
        
    }

    // Update is called once per frame
    void Update()
    {
        server.Accept_Client();


        (message_type, message) = server.ReceiveData();
        
        //data format pos_x,pos_y,pos_z,vit_x,vit_y,vit_z,accel_x,accel_y,accel_x,rot_x,rot_y,rot_z,rotvit_x,rotvit_y,rotvit_z,rotaccel_x,rotaccel_y,rotaccel_z
        server.SendData(quadcopterPhysic);

        

        if (message_type == "2")
        {
            //reset();
            Debug.Log(message_type);
            Debug.Log(message);
            quadcopterPhysic.Reset();
        } 
        if (message_type == "1")
        {
            //data();
            //Debug.Log(message_type);
            //Debug.Log(message);

            string[] valeurs_separees = message.Split(',');
            
            float[] parametres = new float[4];
            for (int i = 0; i < valeurs_separees.Length; i++)
                {
                    parametres[i] = float.Parse(valeurs_separees[i]);
                    Debug.Log(parametres[i]);
                }
            quadcopterPhysic.Command(parametres[0], parametres[1], parametres[2], parametres[3]);
        }

    }
}
