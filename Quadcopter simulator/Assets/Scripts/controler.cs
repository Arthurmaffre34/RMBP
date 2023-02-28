using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.IO;


public class controler : MonoBehaviour
{
    
    private TcpListener listener = null;
    private TcpClient client = null;
    private NetworkStream ns = null;
    string msg;

    public int connection = 0;
    
    // Start is called before the first frame update
    void Awake()
    {
        listener = new TcpListener(Dns.GetHostEntry("localhost").AddressList[1], 5697);
        listener.Start();
        Debug.Log("is listening");

        
    }

    // Update is called once per frame
    void Update()
    {
        if (listener.Pending())
        {
            client = listener.AcceptTcpClient();
            Debug.Log("Connected");
            connection = 1;
        }
        if (connection == 0) {
            return;
        }
        
        ns = client.GetStream();

        
        

        
        Debug.Log("msg");

        if ((ns != null) && (ns.DataAvailable))
        {
            byte[] buffer = new byte[1024];
            int bytesRead = ns.Read(buffer, 0, buffer.Length);
            string message = System.Text.Encoding.ASCII.GetString(buffer, 0, bytesRead);
            Debug.Log(message);
        }
    }
}
