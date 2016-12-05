using UnityEngine;
using System.Collections;
using WebSocketSharp;
using UnityEngine.UI;
using SimpleJSON;

public class VrTrackerUser : MonoBehaviour {
	
	private WebSocket ws;
	private Vector3 position;
	private Vector3 previousPosition;
	public string GatewayIP = "192.168.42.1"; // 85 // Set your gateway IP here
	public float scale = 1f;
	public float x_offset = 0.0f;
	public float y_offset = 0.0f;
	public float z_offset = 0.0f;
	public string WebsocketPort = "85";

	private string textString;
	private bool jsonSend;
	public Transform CameraLocation;
	private Vector3 CameraPosition;


	private bool ButtonActive;

	void OnGUI(){
		jsonSend = false;
		//CONNECT

		if (ButtonActive == true) {

			if (GUI.Button (new Rect (530, 330, 250, 150), "Connect")) { 
				// Create and Open the websocket
				ws = new WebSocket ("ws://" + GatewayIP + ":" + WebsocketPort );
				ws.OnOpen += OnOpenHandler;
				ws.OnMessage += OnMessageHandler;
				ws.OnClose += OnCloseHandler;

				ws.ConnectAsync ();	
				jsonSend = false;
				Debug.Log ("ConnectButtonClick");
			}
			//DISCONNECT
			if (GUI.Button (new Rect (10, 330, 250, 150), "Disconnect")) { //Correct
				jsonSend = false;
				ws.CloseAsync ();
				Debug.Log ("ButtonClick");
			}


			//Debug.Log ("Button Function Run");
			//JSON SENDING
			var I = new JSONClass ();



			if (jsonSend == true) {
				ws.SendAsync (I.ToString (), OnSendComplete);
				jsonSend = false;
			}

		}
	}

	void Awake(){
		ws = new WebSocket ("ws://" + GatewayIP + ":" + WebsocketPort );
		ws.OnOpen += OnOpenHandler;
		ws.OnMessage += OnMessageHandler;
		ws.OnClose += OnCloseHandler;

		ws.ConnectAsync ();	
	}

	void Start () {

	}
	
	private void OnOpenHandler(object sender, System.EventArgs e) {
		Debug.Log("Connected to Gateway!");
		ButtonActive = false;

	}
	
	private void OnMessageHandler(object sender, MessageEventArgs e) {

		//Debug.Log(e.Data);
		textString = e.Data;
		var MessageArray = JSONNode.Parse(textString);
//		Debug.Log (MessageArray["MessageType"]);
		if (MessageArray["MessageType"].Value == "Coordinates"){
			var CoordinatesArray = JSONNode.Parse(MessageArray ["Coordinates"].ToString()) ;
			Debug.Log ( CoordinatesArray );
			float Xf = (CoordinatesArray ["X"].AsFloat * 1.0f) +  x_offset; 
			float Yf  = CoordinatesArray ["Y"].AsFloat + y_offset;
			float Zf = (CoordinatesArray ["Z"].AsFloat * 1.0f) + z_offset;
		//	Debug.Log  ("X: " + Xf + " Y: " + Yf + " Z: " + Zf);
			CameraPosition =  new Vector3 (Xf, Yf, Zf);

			}

		// Here you can add some post treatment on the position (remove weird datas, add a Kalman filter, smoothen the curve...
			
		/*if (Mathf.Abs (previousPosition [0] - position [0]) > 20 || Mathf.Abs (previousPosition [1] - position [1]) > 20 || Mathf.Abs (previousPosition [2] - position [2]) > 20) {
			Debug.LogError ("Incoherent 3D position received");
		} else {
			position = previousPosition;
			Debug.Log (position);
		}*/
	}

	private void OnCloseHandler(object sender, CloseEventArgs e) {
		Debug.Log("Connection to Gateway closed for this reason: " + e.Reason);
		textString = "Connection to Gateway closed for this reason: " + e.Reason;
		ButtonActive = true;
	}

	private void OnSendComplete(bool success) {

	}
	
	// Update is called once per frame
	void Update () {

		CameraLocation.position = CameraPosition * scale;

	}

	void OnApplicationQuit() {
		Debug.Log("Application ending after " + Time.time + " seconds");
		ws.Close();
	}

	void SendConfigCoordinates(string Coords){
		ws.SendAsync (Coords, OnSendComplete);
	}
	
}
