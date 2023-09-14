using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

public class GodsView : MonoBehaviour
{   
    void CamCapture()
    {
        Camera Cam = this.GetComponent<Camera>();

        RenderTexture currentRT = RenderTexture.active;
        RenderTexture.active = Cam.targetTexture;
 
        Cam.Render();

        Texture2D Image = new Texture2D(Cam.targetTexture.width, Cam.targetTexture.height);
        Image.ReadPixels(new Rect(0, 0, Cam.targetTexture.width, Cam.targetTexture.height), 0, 0);
        Image.Apply();
        RenderTexture.active = currentRT;
 
        var Bytes = Image.EncodeToPNG();
        Destroy(Image);
        File.WriteAllBytes($"/Users/ken/Desktop/GodsView.png", Bytes);
    }

    // Start is called before the first frame update
    void Start()
    {


    }

    // Update is called once per frame
    void Update()
    {
        
    }

    //FixedUpdate
    void FixedUpdate()
    {   
        CamCapture();
        UnityEditor.EditorApplication.isPlaying = false;
    }
}   
