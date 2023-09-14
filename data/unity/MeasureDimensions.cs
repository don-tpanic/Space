using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MeasureDimensions : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {   
        //First we log the name of this object whose
        //measuresments we try to print
        Debug.Log("Object: " + gameObject.name);
        //Log the width, height and depth of the object in real size
        //Because this is meant to be a generic script, 
        //Width height and depth are not necessarily the same as the x, y and z axes
        //To be consistent we should use the x, y and z axes
        Debug.Log("Width: " + gameObject.GetComponent<Renderer>().bounds.size.x);
        Debug.Log("Height: " + gameObject.GetComponent<Renderer>().bounds.size.y);
        Debug.Log("Depth: " + gameObject.GetComponent<Renderer>().bounds.size.z);

        //If the object name is "Plane", we also measure its distance to the 4 walls
        //which are objects "Wall 2", "Wall 4", "FrontWallClean", "FrontWallClean (1)"
        //We assume that the plane is always placed on the floor
        if (gameObject.name == "Plane")
        {   
            Debug.Log("Computing Plance distance to outer walls...");
            //We get the position of the plane
            Vector3 planePosition = gameObject.transform.position;
            //We get the position of the walls
            Vector3 wall2Position = GameObject.Find("Wall 2").transform.position;
            Vector3 wall4Position = GameObject.Find("Wall 4").transform.position;
            Vector3 frontWallCleanPosition = GameObject.Find("FrontWallClean").transform.position;
            Vector3 frontWallClean1Position = GameObject.Find("FrontWallClean (1)").transform.position;
            //We calculate the distance between the plane and the walls
            float distanceToWall2 = Mathf.Abs(planePosition.x - wall2Position.x);
            float distanceToWall4 = Mathf.Abs(planePosition.z - wall4Position.z);
            float distanceToFrontWallClean = Mathf.Abs(planePosition.z - frontWallCleanPosition.z);
            float distanceToFrontWallClean1 = Mathf.Abs(planePosition.x - frontWallClean1Position.x);
            //We log the distances
            Debug.Log("Distance to Wall 2: " + distanceToWall2);
            Debug.Log("Distance to Wall 4: " + distanceToWall4);
            Debug.Log("Distance to FrontWallClean: " + distanceToFrontWallClean);
            Debug.Log("Distance to FrontWallClean (1): " + distanceToFrontWallClean1);
        }
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
