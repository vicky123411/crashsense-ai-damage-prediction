// Stand-ins for Ramp Run classes, only for compiling the CrashSense Unity scripts outside Unity.
using UnityEngine;
public class CarDamage : MonoBehaviour { }
public static class TrackLayout
{
    public static bool OnRoad(float x, float z) { return true; }
    public static float Height(float x, float z) { return 0f; }
}
