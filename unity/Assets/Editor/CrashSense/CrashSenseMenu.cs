// Editor menu for CrashSense in Unity: a self test that runs both engines on fixed crashes and writes
// the results to <project>/Captures/crashsense_selftest.txt (also printed in the Console).
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using UnityEditor;
using UnityEngine;
using CrashSense;

public static class CrashSenseMenu
{
    [MenuItem("Tools/CrashSense/Self Test", false, 0)]
    public static void SelfTest()
    {
        var sb = new StringBuilder();
        var inv = CultureInfo.InvariantCulture;
        if (!CrashSenseCar.EnsureLoaded()) { UnityEngine.Debug.LogError("CrashSense self test: data not loaded"); return; }
        var ph = CrashSenseCar.PhysicsEngine;
        var model = CrashSenseCar.Model;
        sb.AppendLine("CrashSense self test in Unity " + Application.unityVersion + ", " + DateTime.Now.ToString("s", inv));
        var cases = new[]
        {
            ("front, concrete wall, 56 km/h", "front", "concrete_wall", 56.0, 0.0, 0.0),
            ("front, steel pole, 50 km/h, offset 0.5", "front", "steel_pole", 50.0, 0.5, 0.0),
            ("left side, steel pole, 29 km/h", "left", "steel_pole", 29.0, 0.0, 0.0),
            ("rear, concrete block, 30 km/h", "rear", "concrete_block", 30.0, 0.0, 0.0),
            ("front, foam barrier, 65 km/h, 15 deg", "front", "foam_barrier", 65.0, 0.0, 15.0),
        };
        foreach (var (name, face, obstacle, kmh, offset, angle) in cases)
        {
            var req = new Dictionary<string, object>
            {
                { "speed_kmh", kmh }, { "face", face }, { "offset", offset }, { "angle_deg", angle },
                { "obstacle", new Dictionary<string, object> { { "name", obstacle } } },
            };
            var sc = ph.ScenarioFromDict(req);
            model.Predict(ph, sc); ph.Simulate(sc);                       // warm-up
            var sw = Stopwatch.StartNew();
            Result m = null;
            for (int k = 0; k < 20; k++) m = model.Predict(ph, sc);
            double msModel = sw.Elapsed.TotalMilliseconds / 20;
            sw.Restart();
            Result p = null;
            for (int k = 0; k < 20; k++) p = ph.Simulate(sc);
            double msPhys = sw.Elapsed.TotalMilliseconds / 20;
            double mae = Enumerable.Range(0, m.Damage.Length).Average(i => Math.Abs(m.Damage[i] - p.Damage[i])) * 100;
            var worst = Enumerable.Range(0, p.Damage.Length).OrderByDescending(i => p.Damage[i]).Take(3)
                .Select(i => ph.T.PartNames[i] + " " + p.Damage[i].ToString("0.00", inv) + "/" + m.Damage[i].ToString("0.00", inv));
            sb.AppendLine(string.Format(inv, "{0}: crush physics {1:0.000} m / model {2:0.000} m; damage gap {3:0.00} pt; {4:0.00} ms model, {5:0.00} ms physics; worst (physics/model): {6}",
                name, p.CrushDepth, m.CrushDepth, mae, msModel, msPhys, string.Join(", ", worst)));
        }
        string folder = Path.GetFullPath(Path.Combine(Application.dataPath, "../Captures"));
        Directory.CreateDirectory(folder);
        File.WriteAllText(Path.Combine(folder, "crashsense_selftest.txt"), sb.ToString());
        UnityEngine.Debug.Log(sb.ToString());
    }
}
