// Runs CrashSense inside Unity on every real crash of the car, with no Python server:
// the crash (speed, side, offset, angle, what was hit) is turned into a CrashSense scenario,
// the trained graph network and the physics layer both predict the damage report, and the
// result is shown on screen and saved to <project>/Captures/crashsense_reports.jsonl.
//
// It only reads the crash; it does not change how the car breaks (CarDamage still does that).
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using UnityEngine;
using CrashSense;
using Debug = UnityEngine.Debug;

[DisallowMultipleComponent]
public class CrashSenseCar : MonoBehaviour
{
    public enum EngineKind { Model, Physics }

    [Tooltip("Which CrashSense engine drives the on-screen report. Both are always run and saved.")]
    public EngineKind shown = EngineKind.Model;
    [Tooltip("Closing speed (m/s) below which a contact is not treated as a crash.")]
    public float minCrashSpeed = 3f;
    [Tooltip("Seconds after a crash before another one is reported.")]
    public float cooldown = 0.6f;
    public bool showReport = true;

    static Tables tables;
    static CrashPhysics physics;
    static GraphModel model;
    static string loadError;

    Rigidbody rb;
    Bounds localBounds;
    float lastReport = -100f;

    public Result LastModel { get; private set; }
    public Result LastPhysics { get; private set; }
    public Scenario LastScenario { get; private set; }
    public double LastModelMs { get; private set; }
    public double LastPhysicsMs { get; private set; }
    public string LastObstacleLabel { get; private set; }
    public int Reports { get; private set; }

    /// <summary>Loads static.json, model.json and model.bin from StreamingAssets/CrashSense (once).</summary>
    public static bool EnsureLoaded()
    {
        if (physics != null) return true;
        if (loadError != null) return false;
        try
        {
            string dir = Path.Combine(Application.streamingAssetsPath, "CrashSense");
            tables = Tables.FromJson(File.ReadAllText(Path.Combine(dir, "static.json")));
            physics = new CrashPhysics(tables);
            model = GraphModel.Load(File.ReadAllText(Path.Combine(dir, "model.json")), File.ReadAllBytes(Path.Combine(dir, "model.bin")));
            return true;
        }
        catch (Exception e)
        {
            loadError = e.Message;
            Debug.LogError("CrashSense: could not load its data from StreamingAssets/CrashSense: " + e.Message);
            return false;
        }
    }

    public static CrashPhysics PhysicsEngine { get { EnsureLoaded(); return physics; } }
    public static GraphModel Model { get { EnsureLoaded(); return model; } }

    void Awake()
    {
        rb = GetComponent<Rigidbody>();
        EnsureLoaded();
    }

    void Start()
    {
        // size of the car in its own frame, from its renderers (used to tell which side was hit)
        var rends = GetComponentsInChildren<Renderer>();
        if (rends.Length > 0)
        {
            var b = new Bounds(transform.InverseTransformPoint(rends[0].bounds.center), Vector3.zero);
            foreach (var r in rends)
            {
                var wb = r.bounds;
                for (int i = 0; i < 8; i++)
                {
                    var c = wb.center + Vector3.Scale(wb.extents, new Vector3((i & 1) == 0 ? -1 : 1, (i & 2) == 0 ? -1 : 1, (i & 4) == 0 ? -1 : 1));
                    b.Encapsulate(transform.InverseTransformPoint(c));
                }
            }
            localBounds = b;
        }
        else localBounds = new Bounds(Vector3.zero, new Vector3(1.8f, 1.45f, 4.2f));
    }

    void OnCollisionEnter(Collision collision)
    {
        if (physics == null || Time.time < lastReport + cooldown) return;
        Collider other = collision.collider;
        if (other == null || other.isTrigger || other.attachedRigidbody == rb) return;
        int count = collision.contactCount;
        if (count == 0) return;
        Vector3 point = Vector3.zero, normal = Vector3.zero;
        for (int i = 0; i < count; i++) { var c = collision.GetContact(i); point += c.point; normal += c.normal; }
        point /= count;
        normal = normal.normalized;
        if (Vector3.Dot(normal, transform.up) > 0.65f) return;      // landing on the wheels, not a crash

        SurfaceKind surface = ImpactSurface.Resolve(other, point);
        if (surface == SurfaceKind.Ground || surface == SurfaceKind.Asphalt || surface == SurfaceKind.Grass || surface == SurfaceKind.Dirt) return;
        Vector3 rel = collision.relativeVelocity;
        float closing = Mathf.Abs(Vector3.Dot(rel, normal));
        if (closing < minCrashSpeed) return;

        string label;
        var request = BuildRequest(collision, other, point, rel, surface, out label);
        if (request == null) return;
        lastReport = Time.time;
        Run(request, label);
    }

    /// <summary>The CrashSense request for a Unity collision (the same JSON layout the Python server takes).</summary>
    Dictionary<string, object> BuildRequest(Collision collision, Collider other, Vector3 point, Vector3 rel, SurfaceKind surface, out string label)
    {
        // side of the car: compare the contact point with the car's half-length and half-width
        Vector3 lp = transform.InverseTransformPoint(point) - localBounds.center;
        float hl = Mathf.Max(localBounds.extents.z, 0.5f), hw = Mathf.Max(localBounds.extents.x, 0.5f);
        string face;
        float offset;
        Vector3 faceAxis;
        if (Mathf.Abs(lp.z) / hl >= Mathf.Abs(lp.x) / hw)
        {
            face = lp.z >= 0 ? "front" : "rear";
            offset = Mathf.Clamp(-lp.x / hw, -1f, 1f);           // CrashSense: y points to the car's left
            faceAxis = Vector3.forward;
        }
        else
        {
            face = lp.x >= 0 ? "right" : "left";
            offset = Mathf.Clamp(lp.z / hl, -1f, 1f);            // along the side, x points forward
            faceAxis = Vector3.right;
        }
        Vector3 vLocal = transform.InverseTransformDirection(rel);
        float speed = rel.magnitude;
        float angle = speed > 0.01f ? Mathf.Acos(Mathf.Clamp01(Mathf.Abs(Vector3.Dot(vLocal.normalized, faceAxis)))) * Mathf.Rad2Deg : 0f;
        angle = Mathf.Min(angle, 60f);

        // what was hit: a preset from the surface tag and the shape of the collider
        Bounds ob = other.bounds;
        float across = Mathf.Max(ob.size.x, ob.size.z), thin = Mathf.Min(ob.size.x, ob.size.z), tall = ob.size.y;
        Rigidbody orb = other.attachedRigidbody;
        bool movable = orb != null && !orb.isKinematic;
        string preset;
        switch (surface)
        {
            case SurfaceKind.Metal: preset = (across < 0.6f && tall > 1.5f) ? "steel_pole" : "steel_crate"; break;
            case SurfaceKind.Plastic: preset = "foam_barrier"; break;
            default: preset = across >= 1.5f ? "concrete_wall" : "concrete_block"; break;   // Concrete, Rock
        }
        var obstacle = new Dictionary<string, object> { { "name", preset } };
        obstacle["width"] = (double)Mathf.Clamp(across, 0.1f, 6f);
        obstacle["height"] = (double)Mathf.Clamp(tall, 0.2f, 6f);
        obstacle["z_base"] = (double)Mathf.Max(0f, ob.min.y - (transform.position.y + localBounds.min.y));
        if (movable) obstacle["mass_kg"] = (double)orb.mass;
        else obstacle["fixed"] = true;

        label = preset.Replace('_', ' ') + (movable ? string.Format(CultureInfo.InvariantCulture, " ({0:0} kg, movable)", orb.mass) : " (fixed)");
        return new Dictionary<string, object>
        {
            { "speed_ms", (double)speed }, { "face", face }, { "offset", (double)offset }, { "angle_deg", (double)angle },
            { "vehicle", new Dictionary<string, object> { { "mass_kg", (double)(rb != null ? rb.mass : 1100f) } } },
            { "obstacle", obstacle },
        };
    }

    /// <summary>Runs both engines on a request and keeps, shows and saves the result.</summary>
    public void Run(Dictionary<string, object> request, string obstacleLabel)
    {
        if (!EnsureLoaded()) return;
        var sc = physics.ScenarioFromDict(request);
        var sw = Stopwatch.StartNew();
        var m = model.Predict(physics, sc);
        sw.Stop();
        LastModelMs = sw.Elapsed.TotalMilliseconds;
        sw.Restart();
        var p = physics.Simulate(sc);
        sw.Stop();
        LastPhysicsMs = sw.Elapsed.TotalMilliseconds;
        LastScenario = sc; LastModel = m; LastPhysics = p; LastObstacleLabel = obstacleLabel;
        Reports++;
        Save(sc, m, p, obstacleLabel);
        Debug.Log(string.Format(CultureInfo.InvariantCulture,
            "CrashSense: {0:0} km/h, {1} hit, {2}: model crush {3:0.00} m ({4:0.00} ms), physics crush {5:0.00} m ({6:0.00} ms), worst part {7}",
            sc.Speed * 3.6, sc.Face, obstacleLabel, m.CrushDepth, LastModelMs, p.CrushDepth, LastPhysicsMs, WorstParts(m, 1)));
    }

    string WorstParts(Result r, int k)
    {
        return string.Join(", ", Enumerable.Range(0, r.Damage.Length).OrderByDescending(i => r.Damage[i]).Take(k)
            .Select(i => tables.PartNames[i] + " " + tables.LevelNames[r.Level[i]]).ToArray());
    }

    static string Json(Scenario sc, Result r, double ms, string label)
    {
        var inv = CultureInfo.InvariantCulture;
        var sb = new StringBuilder();
        sb.Append("{\"engine\":\"").Append(r.Engine).Append("\",\"ms\":").Append(ms.ToString("0.000", inv));
        sb.Append(",\"impact\":{\"face\":\"").Append(sc.Face).Append("\",\"speed_kmh\":").Append((sc.Speed * 3.6).ToString("0.0", inv));
        sb.Append(",\"offset\":").Append(sc.Offset.ToString("0.000", inv)).Append(",\"angle_deg\":").Append(sc.AngleDeg.ToString("0.0", inv));
        sb.Append(",\"car_mass_kg\":").Append(sc.Vehicle.MassKg.ToString("0", inv)).Append(",\"obstacle\":\"").Append(label).Append("\"}");
        sb.Append(",\"summary\":{\"crush_depth_m\":").Append(r.CrushDepth.ToString("0.0000", inv));
        sb.Append(",\"cabin_intrusion_m\":").Append(r.Intrusion.ToString("0.0000", inv));
        sb.Append(",\"peak_force_kn\":").Append(r.PeakForceKn.ToString("0.00", inv));
        sb.Append(",\"energy_total_kj\":").Append(r.EDissKj.ToString("0.00", inv));
        sb.Append(",\"obstacle_damage\":").Append(r.ObstacleDamage.ToString("0.0000", inv));
        sb.Append(",\"airbag_front\":").Append(r.AirbagFront ? "true" : "false").Append(",\"airbag_side\":").Append(r.AirbagSide ? "true" : "false");
        sb.Append(",\"went_through_car\":").Append(r.Through ? "true" : "false").Append("},\"parts\":[");
        for (int i = 0; i < r.Damage.Length; i++)
        {
            if (i > 0) sb.Append(',');
            sb.Append("{\"id\":\"").Append(tables.PartNames[i]).Append("\",\"damage\":").Append(r.Damage[i].ToString("0.0000", inv));
            sb.Append(",\"level\":").Append(r.Level[i]).Append(",\"detached\":").Append(r.Detached[i] ? "true" : "false").Append('}');
        }
        sb.Append("],\"effects\":{");
        sb.Append(string.Join(",", r.Effects.Select(kv => "\"" + kv.Key + "\":" + (kv.Value ? "true" : "false")).ToArray()));
        sb.Append("}}");
        return sb.ToString();
    }

    void Save(Scenario sc, Result m, Result p, string label)
    {
        try
        {
            string folder = Path.GetFullPath(Path.Combine(Application.dataPath, "../Captures"));
            Directory.CreateDirectory(folder);
            string line = "{\"time\":\"" + DateTime.Now.ToString("s", CultureInfo.InvariantCulture) + "\",\"model\":" + Json(sc, m, LastModelMs, label)
                + ",\"physics\":" + Json(sc, p, LastPhysicsMs, label) + "}\n";
            File.AppendAllText(Path.Combine(folder, "crashsense_reports.jsonl"), line);
            File.AppendAllText(Path.Combine(folder, "log.txt"), string.Format(CultureInfo.InvariantCulture,
                "CRASHSENSE: {0:0} km/h {1} into {2}; model crush {3:0.00} m in {4:0.00} ms; physics crush {5:0.00} m in {6:0.00} ms; worst parts {7}\n",
                sc.Speed * 3.6, sc.Face, label, m.CrushDepth, LastModelMs, p.CrushDepth, LastPhysicsMs, WorstParts(m, 4)));
        }
        catch (Exception e) { Debug.LogWarning("CrashSense: could not save the report: " + e.Message); }
    }

    // ------------------------------------------------------------------ on-screen report

    static readonly Color[] LevelColors =
    {
        new Color(0.55f, 0.62f, 0.70f), new Color(0.90f, 0.72f, 0.20f), new Color(0.93f, 0.55f, 0.20f),
        new Color(0.88f, 0.30f, 0.17f), new Color(0.60f, 0.08f, 0.08f)
    };

    GUIStyle text, title;

    void OnGUI()
    {
        if (!showReport) return;
        if (text == null)
        {
            text = new GUIStyle(GUI.skin.label) { fontSize = 13, richText = true };
            text.normal.textColor = Color.white;
            title = new GUIStyle(text) { fontSize = 15, fontStyle = FontStyle.Bold };
        }
        float w = 340f, x = Screen.width - w - 12f, y = 12f;
        if (physics == null)
        {
            DrawBox(new Rect(x, y, w, 46f));
            GUI.Label(new Rect(x + 10, y + 6, w - 20, 40), "CrashSense: data not loaded\n" + loadError, text);
            return;
        }
        var r = shown == EngineKind.Model ? LastModel : LastPhysics;
        if (r == null)
        {
            DrawBox(new Rect(x, y, w, 46f));
            GUI.Label(new Rect(x + 10, y + 6, w - 20, 40), "<b>CrashSense AI</b> ready\nwaiting for a crash...", text);
            return;
        }
        var other = shown == EngineKind.Model ? LastPhysics : LastModel;
        float h = 470f;
        DrawBox(new Rect(x, y, w, h));
        float cy = y + 8;
        GUI.Label(new Rect(x + 10, cy, w - 20, 22), shown == EngineKind.Model ? "CrashSense AI (graph network)" : "CrashSense physics layer", title);
        cy += 22;
        var sc = LastScenario;
        var inv = CultureInfo.InvariantCulture;
        GUI.Label(new Rect(x + 10, cy, w - 20, 40), string.Format(inv,
            "{0:0} km/h, {1} hit, angle {2:0}°\n{3}", sc.Speed * 3.6, sc.Face, sc.AngleDeg, LastObstacleLabel), text);
        cy += 40;
        GUI.Label(new Rect(x + 10, cy, w - 20, 60), string.Format(inv,
            "crush depth <b>{0:0.00} m</b>  (physics layer {1:0.00} m)\ncabin intrusion {2:0.00} m   airbags: {3}\nreport in {4:0.00} ms (physics {5:0.00} ms)",
            r.CrushDepth, other.CrushDepth, r.Intrusion, r.AirbagFront || r.AirbagSide ? (r.AirbagFront ? "front" : "side") : "no",
            shown == EngineKind.Model ? LastModelMs : LastPhysicsMs, shown == EngineKind.Model ? LastPhysicsMs : LastModelMs), text);
        cy += 60;

        // top view of the 28 parts, coloured by damage level (front of the car at the top)
        float mapW = 120f, mapH = 250f, mx = x + 12, my = cy + 4;
        DrawCarMap(r, new Rect(mx, my, mapW, mapH));
        // worst parts
        float lx = mx + mapW + 14, ly = my;
        GUI.Label(new Rect(lx, ly, w - mapW - 30, 20), "<b>most damaged parts</b>", text);
        ly += 20;
        foreach (int i in Enumerable.Range(0, r.Damage.Length).OrderByDescending(i => r.Damage[i]).Take(10))
        {
            if (r.Damage[i] <= 0) break;
            var c = GUI.color;
            GUI.color = LevelColors[r.Level[i]];
            GUI.DrawTexture(new Rect(lx, ly + 5, 10, 10), Texture2D.whiteTexture);
            GUI.color = c;
            GUI.Label(new Rect(lx + 14, ly, 200, 20), string.Format(inv, "{0} {1:0.00}{2}", tables.PartNames[i], r.Damage[i], r.Detached[i] ? " off" : ""), text);
            ly += 19;
        }
        cy = my + mapH + 8;
        var fx = r.Effects.Where(kv => kv.Value).Select(kv => kv.Key.Replace('_', ' ')).ToArray();
        GUI.Label(new Rect(x + 10, cy, w - 20, 40), "effects: " + (fx.Length > 0 ? string.Join(", ", fx) : "none")
            + "\n<size=11>saved to Captures/crashsense_reports.jsonl (" + Reports + ")</size>", text);
    }

    static void DrawBox(Rect r)
    {
        var c = GUI.color;
        GUI.color = new Color(0.05f, 0.07f, 0.1f, 0.82f);
        GUI.DrawTexture(r, Texture2D.whiteTexture);
        GUI.color = c;
    }

    void DrawCarMap(Result r, Rect area)
    {
        // CrashSense car frame: x forward (-2.025..2.025), y left (-0.875..0.875)
        float L = 4.05f, W = 1.75f;
        float sx = area.width / W, sy = area.height / L;
        var c0 = GUI.color;
        GUI.color = new Color(1, 1, 1, 0.08f);
        GUI.DrawTexture(area, Texture2D.whiteTexture);
        // draw big, low parts first so that small parts stay visible on top
        foreach (int i in Enumerable.Range(0, tables.NParts).OrderByDescending(i => tables.Parts[i].Size[0] * tables.Parts[i].Size[1]))
        {
            var p = tables.Parts[i];
            float px = (float)(-p.Center[1] - p.Size[1] / 2 + W / 2), pz = (float)(L / 2 - (p.Center[0] + p.Size[0] / 2));
            var rect = new Rect(area.x + px * sx, area.y + pz * sy, Mathf.Max(2f, (float)p.Size[1] * sx), Mathf.Max(2f, (float)p.Size[0] * sy));
            var col = LevelColors[r.Level[i]];
            col.a = r.Damage[i] > 0 ? 0.55f + 0.45f * (float)r.Damage[i] : 0.35f;
            GUI.color = col;
            GUI.DrawTexture(rect, Texture2D.whiteTexture);
        }
        GUI.color = c0;
        GUI.Label(new Rect(area.x, area.y - 2, area.width, 16), "<size=10>front</size>", text);
    }
}

/// <summary>Puts a CrashSenseCar on every car with CarDamage when a scene starts, so no manual set-up is needed.</summary>
public static class CrashSenseAutoAttach
{
    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    static void Attach()
    {
#if UNITY_6000_6_OR_NEWER
        var cars = UnityEngine.Object.FindObjectsByType<CarDamage>();
#elif UNITY_2023_1_OR_NEWER
        var cars = UnityEngine.Object.FindObjectsByType<CarDamage>(FindObjectsSortMode.None);
#else
        var cars = UnityEngine.Object.FindObjectsOfType<CarDamage>();
#endif
        foreach (var car in cars)
            if (car.GetComponent<CrashSenseCar>() == null) car.gameObject.AddComponent<CrashSenseCar>();
    }
}
