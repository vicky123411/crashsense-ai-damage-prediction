// Checks the C# engine (unity/Assets/Scripts/CrashSense) against the Python reference cases.
//   dotnet run --project tests/csharp -c Release
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using CrashSense;

static class Program
{
    static int failures;
    static readonly Dictionary<string, double> Worst = new Dictionary<string, double>();

    static void Fail(string m) { failures++; if (failures <= 25) Console.WriteLine("FAIL " + m); }

    static void Num(string tag, double got, double want, double tol, string where)
    {
        if (double.IsNaN(want) && double.IsNaN(got)) return;
        double d = Math.Min(Math.Abs(got - want), Math.Abs(got - want) / (1e-12 + Math.Abs(want)));
        Worst[tag] = Math.Max(Worst.TryGetValue(tag, out var w) ? w : 0, d);
        if (!(d <= tol)) Fail($"{where} {tag}: got {got:R}, want {want:R}");
    }

    static void Arr(string tag, IList<double> got, IList<double> want, double tol, int i)
    {
        if (got.Count != want.Count) { Fail($"case {i} {tag}: length {got.Count} vs {want.Count}"); return; }
        for (int k = 0; k < want.Count; k++) Num(tag, got[k], want[k], tol, $"case {i}[{k}]");
    }

    static void Same<T>(string tag, IList<T> got, IList<T> want, int i)
    {
        if (!got.SequenceEqual(want)) Fail($"case {i} {tag}: [{string.Join(",", got)}] vs [{string.Join(",", want)}]");
    }

    static int Main(string[] args)
    {
        string root = args.Length > 0 ? args[0] : Path.Combine("..", "..");
        string data = Path.Combine(root, "docs", "data");
        var tables = Tables.FromJson(File.ReadAllText(Path.Combine(data, "static.json")));
        var ph = new CrashPhysics(tables);
        var model = GraphModel.Load(File.ReadAllText(Path.Combine(data, "model.json")), File.ReadAllBytes(Path.Combine(data, "model.bin")));
        var doc = MiniJson.O(MiniJson.Parse(File.ReadAllText(Path.Combine(root, "tests", "web", "parity_cases.json"))));

        var erf = MiniJson.O(doc["erf"]);
        var ex = MiniJson.Ds(erf["x"]); var ey = MiniJson.Ds(erf["y"]);
        for (int k = 0; k < ex.Length; k++) Num("erf", MathX.Erf(ex[k]), ey[k], 1e-14, "x=" + ex[k]);

        const double PHYS = 1e-9, MODEL = 2e-4;
        var cases = MiniJson.L(doc["cases"]);
        int nFlagDiffs = 0;
        var sw = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 0; i < cases.Count; i++)
        {
            var c = MiniJson.O(cases[i]);
            var e = MiniJson.O(c["expected"]);
            var p = MiniJson.O(e["physics"]);
            var sc = ph.ScenarioFromDict(MiniJson.O(c["request"]));
            var res = ph.Simulate(sc);
            Arr("damage", res.Damage, MiniJson.Ds(p["damage"]), PHYS, i);
            Same("level", res.Level, MiniJson.L(p["level"]).Select(x => (int)MiniJson.D(x)).ToList(), i);
            Same("detached", res.Detached, MiniJson.L(p["detached"]).Select(x => (bool)x).ToList(), i);
            Same("direct", res.Direct, MiniJson.L(p["direct"]).Select(x => (bool)x).ToList(), i);
            var nums = new (string, double)[] { ("crush_depth", res.CrushDepth), ("intrusion", res.Intrusion), ("peak_force_kn", res.PeakForceKn),
                ("e_diss_kj", res.EDissKj), ("e_car_kj", res.ECarKj), ("e_obstacle_kj", res.EObstacleKj), ("unabsorbed_kj", res.UnabsorbedKj),
                ("obstacle_damage", res.ObstacleDamage), ("obstacle_depth", res.ObstacleDepth), ("delta_v", res.DeltaV),
                ("obstacle_speed_after", res.ObstacleSpeedAfter) };
            foreach (var (k, v) in nums) Num(k, v, MiniJson.D(p[k]), PHYS, $"case {i}");
            if (res.ObstacleLevel != (int)MiniJson.D(p["obstacle_level"])) Fail($"case {i} obstacle_level");
            foreach (var (k, v) in new (string, bool)[] { ("breakthrough", res.Breakthrough), ("through", res.Through), ("airbag_front", res.AirbagFront), ("airbag_side", res.AirbagSide) })
                if (v != (bool)p[k]) Fail($"case {i} {k}: {v} vs {p[k]}");
            foreach (var kv in MiniJson.O(p["effects"]))
                if (res.Effects[kv.Key] != (bool)kv.Value) Fail($"case {i} effect {kv.Key}");

            var geo = ph.ContactGeometry(sc);
            var eg = MiniJson.O(e["geo"]);
            Arr("geo.overlap", geo.Overlap, MiniJson.Ds(eg["overlap"]), PHYS, i);
            Arr("geo.dist", geo.Dist, MiniJson.Ds(eg["dist"]), PHYS, i);
            Arr("geo.point", geo.Point, MiniJson.Ds(eg["point"]), PHYS, i);

            float[] node, glob; int nodeLen;
            ph.ScenarioInputs(sc, geo, out node, out glob, out nodeLen);
            if (e.ContainsKey("features") && e["features"] != null)
            {
                var ef = MiniJson.O(e["features"]);
                var en = MiniJson.L(ef["node"]).SelectMany(r => MiniJson.Ds(r)).ToList();
                Arr("features.node", node.Select(x => (double)x).ToList(), en, 2e-6, i);
                Arr("features.glob", glob.Select(x => (double)x).ToList(), MiniJson.Ds(ef["glob"]), 2e-6, i);
            }
            double[] dam, det, reg, fl;
            model.PredictRaw(node, glob, tables.NParts, out dam, out det, out reg, out fl);
            var em = MiniJson.O(e["model"]);
            Arr("model.damage", dam, MiniJson.Ds(em["damage"]), MODEL, i);
            Arr("model.detach_prob", det, MiniJson.Ds(em["detach_prob"]), MODEL, i);
            Arr("model.reg", reg, MiniJson.Ds(em["reg"]), MODEL, i);
            Arr("model.flags_prob", fl, MiniJson.Ds(em["flags_prob"]), MODEL, i);
            var mres = model.Predict(ph, sc);
            var mrep = MiniJson.O(e["model_report"]);
            var mparts = MiniJson.L(mrep["parts"]);
            for (int k = 0; k < mparts.Count; k++)
            {
                var mp = MiniJson.O(mparts[k]);
                if (Math.Abs(Math.Round(mres.Damage[k], 4) - MiniJson.D(mp["damage"])) > 1e-3) Fail($"case {i} model part {k}");
                if (mres.Detached[k] != (bool)mp["detached"]) nFlagDiffs++;
            }
        }
        sw.Stop();
        Console.WriteLine($"{cases.Count} cases checked, {failures} failures ({sw.Elapsed.TotalMilliseconds / cases.Count:F2} ms per case incl. checks).");
        foreach (var kv in Worst.OrderByDescending(kv => kv.Value).Take(10)) Console.WriteLine($"  {kv.Key,-22} {kv.Value:E2}");
        if (nFlagDiffs > 0) Console.WriteLine($"model yes/no outputs on the 0.5 line: {nFlagDiffs}");
        return failures > 0 ? 1 : 0;
    }
}
