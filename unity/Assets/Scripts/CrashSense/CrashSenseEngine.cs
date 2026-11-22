// CrashSense engine in plain C#: the physics layer and the trained graph network, with no Python server.
// It is a line-by-line copy of the JavaScript engine of the website (docs/js), which is itself checked
// against the Python code; tests/csharp checks this file against the same reference cases (tests/web/parity_cases.json).
// No UnityEngine dependency: the Unity scripts (CrashSenseCar etc.) call into this file.
//
// It is a simplified energy model for visual plausibility, not a validated crash-engineering model.
using System;
using System.Collections.Generic;
using System.Linq;

namespace CrashSense
{
    public class Material
    {
        public double Density, YoungsGpa, StrengthMpa, FractureStrain;
        public bool Brittle;
    }

    public class Part
    {
        public string Name, Kind, Material;
        public double ThicknessMm, CrushEff, AttachKn, MassKg;   // AttachKn = +inf for welded parts
        public double[] Center, Size, Section;
        public Part Copy() { return (Part)MemberwiseClone(); }
    }

    public class Obstacle
    {
        public string Name, Material, Shape, Form, After;
        public double Width, Depth, Height, WallMm, Perimeter, Stroke, ZBase;
        public double MassKg = double.NaN;                     // NaN = fixed in place
        public bool Fixed { get { return double.IsNaN(MassKg); } }
        public Obstacle Copy() { return (Obstacle)MemberwiseClone(); }
    }

    public class Vehicle
    {
        public double MassKg = 1100.0;
        public Part[] Parts;
    }

    public class Scenario
    {
        public Vehicle Vehicle;
        public Obstacle Obstacle;
        public string Face = "front";      // front, rear, left, right
        public double Offset;              // -1 .. 1 along the struck face
        public double AngleDeg;            // to the face normal
        public double Speed;               // m/s
    }

    /// <summary>Static data of the model: parts, graph, materials, presets, constants (data/static.json).</summary>
    public class Tables
    {
        public Dictionary<string, double> C = new Dictionary<string, double>();
        public Dictionary<string, double> ShapeCut = new Dictionary<string, double>();
        public double[] LevelEdges;
        public string[] LevelNames, Faces, Kinds, Shapes, Forms, AfterModes;
        public Dictionary<string, double> FacePlane = new Dictionary<string, double>(), FaceHalfSpan = new Dictionary<string, double>(),
            CabinDepth = new Dictionary<string, double>(), Extent = new Dictionary<string, double>();
        public Dictionary<string, Material> Materials = new Dictionary<string, Material>();
        public Part[] Parts;
        public string[] PartNames;
        public Dictionary<string, int> PartIndex = new Dictionary<string, int>();
        public int NParts;
        public double[] Adj;                                   // n*n, row i * n + column j
        public Dictionary<string, Obstacle> Presets = new Dictionary<string, Obstacle>();

        public static Tables FromJson(string json)
        {
            var d = MiniJson.O(MiniJson.Parse(json));
            var t = new Tables();
            foreach (var kv in MiniJson.O(d["constants"])) t.C[kv.Key] = MiniJson.D(kv.Value);
            foreach (var kv in MiniJson.O(d["shape_cut"])) t.ShapeCut[kv.Key] = MiniJson.D(kv.Value);
            t.LevelEdges = MiniJson.Ds(d["level_edges"]);
            t.LevelNames = MiniJson.Ss(d["level_names"]);
            t.Faces = MiniJson.Ss(d["faces"]);
            t.Kinds = MiniJson.Ss(d["kinds"]);
            t.Shapes = MiniJson.Ss(d["shapes"]);
            t.Forms = MiniJson.Ss(d["forms"]);
            t.AfterModes = MiniJson.Ss(d["after_modes"]);
            foreach (var kv in MiniJson.O(d["face_plane"])) t.FacePlane[kv.Key] = MiniJson.D(kv.Value);
            foreach (var kv in MiniJson.O(d["face_half_span"])) t.FaceHalfSpan[kv.Key] = MiniJson.D(kv.Value);
            foreach (var kv in MiniJson.O(d["cabin_depth"])) t.CabinDepth[kv.Key] = MiniJson.D(kv.Value);
            foreach (var kv in MiniJson.O(d["extent"])) t.Extent[kv.Key] = MiniJson.D(kv.Value);
            foreach (var kv in MiniJson.O(d["materials"]))
            {
                var m = MiniJson.O(kv.Value);
                t.Materials[kv.Key] = new Material
                {
                    Density = MiniJson.D(m["density"]), YoungsGpa = MiniJson.D(m["youngs_gpa"]),
                    StrengthMpa = MiniJson.D(m["strength_mpa"]), FractureStrain = MiniJson.D(m["fracture_strain"]),
                    Brittle = (bool)m["brittle"]
                };
            }
            var parts = MiniJson.L(d["parts"]);
            t.Parts = new Part[parts.Count];
            for (int i = 0; i < parts.Count; i++)
            {
                var p = MiniJson.O(parts[i]);
                t.Parts[i] = new Part
                {
                    Name = (string)p["name"], Kind = (string)p["kind"], Material = (string)p["material"],
                    ThicknessMm = MiniJson.D(p["thickness_mm"]), CrushEff = MiniJson.D(p["crush_eff"]),
                    AttachKn = p["attach_kn"] == null ? double.PositiveInfinity : MiniJson.D(p["attach_kn"]),
                    MassKg = MiniJson.D(p["mass_kg"]), Center = MiniJson.Ds(p["center"]), Size = MiniJson.Ds(p["size"]),
                    Section = MiniJson.Ds(p["section"])
                };
            }
            t.NParts = t.Parts.Length;
            t.PartNames = t.Parts.Select(p => p.Name).ToArray();
            for (int i = 0; i < t.NParts; i++) t.PartIndex[t.PartNames[i]] = i;
            int n = t.NParts;
            t.Adj = new double[n * n];
            foreach (var e in MiniJson.L(d["edges"]))
            {
                var ed = MiniJson.L(e);
                int i = (int)MiniJson.D(ed[0]), j = (int)MiniJson.D(ed[1]);
                double w = MiniJson.D(ed[2]);
                t.Adj[i * n + j] = w;
                t.Adj[j * n + i] = w;
            }
            foreach (var kv in MiniJson.O(d["obstacle_presets"]))
                t.Presets[kv.Key] = ObstacleFrom(MiniJson.O(MiniJson.O(kv.Value)["default"]));
            return t;
        }

        static Obstacle ObstacleFrom(Dictionary<string, object> o)
        {
            return new Obstacle
            {
                Name = (string)o["name"], Material = (string)o["material"], Shape = (string)o["shape"], Form = (string)o["form"],
                After = (string)o["after"], Width = MiniJson.D(o["width"]), Depth = MiniJson.D(o["depth"]),
                Height = MiniJson.D(o["height"]), WallMm = MiniJson.D(o["wall_mm"]), Perimeter = MiniJson.D(o["perimeter"]),
                Stroke = MiniJson.D(o["stroke"]), ZBase = MiniJson.D(o["z_base"]),
                MassKg = o["mass_kg"] == null ? double.NaN : MiniJson.D(o["mass_kg"])
            };
        }
    }

    public class Geometry
    {
        public int Ax, Along;
        public double Sign, Width, Centre, Lo, Hi, Z0, Z1, Arm;
        public double[] Overlap, DepthStart, Stroke, Dist, Point;
    }

    /// <summary>The damage report (the same fields as the JSON report of the Python code).</summary>
    public class Result
    {
        public string Engine;
        public double[] Damage;
        public int[] Level;
        public bool[] Detached, Direct;
        public double CrushDepth, Intrusion, PeakForceKn, EDissKj, DeltaV, ObstacleSpeedAfter, ObstacleDamage;
        public double ECarKj = double.NaN, EObstacleKj = double.NaN, UnabsorbedKj = double.NaN, ObstacleDepth = double.NaN;
        public int ObstacleLevel;
        public bool Breakthrough, AirbagFront, AirbagSide, Through;
        public Dictionary<string, bool> Effects;
    }

    public static class MathX
    {
        // Sum the way numpy adds a float64 row (blocks of eight, then pairwise), so results match bit for bit.
        public static double SumNp(double[] a) { return SumNp(a, 0, a.Length); }

        static double SumNp(double[] a, int off, int n)
        {
            if (n < 8)
            {
                double res = 0;
                for (int i = 0; i < n; i++) res += a[off + i];
                return res;
            }
            if (n <= 128)
            {
                var r = new double[8];
                for (int j = 0; j < 8; j++) r[j] = a[off + j];
                int i = 8;
                for (; i < n - (n % 8); i += 8) for (int j = 0; j < 8; j++) r[j] += a[off + i + j];
                double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
                for (; i < n; i++) res += a[off + i];
                return res;
            }
            int n2 = n >> 1;
            n2 -= n2 % 8;
            return SumNp(a, off, n2) + SumNp(a, off + n2, n - n2);
        }

        public static double Erf(double x)
        {
            double ax = Math.Abs(x);
            if (ax < 2.5)
            {
                double term = ax, sum = ax, x2 = ax * ax;
                for (int n = 1; n < 80; n++)
                {
                    term *= -x2 / n;
                    double add = term / (2 * n + 1);
                    sum += add;
                    if (Math.Abs(add) < 1e-17 * Math.Abs(sum)) break;
                }
                return Math.Sign(x) * (2 / Math.Sqrt(Math.PI)) * sum;
            }
            if (ax > 6) return Math.Sign(x);
            double f = ax;
            for (int k = 60; k >= 1; k--) f = ax + (k / 2.0) / f;
            double erfc = Math.Exp(-ax * ax) / (Math.Sqrt(Math.PI) * f);
            return Math.Sign(x) * (1 - erfc);
        }

        public static double Gelu(double x) { return 0.5 * x * (1 + Erf(x * 0.7071067811865476)); }
        public static double Sigmoid(double x) { return 1 / (1 + Math.Exp(-x)); }
        public static double Radians(double deg) { return deg * (Math.PI / 180); }
    }

    /// <summary>The physics layer (crashdamage/physics.py).</summary>
    public class Physics
    {
        public readonly Tables T;
        public Physics(Tables t) { T = t; }
        double K(string name) { return T.C[name]; }

        public Vehicle DefaultVehicle(double massKg = 1100.0)
        {
            return new Vehicle { MassKg = massKg, Parts = T.Parts.Select(p => p.Copy()).ToArray() };
        }

        public Obstacle Preset(string name)
        {
            Obstacle o;
            if (!T.Presets.TryGetValue(name, out o)) throw new ArgumentException("unknown obstacle preset " + name);
            return o.Copy();
        }

        static readonly string[] ObstacleFields = { "name", "material", "shape", "form", "width", "depth", "height", "wall_mm",
            "perimeter", "mass_kg", "after", "stroke", "z_base" };

        /// <summary>Obstacle from a request dictionary (obstacle_from_dict): 0 and "" keep the preset value, except z_base.</summary>
        public Obstacle ObstacleFromDict(Dictionary<string, object> d)
        {
            string name = d.ContainsKey("name") && d["name"] != null ? (string)d["name"]
                : d.ContainsKey("preset") && d["preset"] != null ? (string)d["preset"] : "concrete_wall";
            string baseName = d.ContainsKey("base") && d["base"] != null ? (string)d["base"] : name;
            var ob = Preset(baseName);
            ob.Name = name;
            foreach (var kv in d)
            {
                if (Array.IndexOf(ObstacleFields, kv.Key) < 0 || kv.Key == "name" || kv.Value == null) continue;
                if (kv.Value is string && (string)kv.Value == "") continue;
                if (kv.Value is double && (double)kv.Value == 0 && kv.Key != "z_base") continue;
                switch (kv.Key)
                {
                    case "material": ob.Material = (string)kv.Value; if (!T.Materials.ContainsKey(ob.Material)) throw new ArgumentException("unknown material " + ob.Material); break;
                    case "shape": ob.Shape = (string)kv.Value; break;
                    case "form": ob.Form = (string)kv.Value; break;
                    case "after": ob.After = (string)kv.Value; break;
                    case "width": ob.Width = MiniJson.D(kv.Value); break;
                    case "depth": ob.Depth = MiniJson.D(kv.Value); break;
                    case "height": ob.Height = MiniJson.D(kv.Value); break;
                    case "wall_mm": ob.WallMm = MiniJson.D(kv.Value); break;
                    case "perimeter": ob.Perimeter = MiniJson.D(kv.Value); break;
                    case "mass_kg": ob.MassKg = MiniJson.D(kv.Value); break;
                    case "stroke": ob.Stroke = MiniJson.D(kv.Value); break;
                    case "z_base": ob.ZBase = MiniJson.D(kv.Value); break;
                }
            }
            bool fixedFlag = (d.ContainsKey("fixed") && d["fixed"] is bool && (bool)d["fixed"])
                || (d.ContainsKey("fixed_in_place") && d["fixed_in_place"] is bool && (bool)d["fixed_in_place"]);
            if (fixedFlag) ob.MassKg = double.NaN;
            return ob;
        }

        static Dictionary<string, object> Pairs(object x, string valueKey)
        {
            var o = new Dictionary<string, object>();
            if (x == null) return o;
            if (x is List<object>) { foreach (var it in (List<object>)x) { var e = MiniJson.O(it); o[(string)e["part"]] = e[valueKey]; } return o; }
            foreach (var kv in MiniJson.O(x)) o[kv.Key] = kv.Value;
            return o;
        }

        /// <summary>Scenario from a request dictionary (scenario_from_dict in physics.py; the JSON the game sends).</summary>
        public Scenario ScenarioFromDict(Dictionary<string, object> d)
        {
            var vd = d.ContainsKey("vehicle") && d["vehicle"] != null ? MiniJson.O(d["vehicle"]) : new Dictionary<string, object>();
            double mass = vd.ContainsKey("mass_kg") && vd["mass_kg"] != null ? MiniJson.D(vd["mass_kg"]) : 1100.0;
            if (mass == 0 || double.IsNaN(mass)) mass = 1100.0;
            double scale = vd.ContainsKey("thickness_scale") && vd["thickness_scale"] != null ? MiniJson.D(vd["thickness_scale"]) : 1.0;
            if (scale == 0 || double.IsNaN(scale)) scale = 1.0;
            var mats = Pairs(vd.ContainsKey("materials") ? vd["materials"] : null, "material");
            var thick = Pairs(vd.ContainsKey("thickness_mm") ? vd["thickness_mm"] : null, "mm");
            var parts = T.Parts.Select(p =>
            {
                var q = p.Copy();
                if (mats.ContainsKey(p.Name)) q.Material = (string)mats[p.Name];
                if (!T.Materials.ContainsKey(q.Material)) throw new ArgumentException("unknown material " + q.Material + " for part " + p.Name);
                q.ThicknessMm = thick.ContainsKey(p.Name) ? MiniJson.D(thick[p.Name]) : p.ThicknessMm * scale;
                return q;
            }).ToArray();
            var obd = d.ContainsKey("obstacle") && d["obstacle"] != null ? MiniJson.O(d["obstacle"])
                : new Dictionary<string, object> { { "name", "concrete_wall" } };
            double speedMs = d.ContainsKey("speed_ms") && d["speed_ms"] != null ? MiniJson.D(d["speed_ms"]) : 0.0;
            double speed = speedMs > 0 ? speedMs
                : (d.ContainsKey("speed_kmh") && d["speed_kmh"] != null ? MiniJson.D(d["speed_kmh"]) : 50.0) / 3.6;
            var sc = new Scenario
            {
                Vehicle = new Vehicle { MassKg = mass, Parts = parts },
                Obstacle = ObstacleFromDict(obd),
                Face = d.ContainsKey("face") && d["face"] != null ? (string)d["face"] : "front",
                Offset = d.ContainsKey("offset") && d["offset"] != null ? MiniJson.D(d["offset"]) : 0.0,
                AngleDeg = d.ContainsKey("angle_deg") && d["angle_deg"] != null ? MiniJson.D(d["angle_deg"]) : 0.0,
                Speed = speed,
            };
            if (Array.IndexOf(T.Faces, sc.Face) < 0) throw new ArgumentException("face must be one of " + string.Join(", ", T.Faces));
            if (!(sc.Offset >= -1 && sc.Offset <= 1)) throw new ArgumentException("offset must be between -1 and 1");
            if (!(sc.Speed >= 0)) throw new ArgumentException("speed must be >= 0");
            return sc;
        }

        public int LevelOf(double d)
        {
            int k = 0;
            foreach (var e in T.LevelEdges) if (e <= d) k++;
            return k;
        }

        public double YawInertia(Vehicle v) { return v.MassKg * (K("CAR_LENGTH") * K("CAR_LENGTH") + K("CAR_WIDTH") * K("CAR_WIDTH")) / 12.0; }

        public Geometry ContactGeometry(Scenario sc)
        {
            var ob = sc.Obstacle; string face = sc.Face; var parts = sc.Vehicle.Parts;
            double H = K("CAR_HEIGHT");
            int ax = face == "front" || face == "rear" ? 0 : 1, along = 1 - ax;
            double sign = face == "front" || face == "left" ? 1.0 : -1.0;
            double half = T.FaceHalfSpan[face];
            double width = Math.Min(ob.Width, 2 * half);
            double c = sc.Offset * (half - width / 2);
            double lo = c - width / 2, hi = c + width / 2;
            double z0 = Math.Max(0.0, ob.ZBase), z1 = Math.Min(H, ob.ZBase + ob.Height);
            if (z1 <= z0) { z0 = H - 0.05; z1 = H; }   // the object passes over the car; it touches the roof line
            int n = parts.Length;
            var g = new Geometry
            {
                Ax = ax, Along = along, Sign = sign, Width = width, Centre = c, Lo = lo, Hi = hi, Z0 = z0, Z1 = z1, Arm = Math.Abs(c),
                Overlap = new double[n], DepthStart = new double[n], Stroke = new double[n], Dist = new double[n], Point = new double[3]
            };
            for (int i = 0; i < n; i++)
            {
                var p = parts[i];
                double qMax = sign * p.Center[ax] + p.Size[ax] / 2;
                g.DepthStart[i] = Math.Max(0.0, T.FacePlane[face] - qMax);
                g.Stroke[i] = p.Size[ax] * p.CrushEff;
                double a0 = p.Center[along] - p.Size[along] / 2, a1 = p.Center[along] + p.Size[along] / 2;
                double ovA = Math.Max(Math.Min(hi, a1) - Math.Max(lo, a0), 0) / p.Size[along];
                double zp0 = p.Center[2] - p.Size[2] / 2, zp1 = p.Center[2] + p.Size[2] / 2;
                double ovZ = Math.Max(Math.Min(z1, zp1) - Math.Max(z0, zp0), 0) / p.Size[2];
                g.Overlap[i] = Math.Min(Math.Max(ovA * ovZ, 0), 1);
            }
            g.Point[ax] = sign * T.FacePlane[face];
            g.Point[along] = c;
            g.Point[2] = 0.5 * (z0 + z1);
            for (int i = 0; i < n; i++)
            {
                var ctr = parts[i].Center;
                double dx = ctr[0] - g.Point[0], dy = ctr[1] - g.Point[1], dz = ctr[2] - g.Point[2];
                g.Dist[i] = Math.Sqrt(dx * dx + dy * dy + dz * dz);
            }
            return g;
        }

        public void CarForces(Scenario sc, Geometry geo, out List<double> b, out List<double[]> f, out double[] fPart, out bool[] engaged)
        {
            var parts = sc.Vehicle.Parts; int n = parts.Length, ax = geo.Ax;
            double cutSheet = T.ShapeCut[sc.Obstacle.Shape];
            fPart = new double[n]; engaged = new bool[n];
            var start = geo.DepthStart;
            for (int i = 0; i < n; i++)
            {
                var p = parts[i];
                double sigma = T.Materials[p.Material].StrengthMpa * 1e6;
                double t = p.ThicknessMm * 1e-3;
                double cut = t * 1e3 < K("SHEET_MM") ? cutSheet : 1.0;
                double fFull = K("FOLD") * sigma * t * p.Section[ax] * cut;
                engaged[i] = geo.Overlap[i] > K("MIN_OVERLAP");
                fPart[i] = engaged[i] ? fFull * geo.Overlap[i] : 0.0;
            }
            var end = new double[n];
            for (int i = 0; i < n; i++) end[i] = start[i] + geo.Stroke[i];
            double maxDepth = T.Extent[sc.Face];
            var pts = new List<double> { 0.0 };
            for (int i = 0; i < n; i++) if (engaged[i]) pts.Add(start[i]);
            for (int i = 0; i < n; i++) if (engaged[i]) pts.Add(end[i]);
            pts.Add(maxDepth);
            var sorted = pts.Select(x => Math.Min(Math.Max(x, 0.0), maxDepth)).ToList();
            sorted.Sort();
            b = new List<double>();
            for (int i = 0; i < sorted.Count; i++) if (i == 0 || sorted[i] != sorted[i - 1]) b.Add(sorted[i]);
            var packed = new double[n];
            for (int i = 0; i < n; i++)
                packed[i] = parts[i].CrushEff < K("BLOCK_EFF") ? Math.Min(K("DENS") * fPart[i], K("F_BLOCK_CAP")) : 0.0;
            f = new List<double[]>();
            for (int k = 0; k < b.Count - 1; k++)
            {
                double mid = 0.5 * (b[k] + b[k + 1]);
                var row = new double[n];
                for (int i = 0; i < n; i++) row[i] = mid < start[i] ? 0.0 : (mid < end[i] ? fPart[i] : packed[i]);
                f.Add(row);
            }
        }

        public double PlateauForce(Obstacle ob, double contactWidth, double contactHeight)
        {
            double sigma = T.Materials[ob.Material].StrengthMpa * 1e6;
            double w = Math.Min(ob.Width, contactWidth);
            switch (ob.Form)
            {
                case "solid": return K("SOLID_K") * sigma * w * ob.Depth;
                case "thin_wall": return K("FOLD") * sigma * ob.WallMm * 1e-3 * ob.Perimeter;
                case "slender": return K("SLENDER_K") * sigma * w * ob.Depth;
                case "foam": return sigma * w * Math.Min(ob.Height, contactHeight);
            }
            throw new ArgumentException("unknown form " + ob.Form);
        }

        public void ImpactEnergy(Scenario sc, Geometry geo, out double eDiss, out double dv, out double vObs)
        {
            double mC = sc.Vehicle.MassKg; var ob = sc.Obstacle;
            double invEffCar = 1.0 / mC + (geo.Arm * geo.Arm) / YawInertia(sc.Vehicle);
            double invMu = invEffCar + (ob.Fixed ? 0.0 : 1.0 / ob.MassKg);
            double mu = 1.0 / invMu;
            double vN = sc.Speed * Math.Cos(MathX.Radians(sc.AngleDeg));
            double e = K("RESTITUTION");
            eDiss = 0.5 * mu * (vN * vN) * (1 - e * e);
            double impulse = (1 + e) * mu * vN;
            dv = impulse / mC;
            vObs = ob.Fixed ? 0.0 : impulse / ob.MassKg;
        }

        class Solution
        {
            public double[] EP, FMax;
            public double Dc, DO, EObs, Peak, ELeft;
            public bool Breakthrough, Through;
        }

        // March the two bodies into each other. The body with the lower current resistance gives way.
        static Solution Solve(List<double> bc, List<double[]> fc, double[] bo, double[] foSeg, double eDiss)
        {
            int nSeg = fc.Count, n = fc[0].Length;
            var fcTot = fc.Select(MathX.SumNp).ToArray();
            var ahead = new bool[nSeg];
            bool any = false;
            for (int k = nSeg - 1; k >= 0; k--) { any = any || fcTot[k] > 0; ahead[k] = any; }
            var s = new Solution { EP = new double[n], FMax = new double[n] };
            int kc = 0, ko = 0;
            double dc = 0.0, dO = 0.0, eLeft = eDiss, eObs = 0.0, peak = 0.0;
            for (int iter = 0; iter < 10000; iter++)
            {
                if (eLeft <= 1e-6) break;
                double fCar = fcTot[kc], fObs = foSeg[ko];
                if (fObs <= 0.0) { s.Breakthrough = true; break; }
                if (fCar <= fObs)
                {
                    if (fCar <= 0 && !ahead[kc]) { s.Through = true; break; }
                    double length = bc[kc + 1] - dc;
                    double adv = fCar <= 0 ? length : Math.Min(length, eLeft / fCar);
                    for (int i = 0; i < n; i++) { s.EP[i] += fc[kc][i] * adv; s.FMax[i] = Math.Max(s.FMax[i], fc[kc][i]); }
                    peak = Math.Max(peak, fCar);
                    eLeft -= fCar * adv;
                    dc += adv;
                    if (adv >= length - 1e-12 && eLeft > 1e-6)
                    {
                        if (kc + 2 < bc.Count) kc += 1;
                        else { s.Through = true; break; }
                    }
                }
                else
                {
                    double length = bo[ko + 1] - dO;
                    double adv = Math.Min(length, eLeft / fObs);
                    eObs += fObs * adv;
                    eLeft -= fObs * adv;
                    dO += adv;
                    for (int i = 0; i < n; i++) s.FMax[i] = Math.Max(s.FMax[i], fc[kc][i] * (fObs / fCar));
                    peak = Math.Max(peak, fObs);
                    if (adv >= length - 1e-12 && ko + 2 < bo.Length) ko += 1;
                }
            }
            s.Dc = dc; s.DO = dO; s.EObs = eObs; s.Peak = peak; s.ELeft = Math.Max(eLeft, 0.0);
            return s;
        }

        public Result Simulate(Scenario sc)
        {
            var parts = sc.Vehicle.Parts; int n = parts.Length;
            var geo = ContactGeometry(sc);
            double eDiss, dv, vObs;
            ImpactEnergy(sc, geo, out eDiss, out dv, out vObs);
            List<double> bc; List<double[]> fc; double[] fPart; bool[] engaged;
            CarForces(sc, geo, out bc, out fc, out fPart, out engaged);
            var ob = sc.Obstacle;
            double fO = PlateauForce(ob, geo.Width, geo.Z1 - geo.Z0);
            double after = ob.After == "rigid" ? K("RIGID_FORCE") : 0.0;
            var bo = new[] { 0.0, Math.Max(ob.Stroke, 1e-6), K("BIG_DEPTH") };
            var foSeg = new[] { fO, after };
            var sol = Solve(bc, fc, bo, foSeg, eDiss);

            var capE = new double[n];
            for (int i = 0; i < n; i++) capE[i] = engaged[i] ? fPart[i] * geo.Stroke[i] : 0.0;
            var direct = new bool[n]; var dDir = new double[n]; var fDir = new double[n];
            for (int i = 0; i < n; i++)
            {
                double plastic = capE[i] > 0 ? sol.EP[i] / capE[i] : 0.0;
                double elastic = K("ELASTIC_FLOOR") * (fPart[i] > 0 ? Math.Min(sol.FMax[i], fPart[i]) / fPart[i] : 0.0);
                direct[i] = engaged[i] && sol.FMax[i] > 0;
                dDir[i] = direct[i] ? Math.Min(Math.Max(Math.Max(plastic, elastic), 0), 1) : 0.0;
                fDir[i] = direct[i] ? sol.FMax[i] : 0.0;
            }
            // force travels through the part graph
            var fYield = new double[n]; var gd = new double[n];
            for (int i = 0; i < n; i++)
            {
                var p = parts[i];
                double sigma = T.Materials[p.Material].StrengthMpa * 1e6;
                double meanSection = (p.Section[0] + p.Section[1] + p.Section[2]) / 3;
                fYield[i] = K("FOLD") * sigma * (p.ThicknessMm * 1e-3) * meanSection;
                gd[i] = Math.Min(Math.Max(T.Materials[p.Material].FractureStrain / 0.15, 0.2), 1.0);
            }
            var a = T.Adj;
            var fIn = new double[n]; var dIn = new double[n];
            int hops = (int)K("PROPAGATION_HOPS");
            for (int hop = 0; hop < hops; hop++)
            {
                double decay = hop == 0 ? 1.0 : K("HOP_DECAY");
                var carried = new double[n]; var nbSrc = new double[n];
                for (int i = 0; i < n; i++) { carried[i] = Math.Min(Math.Max(fDir[i], fIn[i]), fYield[i]); nbSrc[i] = Math.Max(dDir[i], dIn[i]); }
                var nbCap = new double[n];
                for (int j = 0; j < n; j++)
                {
                    double mf = double.NegativeInfinity, md = double.NegativeInfinity;
                    for (int i = 0; i < n; i++) { mf = Math.Max(mf, a[i * n + j] * carried[i]); md = Math.Max(md, a[i * n + j] * nbSrc[i]); }
                    fIn[j] = Math.Max(fIn[j], mf * decay);
                    nbCap[j] = md * decay;
                }
                for (int j = 0; j < n; j++)
                {
                    double dNew = Math.Min(K("INDIRECT_GAIN") * Math.Min(1.0, fIn[j] / fYield[j]) / gd[j], nbCap[j]);
                    dIn[j] = Math.Max(dIn[j], Math.Min(dNew, 1.0));
                }
            }
            var r = new Result { Engine = "physics", Damage = new double[n], Level = new int[n], Detached = new bool[n], Direct = direct };
            for (int i = 0; i < n; i++)
            {
                double d = Math.Min(Math.Max(Math.Max(dDir[i], dIn[i]), 0), 1);
                if (d < 0.01) d = 0.0;
                r.Damage[i] = d;
                double f = Math.Max(fDir[i], fIn[i]);
                r.Detached[i] = f >= parts[i].AttachKn * 1e3 && d >= K("DETACH_MIN_DAMAGE");
                r.Level[i] = LevelOf(d);
            }
            double capO = fO * ob.Stroke;
            double dOb = capO > 0 ? Math.Min(1.0, sol.EObs / capO) : 0.0;
            if (sol.Breakthrough && ob.After == "break") dOb = 1.0;
            bool faceFront = sc.Face == "front" && Math.Abs(sc.AngleDeg) <= 30;
            bool side = sc.Face == "left" || sc.Face == "right";
            r.CrushDepth = sol.Dc;
            r.Intrusion = Math.Max(0.0, sol.Dc - T.CabinDepth[sc.Face]);
            r.PeakForceKn = sol.Peak / 1e3;
            r.EDissKj = eDiss / 1e3;
            r.ECarKj = MathX.SumNp(sol.EP) / 1e3;
            r.EObstacleKj = sol.EObs / 1e3;
            r.UnabsorbedKj = sol.ELeft / 1e3;
            r.ObstacleDamage = dOb;
            r.ObstacleLevel = LevelOf(dOb);
            r.ObstacleDepth = sol.DO;
            r.Breakthrough = sol.Breakthrough;
            r.AirbagFront = faceFront && dv >= K("AIRBAG_FRONT_DV");
            r.AirbagSide = side && dv >= K("AIRBAG_SIDE_DV");
            r.DeltaV = dv;
            r.ObstacleSpeedAfter = vObs;
            r.Through = sol.Through;
            r.Effects = EffectHints(sc, r);
            return r;
        }

        static readonly string[] Metals = { "mild_steel", "high_strength_steel", "ultra_high_strength_steel", "aluminium" };

        public Dictionary<string, bool> EffectHints(Scenario sc, Result res)
        {
            var parts = sc.Vehicle.Parts;
            bool hard = sc.Obstacle.Material == "concrete" || sc.Obstacle.Material == "steel";
            bool metalHit = false, glassBreak = false;
            for (int i = 0; i < parts.Length; i++)
            {
                if (res.Direct[i] && Metals.Contains(parts[i].Material) && res.Level[i] >= 2) metalHit = true;
                if (parts[i].Kind == "glass" && res.Level[i] >= 3) glassBreak = true;
            }
            bool leak = res.Level[T.PartIndex["radiator"]] >= 3 || res.Level[T.PartIndex["engine"]] >= 3;
            bool broken = res.Detached.Any(x => x) || res.Level.Any(l => l >= 4) || res.ObstacleDamage >= 1.0 || res.Through;
            return new Dictionary<string, bool>
            {
                { "sparks", hard && metalHit && res.EDissKj > 5.0 },
                { "glass_shards", glassBreak },
                { "smoke", leak },
                { "fluid_leak", leak },
                { "debris", broken },
                { "dust", (sc.Obstacle.Material == "concrete" || sc.Obstacle.Material == "wood") && res.ObstacleLevel >= 2 },
                { "airbag_front", res.AirbagFront },
                { "airbag_side", res.AirbagSide },
            };
        }

        // ---- network inputs (crashdamage/features.py) --------------------------------------------------------

        double[] MaterialFeatures(string name)
        {
            var m = T.Materials[name];
            return new[] { Math.Log10(m.Density), Math.Log10(m.YoungsGpa), Math.Log10(m.StrengthMpa), Math.Log10(m.FractureStrain), m.Brittle ? 1.0 : 0.0 };
        }

        List<double> PartStaticFeatures(Part p)
        {
            var r = new List<double>(MaterialFeatures(p.Material)) { p.ThicknessMm, Math.Log10(p.MassKg) };
            r.AddRange(p.Center); r.AddRange(p.Size);
            r.Add(p.CrushEff); r.Add(double.IsInfinity(p.AttachKn) ? 3.0 : Math.Log10(p.AttachKn));
            r.AddRange(p.Section);
            foreach (var k in T.Kinds) r.Add(p.Kind == k ? 1.0 : 0.0);
            return r;
        }

        List<double> ObstacleFeatures(Obstacle ob)
        {
            var r = new List<double>(MaterialFeatures(ob.Material));
            foreach (var s in T.Shapes) r.Add(ob.Shape == s ? 1.0 : 0.0);
            foreach (var f in T.Forms) r.Add(ob.Form == f ? 1.0 : 0.0);
            double mass = (!ob.Fixed && ob.MassKg != 0) ? Math.Log10(ob.MassKg) : 0.0;
            r.AddRange(new[] { ob.Width, ob.Depth, ob.Height, ob.WallMm, ob.Perimeter, mass, ob.Fixed ? 1.0 : 0.0,
                ob.After == "break" ? 1.0 : 0.0, ob.Stroke, ob.ZBase });
            return r;
        }

        /// <summary>Model inputs: per-part rows (28 x 31) and the global row (39), physics features included, rounded to float32.</summary>
        public void ScenarioInputs(Scenario sc, Geometry geo, out float[] node, out float[] glob, out int nodeLen)
        {
            var parts = sc.Vehicle.Parts; int n = parts.Length;
            double eDiss, dv, vObs;
            ImpactEnergy(sc, geo, out eDiss, out dv, out vObs);
            List<double> b; List<double[]> f; double[] fPart; bool[] engaged;
            CarForces(sc, geo, out b, out f, out fPart, out engaged);
            var cap = new double[n];
            for (int i = 0; i < n; i++) cap[i] = (engaged[i] ? fPart[i] * geo.Stroke[i] : 0.0) / 1e3;
            double eDissKj = eDiss / 1e3;
            double logE = Math.Log10(Math.Max(eDissKj, 1e-3));
            double logCapTotal = Math.Log10(Math.Max(MathX.SumNp(cap), 1e-3));
            nodeLen = 24 + 5 + 2;
            node = new float[n * nodeLen];
            for (int i = 0; i < n; i++)
            {
                var row = PartStaticFeatures(parts[i]);
                row.AddRange(new[] { geo.Overlap[i], geo.DepthStart[i], geo.Stroke[i], geo.Dist[i], geo.Overlap[i] > K("MIN_OVERLAP") ? 1.0 : 0.0,
                    Math.Log10(cap[i] + 1e-3), Math.Log10(eDissKj / (cap[i] + 1e-3) + 1e-3) });
                for (int k = 0; k < nodeLen; k++) node[i * nodeLen + k] = (float)row[k];
            }
            double vN = sc.Speed * Math.Cos(MathX.Radians(sc.AngleDeg));
            var g = new List<double>();
            foreach (var fc in T.Faces) g.Add(sc.Face == fc ? 1.0 : 0.0);
            g.AddRange(new[] { sc.Speed, vN, sc.AngleDeg, sc.Offset, geo.Width, geo.Centre, geo.Z0, geo.Z1, geo.Arm, Math.Log10(sc.Vehicle.MassKg) });
            g.AddRange(ObstacleFeatures(sc.Obstacle));
            g.AddRange(new[] { logE, logCapTotal, logE - logCapTotal });
            glob = g.Select(x => (float)x).ToArray();
        }
    }

    /// <summary>The trained graph network (crashdamage/models.py GraphNet), weights from data/model.json + model.bin.</summary>
    public class GraphModel
    {
        readonly Dictionary<string, float[]> w = new Dictionary<string, float[]>();
        public int Hidden, Layers, NNode, NGlob;

        public static GraphModel Load(string manifestJson, byte[] bin)
        {
            var m = new GraphModel();
            var d = MiniJson.O(MiniJson.Parse(manifestJson));
            var cfg = MiniJson.O(d["config"]);
            m.Hidden = (int)MiniJson.D(cfg["hidden"]);
            m.Layers = (int)MiniJson.D(cfg["layers"]);
            m.NNode = (int)MiniJson.D(d["n_node_feat"]);
            m.NGlob = (int)MiniJson.D(d["n_glob_feat"]);
            if (!(bool)cfg["physics_features"]) throw new NotSupportedException("the C# engine expects a model trained with physics features");
            foreach (var te in MiniJson.L(d["tensors"]))
            {
                var e = MiniJson.O(te);
                int count = 1;
                foreach (var s in MiniJson.L(e["shape"])) count *= (int)MiniJson.D(s);
                var a = new float[count];
                Buffer.BlockCopy(bin, (int)MiniJson.D(e["offset"]), a, 0, count * 4);
                m.w[(string)e["name"]] = a;
            }
            return m;
        }

        double[] Linear(double[] x, int rows, int inDim, string name, int outDim)
        {
            var W = w[name + ".weight"]; var b = w[name + ".bias"];
            var o = new double[rows * outDim];
            for (int r = 0; r < rows; r++)
            {
                int xo = r * inDim;
                for (int k = 0; k < outDim; k++)
                {
                    double acc = b[k];
                    int wo = k * inDim;
                    for (int j = 0; j < inDim; j++) acc += W[wo + j] * x[xo + j];
                    o[r * outDim + k] = acc;
                }
            }
            return o;
        }

        double[] Mlp(double[] x, int rows, int inDim, string name, int hidden, int outDim)
        {
            var a = Linear(x, rows, inDim, name + ".0", hidden);
            for (int i = 0; i < a.Length; i++) a[i] = MathX.Gelu(a[i]);
            return Linear(a, rows, hidden, name + ".2", outDim);
        }

        void LayerNorm(double[] x, int rows, int dim, string name)
        {
            var g = w[name + ".weight"]; var b = w[name + ".bias"];
            for (int r = 0; r < rows; r++)
            {
                double mean = 0;
                for (int k = 0; k < dim; k++) mean += x[r * dim + k];
                mean /= dim;
                double v = 0;
                for (int k = 0; k < dim; k++) { double q = x[r * dim + k] - mean; v += q * q; }
                v /= dim;
                double inv = 1 / Math.Sqrt(v + 1e-5);
                for (int k = 0; k < dim; k++) x[r * dim + k] = (x[r * dim + k] - mean) * inv * g[k] + b[k];
            }
        }

        public void Forward(float[] nodeX, float[] globX, int N, out double[] node, out double[] glob)
        {
            int H = Hidden; var adj = w["adj"];
            var g = Mlp(Array.ConvertAll(globX, v => (double)v), 1, NGlob, "glob_in", H, H);
            var h = Mlp(Array.ConvertAll(nodeX, v => (double)v), N, NNode, "node_in", H, H);
            var cat3 = new double[N * 3 * H];
            for (int k = 0; k < Layers; k++)
            {
                var mm = Linear(h, N, H, "msg." + k, H);
                for (int i = 0; i < N; i++)
                {
                    int o = i * 3 * H;
                    for (int c = 0; c < H; c++) { cat3[o + c] = h[i * H + c]; cat3[o + H + c] = 0; cat3[o + 2 * H + c] = g[c]; }
                    for (int j = 0; j < N; j++)
                    {
                        double a = adj[i * N + j];
                        if (a == 0) continue;
                        for (int c = 0; c < H; c++) cat3[o + H + c] += a * mm[j * H + c];
                    }
                }
                var u = Mlp(cat3, N, 3 * H, "upd." + k, H, H);
                for (int i = 0; i < h.Length; i++) h[i] += u[i];
                LayerNorm(h, N, H, "norm." + k);
            }
            var cat2 = new double[N * 2 * H];
            for (int i = 0; i < N; i++)
                for (int c = 0; c < H; c++) { cat2[i * 2 * H + c] = h[i * H + c]; cat2[i * 2 * H + H + c] = g[c]; }
            node = Mlp(cat2, N, 2 * H, "node_out", H, 2);
            var pooled = new double[3 * H];
            for (int c = 0; c < H; c++)
            {
                double s = 0, mx = double.NegativeInfinity;
                for (int i = 0; i < N; i++) { s += h[i * H + c]; mx = Math.Max(mx, h[i * H + c]); }
                pooled[c] = s / N; pooled[H + c] = mx; pooled[2 * H + c] = g[c];
            }
            glob = Mlp(pooled, 1, 3 * H, "glob_out", H, 7);
        }

        /// <summary>Damage report from the network (export.py Predictor._from_model).</summary>
        /// <summary>Raw network outputs (train.py predict_arrays): damage and detach probabilities per part,
        /// the four scenario numbers (crush m, intrusion m, peak MN, obstacle damage) and three flag probabilities.</summary>
        public void PredictRaw(float[] node, float[] glob, int n, out double[] damage, out double[] detachProb, out double[] reg, out double[] flags)
        {
            var nm = w["norm.node_mean"]; var ns = w["norm.node_std"]; var gm = w["norm.glob_mean"]; var gs = w["norm.glob_std"];
            var nodeN = new float[node.Length]; var globN = new float[glob.Length];
            for (int i = 0; i < n; i++) for (int f = 0; f < NNode; f++) nodeN[i * NNode + f] = (float)(((double)node[i * NNode + f] - nm[f]) / ns[f]);
            for (int f = 0; f < NGlob; f++) globN[f] = (float)(((double)glob[f] - gm[f]) / gs[f]);
            double[] on, og;
            Forward(nodeN, globN, n, out on, out og);
            var rm = w["reg_mean"]; var rs = w["reg_std"];
            damage = new double[n]; detachProb = new double[n];
            for (int i = 0; i < n; i++) { damage[i] = MathX.Sigmoid(on[i * 2]); detachProb[i] = MathX.Sigmoid(on[i * 2 + 1]); }
            reg = new double[4];
            for (int k = 0; k < 4; k++) reg[k] = og[k] * rs[k] + rm[k];
            flags = new double[3];
            for (int k = 0; k < 3; k++) flags[k] = MathX.Sigmoid(og[4 + k]);
        }

        public Result Predict(Physics ph, Scenario sc)
        {
            var T = ph.T; int n = T.NParts;
            var geo = ph.ContactGeometry(sc);
            float[] node, glob; int nodeLen;
            ph.ScenarioInputs(sc, geo, out node, out glob, out nodeLen);
            double[] dam, det, reg, fl;
            PredictRaw(node, glob, n, out dam, out det, out reg, out fl);
            var r = new Result { Engine = "model", Damage = new double[n], Level = new int[n], Detached = new bool[n], Direct = new bool[n] };
            for (int i = 0; i < n; i++)
            {
                double d = Math.Min(Math.Max(dam[i], 0), 1);
                if (d < 0.01) d = 0.0;
                r.Damage[i] = d;
                r.Level[i] = ph.LevelOf(d);
                r.Detached[i] = det[i] >= 0.5;
                r.Direct[i] = geo.Overlap[i] > T.C["MIN_OVERLAP"];
            }
            double eDiss, dv, vObs;
            ph.ImpactEnergy(sc, geo, out eDiss, out dv, out vObs);
            r.CrushDepth = Math.Max(reg[0], 0);
            r.Intrusion = Math.Max(reg[1], 0);
            r.PeakForceKn = Math.Max(reg[2], 0) * 1000.0;
            r.ObstacleDamage = Math.Min(Math.Max(reg[3], 0), 1);
            r.ObstacleLevel = ph.LevelOf(r.ObstacleDamage);
            r.EDissKj = eDiss / 1e3;
            r.DeltaV = dv;
            r.ObstacleSpeedAfter = vObs;
            r.AirbagFront = fl[0] >= 0.5;
            r.AirbagSide = fl[1] >= 0.5;
            r.Through = fl[2] >= 0.5;
            r.Effects = ph.EffectHints(sc, r);
            return r;
        }
    }
}
