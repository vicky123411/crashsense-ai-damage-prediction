using System;
using System.Collections.Generic;
using UnityEngine;

namespace CollisionDamage
{
    [Serializable]
    public class PartBinding
    {
        public string partId;                 // id from the damage report, e.g. "door_l"
        public Transform part;                // the object that represents it in the car
        public bool isGlass;                  // glass cracks and shatters instead of denting
        public Material crackedMaterial;      // optional, used for cracked glass
        public float detachedMassKg = 8f;     // mass of the loose part after it tears off
    }

    // Turns a damage report into visible damage on the car (and on the obstacle):
    // dents, darkened paint, cracked or shattered glass, parts that fall off, particle effects.
    // This only changes how things look. It does not change the physics of the crash.
    public class CollisionDamageApplier : MonoBehaviour
    {
        [Header("Parts")]
        public List<PartBinding> parts = new List<PartBinding>();

        [Header("Dents (meshes need Read/Write enabled in their import settings)")]
        public float maxDentDepth = 0.35f;       // metres at damage = 1
        public float dentRadius = 0.9f;          // metres around the impact point
        [Range(0f, 1f)] public float crumple = 0.35f;   // roughness of heavily damaged surfaces
        [Range(0f, 1f)] public float indirectFactor = 0.4f;   // parts hit only through their neighbours dent less
        public Color scorchColor = new Color(0.08f, 0.08f, 0.08f, 1f);
        [Range(0f, 1f)] public float maxScorch = 0.6f;

        [Header("Parts that fall off")]
        public float detachImpulse = 4f;
        public float detachedLifetime = 20f;     // seconds, 0 = never removed

        [Header("Effects (all optional)")]
        public GameObject sparksPrefab;
        public GameObject smokePrefab;
        public GameObject glassShardsPrefab;
        public GameObject debrisPrefab;
        public GameObject dustPrefab;
        public GameObject airbagFront;           // switched on when the report asks for it
        public GameObject airbagSide;
        public float effectLifetime = 6f;

        [Header("Obstacle")]
        public float obstacleDentDepth = 0.25f;
        public float obstacleDentRadius = 0.8f;

        readonly Dictionary<MeshFilter, Vector3[]> originalVertices = new Dictionary<MeshFilter, Vector3[]>();
        readonly Dictionary<Renderer, Color[]> originalColors = new Dictionary<Renderer, Color[]>();

        // Fills the part list from child objects whose names are part ids, e.g. a child called "door_l".
        [ContextMenu("Bind parts by child name")]
        public void BindByChildName()
        {
            string[] ids =
            {
                "bumper_front", "beam_front", "rail_front_l", "rail_front_r", "radiator", "engine", "bonnet",
                "fender_l", "fender_r", "windscreen", "pillar_a_l", "pillar_a_r", "firewall", "door_l", "door_r",
                "pillar_b_l", "pillar_b_r", "sill_l", "sill_r", "floor_cabin", "roof", "rear_glass", "bumper_rear",
                "rear_structure", "wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"
            };
            parts.Clear();
            Transform[] all = GetComponentsInChildren<Transform>(true);
            foreach (string id in ids)
            {
                foreach (Transform t in all)
                {
                    if (t == transform || t.name != id) continue;
                    PartBinding b = new PartBinding();
                    b.partId = id;
                    b.part = t;
                    b.isGlass = id == "windscreen" || id == "rear_glass";
                    parts.Add(b);
                    break;
                }
            }
            Debug.Log("Bound " + parts.Count + " of " + ids.Length + " parts by name.");
        }

        // worldImpactPoint: where the obstacle touched the car.
        // pushDirection: direction the obstacle pushes (from the contact point into the car).
        public void Apply(DamageReport report, Vector3 worldImpactPoint, Vector3 pushDirection)
        {
            if (report == null || report.parts == null) return;
            Vector3 dir = pushDirection.sqrMagnitude > 1e-6f ? pushDirection.normalized : transform.forward;

            Dictionary<string, PartBinding> lookup = new Dictionary<string, PartBinding>();
            foreach (PartBinding b in parts)
            {
                if (b != null && !string.IsNullOrEmpty(b.partId)) lookup[b.partId] = b;
            }

            foreach (PartDamage pd in report.parts)
            {
                PartBinding binding;
                if (!lookup.TryGetValue(pd.id, out binding) || binding.part == null) continue;
                if (pd.level <= 0 && !pd.detached) continue;

                if (binding.isGlass)
                {
                    ApplyGlass(binding, pd, worldImpactPoint);
                }
                else
                {
                    float scale = pd.direct ? 1f : indirectFactor;
                    foreach (MeshFilter mf in binding.part.GetComponentsInChildren<MeshFilter>())
                    {
                        DentMesh(mf, worldImpactPoint, dir, maxDentDepth * pd.damage * scale, dentRadius, pd.level);
                    }
                    Scorch(binding.part, pd.damage);
                }

                if (pd.detached) Detach(binding, dir);
            }

            SpawnEffects(report.effects, worldImpactPoint, dir);
        }

        // Shows the damage of the obstacle: a dent, and fences or posts that snap fall over.
        public void ApplyToObstacle(Transform obstacle, DamageReport report, Vector3 worldImpactPoint, Vector3 pushDirection)
        {
            if (obstacle == null || report == null || report.obstacle == null) return;
            ObstacleDamage od = report.obstacle;
            Vector3 dir = pushDirection.sqrMagnitude > 1e-6f ? pushDirection.normalized : Vector3.forward;

            foreach (MeshFilter mf in obstacle.GetComponentsInChildren<MeshFilter>())
            {
                DentMesh(mf, worldImpactPoint, dir, obstacleDentDepth * od.damage, obstacleDentRadius, od.level);
            }

            if (od.broken_through || od.level >= 4)
            {
                Rigidbody rb = obstacle.GetComponent<Rigidbody>();
                if (rb == null) rb = obstacle.gameObject.AddComponent<Rigidbody>();
                rb.isKinematic = false;
                rb.AddForce((dir + Vector3.up * 0.3f) * detachImpulse, ForceMode.Impulse);
            }
        }

        // ---- dents ------------------------------------------------------------------------------

        void DentMesh(MeshFilter mf, Vector3 worldPoint, Vector3 worldDir, float depthMetres, float radius, int level)
        {
            if (mf == null || depthMetres <= 1e-4f) return;
            Mesh mesh = mf.mesh;   // this object's own copy of the mesh
            if (mesh == null || !mesh.isReadable) return;

            Vector3[] original;
            if (!originalVertices.TryGetValue(mf, out original))
            {
                original = mesh.vertices;
                originalVertices[mf] = original;
            }

            Transform t = mf.transform;
            Vector3[] result = new Vector3[original.Length];
            for (int i = 0; i < original.Length; i++)
            {
                Vector3 world = t.TransformPoint(original[i]);
                float dist = Vector3.Distance(world, worldPoint);
                float fall = Mathf.Clamp01(1f - dist / radius);
                fall = fall * fall * (3f - 2f * fall);   // smooth edge of the dent

                float rough = 1f;
                if (level >= 3 && crumple > 0f)
                {
                    float n = Mathf.PerlinNoise(world.x * 7f + world.y * 3f, world.z * 7f);
                    rough = 1f + crumple * (n - 0.5f) * 2f;
                }
                result[i] = original[i] + t.InverseTransformVector(worldDir * (depthMetres * fall * rough));
            }
            mesh.vertices = result;
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
        }

        // ---- paint ------------------------------------------------------------------------------

        void Scorch(Transform part, float damage)
        {
            foreach (Renderer r in part.GetComponentsInChildren<Renderer>())
            {
                Material[] mats = r.materials;   // instances, so shared materials stay untouched
                Color[] originals;
                if (!originalColors.TryGetValue(r, out originals))
                {
                    originals = new Color[mats.Length];
                    for (int i = 0; i < mats.Length; i++) originals[i] = ReadColor(mats[i]);
                    originalColors[r] = originals;
                }
                for (int i = 0; i < mats.Length; i++)
                {
                    WriteColor(mats[i], Color.Lerp(originals[i], scorchColor, Mathf.Clamp01(damage) * maxScorch));
                }
            }
        }

        static Color ReadColor(Material m)
        {
            if (m.HasProperty("_BaseColor")) return m.GetColor("_BaseColor");
            if (m.HasProperty("_Color")) return m.GetColor("_Color");
            return Color.white;
        }

        static void WriteColor(Material m, Color c)
        {
            if (m.HasProperty("_BaseColor")) m.SetColor("_BaseColor", c);
            if (m.HasProperty("_Color")) m.SetColor("_Color", c);
        }

        // ---- glass ------------------------------------------------------------------------------

        void ApplyGlass(PartBinding b, PartDamage pd, Vector3 impactPoint)
        {
            Renderer[] rends = b.part.GetComponentsInChildren<Renderer>();
            if (pd.level >= 3 || pd.detached)
            {
                foreach (Renderer r in rends) r.enabled = false;
                Spawn(glassShardsPrefab, b.part.position, Quaternion.identity);
            }
            else if (pd.level >= 2 && b.crackedMaterial != null)
            {
                foreach (Renderer r in rends) r.sharedMaterial = b.crackedMaterial;
            }
        }

        // ---- parts that fall off ----------------------------------------------------------------

        void Detach(PartBinding b, Vector3 pushDir)
        {
            Transform t = b.part;
            if (t == null || !t.gameObject.activeSelf) return;
            t.SetParent(null, true);

            if (t.GetComponentInChildren<Collider>() == null) t.gameObject.AddComponent<BoxCollider>();
            Rigidbody rb = t.GetComponent<Rigidbody>();
            if (rb == null) rb = t.gameObject.AddComponent<Rigidbody>();
            rb.mass = Mathf.Max(0.5f, b.detachedMassKg);

            Vector3 kick = pushDir + Vector3.up * 0.4f + UnityEngine.Random.insideUnitSphere * 0.3f;
            rb.AddForce(kick.normalized * detachImpulse, ForceMode.Impulse);
            rb.AddTorque(UnityEngine.Random.insideUnitSphere * detachImpulse * 0.4f, ForceMode.Impulse);

            if (detachedLifetime > 0f) Destroy(t.gameObject, detachedLifetime);
            b.part = null;   // it is no longer part of the car
        }

        // ---- effects ----------------------------------------------------------------------------

        void SpawnEffects(ReportEffects fx, Vector3 point, Vector3 dir)
        {
            if (fx == null) return;
            Quaternion outward = Quaternion.LookRotation(-dir, Vector3.up);
            if (fx.sparks) Spawn(sparksPrefab, point, outward);
            if (fx.smoke || fx.fluid_leak) Spawn(smokePrefab, point + Vector3.up * 0.3f, Quaternion.identity);
            if (fx.debris) Spawn(debrisPrefab, point, outward);
            if (fx.dust) Spawn(dustPrefab, point, Quaternion.identity);
            if (fx.airbag_front && airbagFront != null) airbagFront.SetActive(true);
            if (fx.airbag_side && airbagSide != null) airbagSide.SetActive(true);
        }

        void Spawn(GameObject prefab, Vector3 position, Quaternion rotation)
        {
            if (prefab == null) return;
            GameObject go = Instantiate(prefab, position, rotation);
            if (effectLifetime > 0f) Destroy(go, effectLifetime);
        }
    }
}
