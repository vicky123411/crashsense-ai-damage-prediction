using UnityEngine;

namespace CollisionDamage
{
    // Put this on anything the car can hit. It says what the object is made of and how it behaves,
    // so the prediction can treat a concrete wall, a steel pole and a wooden fence differently.
    public class ObstacleTag : MonoBehaviour
    {
        [Tooltip("concrete_wall, concrete_block, steel_pole, steel_edge, wood_fence, wood_post, foam_barrier, steel_crate")]
        public string preset = "concrete_wall";

        [Tooltip("Optional override: concrete, steel, wood or foam. Empty keeps the preset.")]
        public string material = "";

        [Tooltip("Take width, height and ground clearance from the collider instead of the preset.")]
        public bool measureFromCollider = true;

        [Tooltip("True: bolted to the ground. False: it can be pushed, its mass comes from the Rigidbody.")]
        public bool fixedInPlace = true;

        [Tooltip("Height of the ground under the obstacle (world Y).")]
        public float groundY = 0f;

        public ObstacleSpec ToSpec(Collider hit, Vector3 contactNormalWorld)
        {
            ObstacleSpec spec = new ObstacleSpec();
            spec.name = preset;
            spec.material = material;
            spec.fixed_in_place = fixedInPlace;

            if (!fixedInPlace)
            {
                Rigidbody rb = hit != null ? hit.attachedRigidbody : null;
                if (rb != null) spec.mass_kg = rb.mass;
            }

            if (measureFromCollider && hit != null)
            {
                Bounds b = hit.bounds;
                Vector3 n = new Vector3(contactNormalWorld.x, 0f, contactNormalWorld.z);
                if (n.sqrMagnitude < 1e-6f) n = Vector3.forward;
                n.Normalize();
                // bounds are axis aligned, so project the size on the contact direction and on the way across it
                spec.depth = Mathf.Abs(n.x) * b.size.x + Mathf.Abs(n.z) * b.size.z;
                spec.width = Mathf.Abs(n.z) * b.size.x + Mathf.Abs(n.x) * b.size.z;
                spec.height = b.size.y;
                spec.z_base = Mathf.Max(0f, b.min.y - groundY);
            }
            return spec;
        }
    }
}
