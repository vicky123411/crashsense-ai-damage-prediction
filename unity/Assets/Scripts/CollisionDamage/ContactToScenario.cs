using UnityEngine;

namespace CollisionDamage
{
    // Turns a Unity collision into the scenario that the prediction server understands.
    //
    // Coordinates: the Python side uses x forward, y left, z up for the car. Unity uses z forward,
    // x right, y up. The conversion is done here, so the server never sees Unity axes.
    public static class ContactToScenario
    {
        // Size of the car in its own local space (from all renderers below it).
        public static Bounds LocalBounds(Transform car)
        {
            Renderer[] renderers = car.GetComponentsInChildren<Renderer>();
            Bounds result = new Bounds(Vector3.zero, Vector3.zero);
            bool first = true;
            foreach (Renderer r in renderers)
            {
                Bounds w = r.bounds;
                for (int i = 0; i < 8; i++)
                {
                    Vector3 corner = w.center + Vector3.Scale(w.extents, new Vector3(
                        (i & 1) == 0 ? -1f : 1f, (i & 2) == 0 ? -1f : 1f, (i & 4) == 0 ? -1f : 1f));
                    Vector3 local = car.InverseTransformPoint(corner);
                    if (first) { result = new Bounds(local, Vector3.zero); first = false; }
                    else result.Encapsulate(local);
                }
            }
            return result;
        }

        public static CollisionRequest Build(Transform car, Bounds localBounds, Collision collision, VehicleSpec vehicle)
        {
            ContactPoint contact = collision.GetContact(0);
            Vector3 p = car.InverseTransformPoint(contact.point) - localBounds.center;
            Vector3 half = localBounds.extents;
            float nz = p.z / Mathf.Max(half.z, 1e-3f);   // -1 rear ... +1 front
            float nx = p.x / Mathf.Max(half.x, 1e-3f);   // -1 left ... +1 right

            CollisionRequest req = new CollisionRequest();
            Vector3 faceNormalLocal;
            if (Mathf.Abs(nz) >= Mathf.Abs(nx))
            {
                req.face = nz >= 0f ? "front" : "rear";
                req.offset = Mathf.Clamp(-nx, -1f, 1f);          // Python y (left) is Unity -x
                faceNormalLocal = new Vector3(0f, 0f, nz >= 0f ? 1f : -1f);
            }
            else
            {
                req.face = nx >= 0f ? "right" : "left";
                req.offset = Mathf.Clamp(nz, -1f, 1f);           // Python x (forward) is Unity z
                faceNormalLocal = new Vector3(nx >= 0f ? 1f : -1f, 0f, 0f);
            }

            // approach angle between the line of motion and the face normal (the sign does not matter)
            Vector3 relLocal = car.InverseTransformDirection(collision.relativeVelocity);
            float a = relLocal.sqrMagnitude > 1e-6f ? Vector3.Angle(relLocal, faceNormalLocal) : 0f;
            req.angle_deg = Mathf.Min(Mathf.Min(a, 180f - a), 85f);

            req.speed_kmh = collision.relativeVelocity.magnitude * 3.6f;
            req.vehicle = vehicle != null ? vehicle : new VehicleSpec();

            ObstacleTag tag = collision.collider.GetComponentInParent<ObstacleTag>();
            if (tag != null)
            {
                req.obstacle = tag.ToSpec(collision.collider, contact.normal);
            }
            else
            {
                Debug.Log("No ObstacleTag on " + collision.collider.name + ", treating it as a concrete wall.");
                req.obstacle = new ObstacleSpec();
            }
            return req;
        }
    }
}
