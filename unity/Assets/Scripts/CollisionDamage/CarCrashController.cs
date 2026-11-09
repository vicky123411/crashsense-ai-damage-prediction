using UnityEngine;

namespace CollisionDamage
{
    // Put this on the car (the object with the Rigidbody and the colliders).
    // When the car hits something it asks the Python server for a damage report and shows it.
    [RequireComponent(typeof(CollisionDamageApplier))]
    public class CarCrashController : MonoBehaviour
    {
        public PredictionClient client;
        public VehicleSpec vehicle = new VehicleSpec();
        [Tooltip("Impacts slower than this are ignored.")]
        public float minSpeedKmh = 8f;
        [Tooltip("Only the first big impact is shown. Turn off to react to every hit.")]
        public bool onlyFirstImpact = true;

        CollisionDamageApplier applier;
        Bounds localBounds;
        bool busy;
        bool handled;

        void Awake()
        {
            applier = GetComponent<CollisionDamageApplier>();
            if (client == null) client = GetComponent<PredictionClient>();
            localBounds = ContactToScenario.LocalBounds(transform);
        }

        void OnCollisionEnter(Collision collision)
        {
            if (busy || (onlyFirstImpact && handled) || client == null) return;
            if (collision.relativeVelocity.magnitude * 3.6f < minSpeedKmh) return;

            CollisionRequest request = ContactToScenario.Build(transform, localBounds, collision, vehicle);
            Vector3 point = collision.GetContact(0).point;
            Vector3 carCentre = transform.TransformPoint(localBounds.center);
            Vector3 pushIntoCar = (carCentre - point).normalized;
            Transform obstacle = collision.collider.transform;

            busy = true;
            StartCoroutine(client.Predict(request,
                report =>
                {
                    busy = false;
                    handled = true;
                    applier.Apply(report, point, pushIntoCar);
                    applier.ApplyToObstacle(obstacle, report, point, -pushIntoCar);
                },
                error =>
                {
                    busy = false;
                    Debug.LogWarning("Damage prediction failed: " + error);
                }));
        }

        // Handy in the editor: right click the component and pick one of these.
        [ContextMenu("Test: 50 km/h into the front, concrete wall")]
        void TestFront()
        {
            SendTest("front", 0f, 50f);
        }

        [ContextMenu("Test: 30 km/h into the left side, steel pole")]
        void TestSidePole()
        {
            SendTest("left", 0f, 30f);
        }

        void SendTest(string face, float offset, float speedKmh)
        {
            if (client == null)
            {
                Debug.LogWarning("Assign a PredictionClient first.");
                return;
            }
            if (applier == null) applier = GetComponent<CollisionDamageApplier>();
            localBounds = ContactToScenario.LocalBounds(transform);

            CollisionRequest request = new CollisionRequest();
            request.speed_kmh = speedKmh;
            request.face = face;
            request.offset = offset;
            request.vehicle = vehicle;
            if (face == "left") request.obstacle.name = "steel_pole";

            Vector3 local = face == "front" ? new Vector3(0f, 0f, localBounds.extents.z)
                          : face == "rear" ? new Vector3(0f, 0f, -localBounds.extents.z)
                          : face == "left" ? new Vector3(-localBounds.extents.x, 0f, 0f)
                          : new Vector3(localBounds.extents.x, 0f, 0f);
            Vector3 point = transform.TransformPoint(localBounds.center + local);
            Vector3 push = (transform.TransformPoint(localBounds.center) - point).normalized;

            StartCoroutine(client.Predict(request,
                report => applier.Apply(report, point, push),
                error => Debug.LogWarning("Damage prediction failed: " + error)));
        }
    }
}
