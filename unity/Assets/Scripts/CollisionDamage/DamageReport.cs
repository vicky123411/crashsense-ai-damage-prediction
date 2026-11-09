using System;

namespace CollisionDamage
{
    // These classes mirror the JSON that the Python server returns. The field names are the
    // JSON keys, so they must not be renamed. Fields that are missing in the JSON stay at 0 / null.

    [Serializable]
    public class PartDamage
    {
        public string id;          // part id, e.g. "door_l" (see GET /parts)
        public float damage;       // 0..1
        public int level;          // 0 intact, 1 scratched, 2 dented, 3 crushed, 4 destroyed
        public string level_name;
        public bool detached;      // the part tears off its mounts
        public bool direct;        // the part lies in the contact patch
    }

    [Serializable]
    public class ObstacleDamage
    {
        public string name;
        public string material;
        public float damage;
        public int level;
        public string level_name;
        public float deformation_m;
        public bool broken_through;   // fences and posts that snap
        public float speed_after_ms;  // speed of a free obstacle after the hit
    }

    [Serializable]
    public class ReportSummary
    {
        public float crush_depth_m;
        public float cabin_intrusion_m;
        public float peak_force_kn;
        public float energy_total_kj;
        public float energy_car_kj;          // only filled by the physics engine
        public float energy_obstacle_kj;     // only filled by the physics engine
        public float energy_unabsorbed_kj;   // only filled by the physics engine
        public bool went_through_car;
        public float delta_v_ms;
    }

    [Serializable]
    public class ReportEffects
    {
        public bool sparks;
        public bool glass_shards;
        public bool smoke;
        public bool fluid_leak;
        public bool debris;
        public bool dust;
        public bool airbag_front;
        public bool airbag_side;
    }

    [Serializable]
    public class ReportImpact
    {
        public string face;
        public float offset;
        public float angle_deg;
        public float speed_ms;
        public float speed_kmh;
    }

    [Serializable]
    public class DamageReport
    {
        public PartDamage[] parts;
        public ObstacleDamage obstacle;
        public ReportSummary summary;
        public ReportEffects effects;
        public ReportImpact impact;
        public string engine;       // "physics" or "model"
    }
}
