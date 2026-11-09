using System;

namespace CollisionDamage
{
    // JsonUtility cannot leave a field out and cannot write dictionaries. The Python side therefore
    // reads 0 and "" as "not given, keep the default", and takes lists of {part, value} pairs.

    [Serializable]
    public class PartMaterial
    {
        public string part;       // e.g. "door_l"
        public string material;   // mild_steel, high_strength_steel, ultra_high_strength_steel, aluminium, plastic, glass
    }

    [Serializable]
    public class PartThickness
    {
        public string part;
        public float mm;
    }

    [Serializable]
    public class VehicleSpec
    {
        public float mass_kg = 1100f;
        public float thickness_scale = 1f;                            // multiplies the default thickness of every part
        public PartMaterial[] materials = new PartMaterial[0];        // per-part material overrides
        public PartThickness[] thickness_mm = new PartThickness[0];   // per-part thickness overrides
    }

    [Serializable]
    public class ObstacleSpec
    {
        public string name = "concrete_wall";   // preset: concrete_wall, concrete_block, steel_pole, steel_edge,
                                                // wood_fence, wood_post, foam_barrier, steel_crate
        public string material = "";            // concrete, steel, wood, foam (empty = preset)
        public string shape = "";               // flat, round, sharp (empty = preset)
        public string form = "";                // solid, thin_wall, slender, foam (empty = preset)
        public float width;                     // metres across the car, 0 = preset
        public float depth;
        public float height;
        public float wall_mm;
        public float perimeter;
        public float mass_kg;                   // 0 = preset
        public float z_base;                    // height of the lowest point above the ground
        public bool fixed_in_place = true;      // true: bolted down, cannot be pushed
    }

    [Serializable]
    public class CollisionRequest
    {
        public float speed_kmh;
        public string face = "front";           // front, rear, left, right (of the car)
        public float offset;                    // -1..1 along that side, 0 = middle
        public float angle_deg;                 // approach angle from the face normal
        public VehicleSpec vehicle = new VehicleSpec();
        public ObstacleSpec obstacle = new ObstacleSpec();
    }
}
