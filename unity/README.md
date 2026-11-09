# Unity side

Scripts that ask the Python server for a damage report and show it on a car and an obstacle.
They only change how things look. They do not change how Unity's physics resolves the crash.

> **Status:** the C# files were checked for syntax and compiled (C# 9, no errors) against stand-in Unity
> types, which catches typos and wrong names between the scripts. They have not been compiled with the real
> Unity assemblies or run inside Unity, so expect to fix small things (names, versions, materials) the first
> time you import them. They were written for Unity 2021.3 or newer.

## Files (`Assets/Scripts/CollisionDamage/`)

| file | job |
|---|---|
| `CollisionRequest.cs` | the scenario that is sent (speed, side, offset, angle, car, obstacle) |
| `DamageReport.cs` | the reply (per-part damage, obstacle damage, summary, effect flags) |
| `PredictionClient.cs` | `UnityWebRequest` call to the local Python server |
| `ObstacleTag.cs` | put on anything the car can hit: preset name, material, fixed or free |
| `ContactToScenario.cs` | turns a Unity `Collision` into a request (picks the side, offset and angle) |
| `CollisionDamageApplier.cs` | dents meshes, darkens paint, cracks glass, tears parts off, spawns effects |
| `CarCrashController.cs` | glue on the car: `OnCollisionEnter` → server → applier |

## Set-up

1. Start the server in the repo folder (the physics layer works without a model):

   ```
   python -m crashdamage serve --model models/gnn_demo.pt --port 8765
   ```

   Leave out `--model` to use the physics layer only. In Unity's Player Settings set
   *Other Settings → Allow downloads over HTTP* to *Always allowed*, because the server is plain HTTP on localhost.

2. Copy `Assets/Scripts/CollisionDamage` into your project.

3. On the car (the object with the `Rigidbody` and its colliders) add `PredictionClient`,
   `CollisionDamageApplier` and `CarCrashController`. Drag the `PredictionClient` into the controller.

4. **Parts.** Name child objects like the part ids, for example `door_l`, `bonnet`, `wheel_fl`, `windscreen`
   (the full list is `GET /parts`), then right-click the `CollisionDamageApplier` and choose
   *Bind parts by child name*. Parts you do not have are simply skipped. For glass, tick `isGlass` and
   optionally give a cracked material.

5. **Meshes.** Dents move vertices, so the car meshes need *Read/Write Enabled* in their import settings.
   Keep part objects side by side (not nested inside each other), otherwise a dent on a parent also moves
   its children.

6. **Obstacles.** Add `ObstacleTag` to the wall, pole, fence, ... Pick the preset that fits. Tick
   `fixedInPlace` for things that cannot move; for pushable objects untick it and give them a `Rigidbody`
   (its mass is sent).

7. **Effects.** Drop particle prefabs into the applier (sparks, smoke, glass shards, debris, dust). Assign the
   airbag objects if you have them. All slots are optional.

8. Try it without crashing anything: press Play, then right-click `CarCrashController` →
   *Test: 50 km/h into the front, concrete wall* (the Python server must be running).

## Coordinates

The Python car uses x forward, y left, z up. Unity uses z forward, x right, y up. `ContactToScenario`
does the conversion. The side that was hit is taken from where the contact point lies on the car's
bounding box; `offset` is the position along that side (−1 to 1); `angle` is the angle between the
relative velocity and the side's normal.

## Changing the car

Per-part materials and thicknesses go into `CarCrashController.vehicle`, for example the material of
`door_l` or `beam_front`. The material names are listed in `docs/method.md`.
