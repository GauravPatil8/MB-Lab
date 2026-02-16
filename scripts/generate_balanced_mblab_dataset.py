#!/usr/bin/env python3
"""
Balanced MB-Lab dataset generator.

Run with Blender:
blender -b -P scripts/generate_balanced_mblab_dataset.py -- \
  --repo path/to/the/repo \
  --out  path/to/dataset \
  --samples 5000 \
  --seed 42 \
  --jitter-sigma 0.03 \
  --jitter-fraction 0.2 \
  --use-expressions \
  --expression-prob 1.0
"""

import argparse
import importlib.util
import itertools
import json
import random
import struct
import sys
from pathlib import Path

import bpy


def import_mblab(repo_dir: Path):
    init_py = repo_dir / "__init__.py"
    if not init_py.exists():
        raise FileNotFoundError(f"Cannot find MB-Lab __init__.py in: {repo_dir}")

    mod_name = "mblab_local"
    if mod_name in sys.modules:
        mod = sys.modules[mod_name]
    else:
        spec = importlib.util.spec_from_file_location(
            mod_name,
            str(init_py),
            submodule_search_locations=[str(repo_dir)],
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)

    try:
        mod.register()
    except Exception:
        pass

    return mod


def patch_headless_mblab(mblab):
    # In background mode, addon preferences may be unavailable for this transient
    # module name, and MB-Lab's remove_censors() then crashes.
    try:
        mblab.algorithms.remove_censors = lambda: None
    except Exception:
        pass


def clear_scene():
    # Background mode can fail operator polls; remove datablocks directly.
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    for arm in list(bpy.data.armatures):
        if arm.users == 0:
            bpy.data.armatures.remove(arm)
    for cam in list(bpy.data.cameras):
        if cam.users == 0:
            bpy.data.cameras.remove(cam)
    for light in list(bpy.data.lights):
        if light.users == 0:
            bpy.data.lights.remove(light)
    for action in list(bpy.data.actions):
        if action.users == 0:
            bpy.data.actions.remove(action)
    for img in list(bpy.data.images):
        if img.users == 0:
            bpy.data.images.remove(img)
    for mat in list(bpy.data.materials):
        if mat.users == 0:
            bpy.data.materials.remove(mat)


def remove_subsurf_modifier(obj):
    # Keep pose deformation via armature, but remove subdivision explicitly.
    to_remove = [m for m in obj.modifiers if m.type == "SUBSURF"]
    for mod in to_remove:
        obj.modifiers.remove(mod)


def export_obj_from_evaluated(obj, out_path: Path):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)
    mesh = bpy.data.meshes.new_from_object(obj_eval, depsgraph=depsgraph)
    mesh.transform(obj.matrix_world)

    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# MB-Lab posed mesh\n")
        for v in mesh.vertices:
            f.write(f"v {v.co.x:.7f} {v.co.y:.7f} {v.co.z:.7f}\n")

        # Preserve original polygon topology (tri/quad/ngon).
        for poly in mesh.polygons:
            face = " ".join(str(i + 1) for i in poly.vertices)
            f.write(f"f {face}\n")

    bpy.data.meshes.remove(mesh)


def export_ply_from_evaluated(obj, out_path: Path):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)
    mesh = bpy.data.meshes.new_from_object(obj_eval, depsgraph=depsgraph)
    mesh.transform(obj.matrix_world)
    # Blender API compatibility: calc_normals() is not available in newer versions.
    if hasattr(mesh, "calc_normals"):
        mesh.calc_normals()
    else:
        mesh.update()

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment MB-Lab posed mesh\n"
        f"element vertex {len(mesh.vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float nx\n"
        "property float ny\n"
        "property float nz\n"
        f"element face {len(mesh.polygons)}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")

    with out_path.open("wb") as f:
        f.write(header)

        # Vertex block: x y z nx ny nz (float32), no UV/colors/custom attrs.
        for v in mesh.vertices:
            f.write(
                struct.pack(
                    "<6f",
                    float(v.co.x),
                    float(v.co.y),
                    float(v.co.z),
                    float(v.normal.x),
                    float(v.normal.y),
                    float(v.normal.z),
                )
            )

        # Face block: vertex count (uchar) + indices (int32).
        for poly in mesh.polygons:
            verts = list(poly.vertices)
            if len(verts) > 255:
                raise RuntimeError("PLY face has more than 255 vertices, unsupported by uchar list count")
            f.write(struct.pack("<B", len(verts)))
            f.write(struct.pack("<" + ("i" * len(verts)), *verts))

    bpy.data.meshes.remove(mesh)


def gender_pose_folder(char_id: str) -> str:
    return "female_poses" if char_id.startswith("f_") else "male_poses"


def is_anime_character(char_id: str) -> bool:
    return "_an" in char_id


def clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def apply_small_morph_jitter(humanoid, rng: random.Random, sigma: float, fraction: float):
    keys = [k for k in humanoid.character_data.keys() if not k.startswith("Expressions_")]
    if not keys:
        return 0

    n = max(1, int(round(len(keys) * fraction)))
    n = min(n, len(keys))
    chosen = rng.sample(keys, n)

    for key in chosen:
        base = float(humanoid.character_data[key])
        humanoid.character_data[key] = clip01(base + rng.gauss(0.0, sigma))

    humanoid.update_character(mode="update_all")
    return n


def list_expression_files(char_id: str, human_expr_dir: Path, anime_expr_dir: Path):
    expr_dir = anime_expr_dir if is_anime_character(char_id) else human_expr_dir
    if not expr_dir.exists():
        return []
    return sorted(expr_dir.glob("*.json"))


def apply_expression_preset(mblab, humanoid, expr_path: Path, expr_scale: float = 1.0):
    creator = mblab.expressionscreator.ExpressionsCreator()
    creator.humanoid = humanoid
    creator.set_lab_version(mblab.bl_info["version"])
    creator.load_face_expression(str(expr_path), reset_unassigned=True)

    # Optional scaling around neutral value 0.5 for stronger/weaker expressions.
    if abs(expr_scale - 1.0) > 1e-6:
        for key, value in humanoid.character_data.items():
            if key.startswith("Expressions_"):
                humanoid.character_data[key] = clip01(0.5 + (float(value) - 0.5) * expr_scale)
        humanoid.update_character(mode="update_all")


def parse_args(argv):
    ap = argparse.ArgumentParser(description="Generate a balanced MB-Lab mesh dataset")
    ap.add_argument("--repo", required=True, help="Path to MB-Lab repo")
    ap.add_argument("--out", required=True, help="Output dataset directory")
    ap.add_argument("--samples", type=int, default=10000, help="Total meshes to generate")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jitter-sigma", type=float, default=0.03,
                    help="Stddev for per-slider perturbation after preset load")
    ap.add_argument("--jitter-fraction", type=float, default=0.20,
                    help="Fraction of sliders to perturb per sample")
    ap.add_argument("--use-expressions", action="store_true",
                    help="Apply combined facial expression presets")
    ap.add_argument("--expression-prob", type=float, default=1.0,
                    help="Probability of applying a face expression on each sample")
    ap.add_argument("--human-expr-dir", default="data/expressions_comb/human_expressions",
                    help="Path relative to --repo (or absolute path) for human expression presets")
    ap.add_argument("--anime-expr-dir", default="data/expressions_comb/anime_expressions",
                    help="Path relative to --repo (or absolute path) for anime expression presets")
    ap.add_argument("--expr-scale-min", type=float, default=0.9,
                    help="Minimum random expression intensity scale")
    ap.add_argument("--expr-scale-max", type=float, default=1.1,
                    help="Maximum random expression intensity scale")
    ap.add_argument("--export", choices=["obj", "ply"], default="ply")
    return ap.parse_args(argv)


def main():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    args = parse_args(argv)

    if args.samples < 1:
        raise ValueError("--samples must be >= 1")
    if not (0.0 < args.jitter_fraction <= 1.0):
        raise ValueError("--jitter-fraction must be in (0, 1]")
    if args.jitter_sigma <= 0.0:
        raise ValueError("--jitter-sigma must be > 0")
    if not (0.0 <= args.expression_prob <= 1.0):
        raise ValueError("--expression-prob must be in [0, 1]")
    if args.expr_scale_min <= 0.0 or args.expr_scale_max <= 0.0:
        raise ValueError("--expr-scale-min and --expr-scale-max must be > 0")
    if args.expr_scale_min > args.expr_scale_max:
        raise ValueError("--expr-scale-min must be <= --expr-scale-max")

    repo_dir = Path(args.repo).resolve()
    out_dir = Path(args.out).resolve()
    mesh_dir = out_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = out_dir / "metadata.jsonl"

    rng = random.Random(args.seed)

    cfg_path = repo_dir / "data" / "characters_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    characters = list(cfg["character_list"])
    human_expr_dir = Path(args.human_expr_dir)
    anime_expr_dir = Path(args.anime_expr_dir)
    if not human_expr_dir.is_absolute():
        human_expr_dir = repo_dir / human_expr_dir
    if not anime_expr_dir.is_absolute():
        anime_expr_dir = repo_dir / anime_expr_dir

    # Balanced split: each character gets either floor(N/C) or ceil(N/C) samples.
    base = args.samples // len(characters)
    rem = args.samples % len(characters)
    char_targets = {cid: base for cid in characters}
    for cid in characters[:rem]:
        char_targets[cid] += 1

    # Pre-compute preset+pose combinations per character.
    combo_map = {}
    expr_map = {}
    for cid in characters:
        preset_folder = cfg[cid]["presets_folder"]
        preset_dir = repo_dir / "data" / "presets" / preset_folder
        pose_dir = repo_dir / "data" / "poses" / gender_pose_folder(cid)

        presets = sorted(preset_dir.glob("*.json"))
        poses = sorted(pose_dir.glob("*.json"))
        if not presets:
            raise RuntimeError(f"No presets found for {cid} in {preset_dir}")
        if not poses:
            raise RuntimeError(f"No poses found for {cid} in {pose_dir}")

        combos = list(itertools.product(presets, poses))
        rng.shuffle(combos)
        combo_map[cid] = combos
        expr_map[cid] = list_expression_files(cid, human_expr_dir, anime_expr_dir)

    mblab = import_mblab(repo_dir)
    patch_headless_mblab(mblab)

    scn = bpy.context.scene
    if hasattr(scn, "mbcrea_root_data"):
        scn.mbcrea_root_data = "data"

    scn.mblab_use_ik = False
    scn.mblab_use_muscle = False
    scn.mblab_use_cycles = False
    scn.mblab_use_eevee = False

    global_idx = 0
    with metadata_path.open("w", encoding="utf-8", newline="\n") as mf:
        for cid in characters:
            target = char_targets[cid]
            combos = combo_map[cid]
            combo_i = 0

            for local_i in range(target):
                clear_scene()

                scn.mblab_character_name = cid
                bpy.ops.mbast.init_character()

                humanoid = mblab.mblab_humanoid
                armature = humanoid.get_armature()
                body = humanoid.get_object()
                remove_subsurf_modifier(body)

                preset_path, pose_path = combos[combo_i % len(combos)]
                combo_i += 1
                if combo_i % len(combos) == 0:
                    rng.shuffle(combos)

                humanoid.load_character(str(preset_path), mix=False)
                changed_sliders = apply_small_morph_jitter(
                    humanoid,
                    rng,
                    sigma=args.jitter_sigma,
                    fraction=args.jitter_fraction,
                )
                expression_path = None
                expression_scale = None
                if args.use_expressions and expr_map[cid] and rng.random() <= args.expression_prob:
                    expression_path = rng.choice(expr_map[cid])
                    expression_scale = rng.uniform(args.expr_scale_min, args.expr_scale_max)
                    apply_expression_preset(
                        mblab,
                        humanoid,
                        expression_path,
                        expr_scale=expression_scale,
                    )

                mblab.mblab_retarget.load_pose(
                    str(pose_path),
                    target_armature=armature,
                    use_retarget=True,
                )

                mesh_ext = args.export
                mesh_name = f"{global_idx:07d}_{cid}.{mesh_ext}"
                mesh_path = mesh_dir / mesh_name
                if args.export == "ply":
                    export_ply_from_evaluated(body, mesh_path)
                else:
                    export_obj_from_evaluated(body, mesh_path)

                rec = {
                    "index": global_idx,
                    "character_id": cid,
                    "character_sample_index": local_i,
                    "mesh": str(mesh_path),
                    "preset": str(preset_path),
                    "pose": str(pose_path),
                    "expression": str(expression_path) if expression_path else None,
                    "expression_scale": expression_scale,
                    "morph_jitter_sigma": args.jitter_sigma,
                    "morph_jitter_fraction": args.jitter_fraction,
                    "morph_jitter_slider_count": changed_sliders,
                    "seed": args.seed,
                }
                mf.write(json.dumps(rec) + "\n")

                global_idx += 1
                if global_idx % 100 == 0:
                    print(f"Generated {global_idx}/{args.samples}")

    print(f"Done. Generated {global_idx} meshes.")
    print(f"Meshes: {mesh_dir}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
