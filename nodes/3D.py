import copy
import itertools
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pynanoinstantmeshes as PyNIM
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh as trimesh_lib
from scipy.spatial import KDTree
from trimesh.constants import tol
from trimesh.grouping import group_rows, merge_vertices

categoryFolder = "Custom pack/3D"

class CP_MeshToTrimesh:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
            },
        }
    
    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "convert"
    CATEGORY = categoryFolder
    DESCRIPTION = "Converts a MESH object (with .v and .f attributes) to a trimesh.Trimesh object."

    def convert(self, mesh):
        if hasattr(mesh, 'v') and hasattr(mesh, 'f'):
            v = mesh.v
            f = mesh.f
        elif hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
            v = mesh.vertices
            f = mesh.faces
        else:
            raise AttributeError("Mesh object must have .v/.f or .vertices/.faces attributes")

        if torch.is_tensor(v):
            v_np = v.detach().cpu().numpy()
        else:
            v_np = np.array(v)
        
        if torch.is_tensor(f):
            f_np = f.detach().cpu().numpy()
        else:
            f_np = np.array(f)

        if v_np.ndim == 3 and v_np.shape[0] == 1:
            v_np = v_np[0]
        if f_np.ndim == 3 and f_np.shape[0] == 1:
            f_np = f_np[0]

        trimesh_obj = trimesh_lib.Trimesh(vertices=v_np, faces=f_np, process=False)

        return (trimesh_obj,)

class CP_TrimeshWeldVertex:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "threshold": ("FLOAT", {
                    "default": 0.00000001,
                    "min": 0.0,
                    "max": 1000.0,
                    "step": 0.00000001,
                    "precision": 8
                }),
                "scale_factor": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 10000.0,
                    "step": 0.01,
                    "precision": 2,
                    "description": "Temporarily scale mesh to improve weld stability"
                }),
                "merge_tex": ("INT", {"default": -1, "min": -1, "max": 1}),
                "merge_norm": ("INT", {"default": -1, "min": -1, "max": 1}),
                "digits_vertex": ("INT", {"default": -1, "min": -1, "max": 100}),
                "digits_norm": ("INT", {"default": -1, "min": -1, "max": 100}),
                "digits_uv": ("INT", {"default": -1, "min": -1, "max": 100}),
            },
        }

    RETURN_TYPES = ("TRIMESH", "INT", "INT", "STRING")
    RETURN_NAMES = ("trimesh", "vertices_before", "vertices_after", "message")
    FUNCTION = "weld"
    CATEGORY = categoryFolder
    DESCRIPTION = "Welds vertices within threshold using scaling for numerical stability."

    def weld(self, trimesh, threshold, scale_factor, merge_tex, merge_norm, digits_vertex, digits_norm, digits_uv):
        vertices_before = len(trimesh.vertices)
        new_mesh = trimesh.copy()

        if threshold <= 0 or vertices_before == 0:
            return (new_mesh, vertices_before, vertices_before)
        scale = scale_factor
        new_mesh.vertices *= scale

        original_merge_tol = tol.merge
        try:
            tol.merge = threshold * scale

            kwargs = {}
            if merge_tex >= 0:
                kwargs['merge_tex'] = bool(merge_tex)
            if merge_norm >= 0:
                kwargs['merge_norm'] = bool(merge_norm)
            if digits_vertex >= 0:
                kwargs['digits_vertex'] = digits_vertex
            if digits_norm >= 0:
                kwargs['digits_norm'] = digits_norm
            if digits_uv >= 0:
                kwargs['digits_uv'] = digits_uv

            new_mesh.merge_vertices(**kwargs)
        finally:
            tol.merge = original_merge_tol
        new_mesh.vertices /= scale
        if hasattr(new_mesh, 'remove_degenerate_faces'):
            new_mesh.remove_degenerate_faces()

        vertices_after = len(new_mesh.vertices)
        message = f"Vertex before = {vertices_before}\nVertex after = {vertices_after}"
        
        return (new_mesh, vertices_before, vertices_after, message)

class CP_STLCheck:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("open_edges", "double_faces", "non_manifold_edges", "degenerate_faces", "total_errors", "message")
    FUNCTION = "check_mesh"
    CATEGORY = categoryFolder
    DESCRIPTION = (
        "Performs an STL check on the mesh, similar to the one in 3ds Max.\n"
        "Returns the counts of: Open Edges, Double Faces, Non-Manifold Edges, Degenerate Faces, and Total Errors."
    )

    def check_mesh(self, trimesh):
        mesh = trimesh.copy()
        
        open_edges_count = 0
        double_faces_count = 0
        non_manifold_edges_count = 0
        degenerate_faces_count = 0

        # --- 1. Open edges
        edges_sorted = np.sort(mesh.edges, axis=1)
        unique_edges = group_rows(edges_sorted, require_count=1)
        open_edges_count = len(unique_edges)

        # --- 2. Double Faces
        sorted_faces = np.sort(mesh.faces, axis=1)
        double_faces_indices = group_rows(sorted_faces, require_count=2)
        double_faces_count = len(double_faces_indices)

        # --- 3. Multiple Edge
        multi_edges = group_rows(edges_sorted, require_count=3)
        non_manifold_edges_count = len(multi_edges)

        # --- 4. Spike
        epsilon = 1e-8
        for face in mesh.faces:
            v0, v1, v2 = mesh.vertices[face]
            edge1 = v1 - v0
            edge2 = v2 - v0
            area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
            if area < epsilon:
                degenerate_faces_count += 1

        # --- 5. Totals
        total_errors = (open_edges_count + double_faces_count + 
                        non_manifold_edges_count + degenerate_faces_count)

        msg = f"Open Edge = {open_edges_count}\nDouble Face = {double_faces_count}\nSpike = {non_manifold_edges_count}\nMultiple Edge = {degenerate_faces_count}\nTotal = {total_errors}"

        return (open_edges_count, double_faces_count, non_manifold_edges_count, 
                degenerate_faces_count, total_errors, msg)

class CP_QuadRemesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "face_count": ("INT", {
                    "default": 10000,
                    "min": 100,
                    "max": 1000000,
                    "step": 100,
                    "description": "Target number of faces (quads + triangles)"
                }),
                "smooth_iter": ("INT", {
                    "default": 5,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "description": "Smoothing iterations"
                }),
                "align_to_boundaries": ("BOOLEAN", {
                    "default": True,
                    "description": "Align quads to boundaries"
                }),
                "crease_angle": ("FLOAT", {
                    "default": 45.0,
                    "min": 0.0,
                    "max": 180.0,
                    "step": 1.0,
                    "description": "Sharp feature angle (degrees). Lower = preserve sharper edges"
                }),
                "smooth_normals": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("TRIMESH", "INT", "INT", "STRING")
    RETURN_NAMES = ("trimesh", "face_count_before", "face_count_after", "message")
    FUNCTION = "quad_remesh"
    CATEGORY = categoryFolder
    DESCRIPTION = "Quad remeshing using Instant Meshes (4-rosy/4-posy field)."

    def quad_remesh(self, trimesh, face_count, smooth_iter, align_to_boundaries, crease_angle, smooth_normals):
        face_count_before = len(trimesh.faces)
        target_vertex_count = max(4, int(face_count * 0.9))

        try:
            new_verts, new_faces = PyNIM.remesh(
                np.array(trimesh.vertices, dtype=np.float32),
                np.array(trimesh.faces, dtype=np.uint32),
                target_vertex_count,
                rosy=4,
                posy=4,
                align_to_boundaries=align_to_boundaries,
                smooth_iter=smooth_iter,
                creaseAngle=crease_angle,
                extrinsic=True,
                deterministic=False
            )
        except Exception as e:
            raise RuntimeError(f"Quad remeshing failed: {e}")

        if isinstance(new_faces, np.ndarray) and new_faces.ndim == 2 and new_faces.shape[1] == 4:
            mesh = trimesh_lib.Trimesh(vertices=new_verts, faces=new_faces, process=False)
        else:
            quad_faces = [f for f in new_faces if len(f) == 4] if isinstance(new_faces, (list, tuple)) else []
            if quad_faces:
                mesh = trimesh_lib.Trimesh(vertices=new_verts, faces=quad_faces, process=False)
            else:
                mesh = trimesh_lib.Trimesh(vertices=new_verts, faces=new_faces, process=False)

        if hasattr(mesh, 'nondegenerate_faces'):
            good_faces = mesh.nondegenerate_faces()
            if isinstance(good_faces, np.ndarray) and good_faces.dtype == bool:
                mesh.update_faces(good_faces)
            elif isinstance(good_faces, np.ndarray) and good_faces.dtype in (np.int32, np.int64):
                mesh.faces = mesh.faces[good_faces]
        else:
            face_areas = mesh.area_faces
            mesh.update_faces(face_areas > 1e-8)

        if smooth_normals:              
            mesh.vertex_normals = trimesh_lib.smoothing.get_vertices_normals(mesh)

        face_count_after = len(mesh.faces)
        message = f"Faces before = {face_count_before}\nFaces after = {face_count_after}"
        return (mesh, face_count_before, face_count_after, message)

class CP_FixMeshIssues:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh_obj": ("TRIMESH",),
                "fix_double_faces": ("BOOLEAN", {"default": True}),
                "fix_spikes": ("BOOLEAN", {"default": True}),
                "fix_multiple_edges": ("BOOLEAN", {"default": True}),
                "fix_open_edges": ("BOOLEAN", {"default": True}),
                "max_hole_edges": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1, "description": "Max count edge holes for fill (0 — no limit)"}),
            },
        }

    RETURN_TYPES = ("TRIMESH", "INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("trimesh", "open_edges_before", "double_faces_before", "spikes_before", "multiple_edges_before", "report")
    FUNCTION = "fix_mesh"
    CATEGORY = categoryFolder
    DESCRIPTION = "Model fixer."

    def fix_mesh(self, trimesh_obj, fix_double_faces, fix_spikes, fix_multiple_edges, fix_open_edges, max_hole_edges):
        mesh = trimesh_obj.copy()

        open_edges_before = self._count_open_edges(mesh)
        double_faces_before = self._count_duplicate_faces(mesh)
        spikes_before = self._count_degenerate_faces(mesh)
        multiple_edges_before = self._count_non_manifold_edges(mesh)

        if fix_double_faces:
            mesh = self._remove_duplicate_faces(mesh)
        if fix_spikes:
            mesh = self._remove_degenerate_faces(mesh)
        if fix_multiple_edges:
            mesh = self._fix_non_manifold_edges(mesh)
        if fix_open_edges:
            mesh = self._close_holes(mesh, max_hole_edges)

        open_edges_after = self._count_open_edges(mesh)
        double_faces_after = self._count_duplicate_faces(mesh)
        spikes_after = self._count_degenerate_faces(mesh)
        multiple_edges_after = self._count_non_manifold_edges(mesh)

        report = (
            f"Open Edge before = {open_edges_before}, after = {open_edges_after}\n"
            f"Double Face before = {double_faces_before}, after = {double_faces_after}\n"
            f"Spike before = {spikes_before}, after = {spikes_after}\n"
            f"Multiple Edge before = {multiple_edges_before}, after = {multiple_edges_after}"
        )

        return (mesh, open_edges_before, double_faces_before, spikes_before, multiple_edges_before, report)

    # ------------------- Countrer functions -------------------
    def _count_open_edges(self, mesh):
        if mesh.is_empty:
            return 0
        edges = mesh.edges_unique
        adj_edges = mesh.face_adjacency_edges
        is_boundary = ~np.isin(edges, adj_edges).all(axis=1)
        return np.sum(is_boundary)

    def _count_duplicate_faces(self, mesh):
        if mesh.is_empty:
            return 0
        faces_sorted = np.sort(mesh.faces, axis=1)
        _, _, counts = np.unique(faces_sorted, axis=0, return_inverse=True, return_counts=True)
        return np.sum(counts > 1)

    def _count_degenerate_faces(self, mesh):
        if mesh.is_empty:
            return 0
        return np.sum(mesh.area_faces < 1e-8)

    def _count_non_manifold_edges(self, mesh):
        if mesh.is_empty:
            return 0
        edges, _, counts = np.unique(mesh.edges_sorted, axis=0, return_inverse=True, return_counts=True)
        return np.sum(counts > 2)

    # ------------------- Fix functions -------------------
    def _remove_duplicate_faces(self, mesh):
        if mesh.is_empty:
            return mesh
        faces_sorted = np.sort(mesh.faces, axis=1)
        _, unique_idx = np.unique(faces_sorted, axis=0, return_index=True)
        mesh.faces = mesh.faces[unique_idx]
        mesh.remove_unreferenced_vertices()
        return mesh

    def _remove_degenerate_faces(self, mesh):
        if mesh.is_empty:
            return mesh
        good = mesh.area_faces > 1e-8
        if np.any(~good):
            mesh.update_faces(good)
            mesh.remove_unreferenced_vertices()
        return mesh

    def _fix_non_manifold_edges(self, mesh):
        if mesh.is_empty:
            return mesh

        edges, _, counts = np.unique(mesh.edges_sorted, axis=0, return_inverse=True, return_counts=True)
        bad_edge_mask = counts > 2
        if not np.any(bad_edge_mask):
            return mesh

        bad_edges = edges[bad_edge_mask]
        bad_set = set(tuple(e) for e in bad_edges)
        faces_to_remove = []
        for i, face in enumerate(mesh.faces):
            e0 = tuple(sorted((face[0], face[1])))
            e1 = tuple(sorted((face[1], face[2])))
            e2 = tuple(sorted((face[2], face[0])))
            if e0 in bad_set or e1 in bad_set or e2 in bad_set:
                faces_to_remove.append(i)

        if faces_to_remove:
            keep = np.ones(len(mesh.faces), dtype=bool)
            keep[faces_to_remove] = False
            mesh.update_faces(keep)
            mesh.remove_unreferenced_vertices()

        return mesh

    def _close_holes(self, mesh, max_hole_edges):
        if mesh.is_empty:
            return mesh
        if max_hole_edges > 0:
            mesh.fill_holes(max_hole_edges)
        else:
            mesh.fill_holes()
        return mesh

class CP_SaveTrimesh:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"trimesh": ("TRIMESH",), "filename": ("STRING", {"default": "output"})}}

    RETURN_TYPES = ()
    FUNCTION = "save"
    CATEGORY = categoryFolder
    OUTPUT_NODE = True

    def save(self, trimesh, filename):
        trimesh.export(f"{filename}.obj")
        return {}

class SparseResBlockS2C3d(nn.Module):
    """TRELLIS.2 sparse spatial-to-channel residual downsample block."""

    def __init__(self, channels: int, out_channels: Optional[int] = None):
        super().__init__()
        from comfy.ldm.trellis2.vae import SparseConv3d, SparseSpatial2Channel
        import comfy.ops

        self.channels = channels
        self.out_channels = out_channels or channels
        self.norm1 = comfy.ops.disable_weight_init.LayerNorm(
            channels, elementwise_affine=True, eps=1e-6
        )
        self.norm2 = comfy.ops.disable_weight_init.LayerNorm(
            self.out_channels, elementwise_affine=False, eps=1e-6
        )
        self.conv1 = SparseConv3d(channels, self.out_channels // 8, 3)
        self.conv2 = SparseConv3d(self.out_channels, self.out_channels, 3)
        self.updown = SparseSpatial2Channel(2)

    def forward(self, x):
        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats, inplace=True))
        h = self.conv1(h)
        h = self.updown(h)
        skip = self.updown(x)
        h = h.replace(self.norm2(h.feats))
        h = h.replace(F.silu(h.feats, inplace=True))
        h = self.conv2(h)
        group = self.channels * 8 // self.out_channels
        skip_feats = skip.feats.reshape(skip.feats.shape[0], self.out_channels, group).mean(dim=-1)
        h.feats.add_(skip_feats.to(h.feats.dtype))
        return h


def _subdivision_from_downsample(coarse):
    from comfy.ldm.trellis2.vae import SparseTensor

    cached = coarse.get_spatial_cache("channel2spatial_2")
    if cached is None:
        raise RuntimeError("TRELLIS.2 encoder did not retain its subdivision map")
    _fine_coords, idx, subidx = cached
    subdivision = torch.zeros(
        (coarse.coords.shape[0], 8), device=coarse.device, dtype=torch.bool
    )
    subdivision[idx.long(), subidx.long()] = True
    return SparseTensor(
        feats=subdivision,
        coords=coarse.coords.clone(),
        shape=torch.Size([coarse.shape[0], 8]),
        scale=coarse._scale,
    )


class ExternalMeshShapeEncoder(nn.Module):
    """Comfy-native implementation of the official TRELLIS.2 shape SLat encoder."""

    def __init__(self):
        super().__init__()
        from comfy.ldm.trellis2.vae import SparseConvNeXtBlock3d, SparseLinear

        model_channels = [64, 128, 256, 512, 1024]
        num_blocks = [0, 4, 8, 16, 4]
        self.input_layer = SparseLinear(6, model_channels[0])
        self.to_latent = SparseLinear(model_channels[-1], 64)
        self.blocks = nn.ModuleList()
        for i, count in enumerate(num_blocks):
            stage = nn.ModuleList(
                SparseConvNeXtBlock3d(model_channels[i]) for _ in range(count)
            )
            if i < len(num_blocks) - 1:
                stage.append(SparseResBlockS2C3d(model_channels[i], model_channels[i + 1]))
            self.blocks.append(stage)

        # Matches the upstream use_fp16=True checkpoint: the torso is fp16 while
        # input/output projections and posterior statistics remain fp32.
        self.blocks.to(dtype=torch.float16)

    def forward(self, vertices, intersected):
        x = vertices.replace(
            torch.cat(
                [vertices.feats - 0.5, intersected.feats.float() - 0.5], dim=1
            )
        )
        h = self.input_layer(x)
        h = h.type(torch.float16)
        subdivisions = []
        for stage in self.blocks:
            for block in stage:
                h = block(h)
                if isinstance(block, SparseResBlockS2C3d):
                    subdivisions.append(_subdivision_from_downsample(h))
        h = h.type(torch.float32)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        posterior = self.to_latent(h)
        mean, _logvar = posterior.feats.chunk(2, dim=-1)
        shape_slat = posterior.replace(mean)
        # The texture decoder expands the latent grid by 16x (64 -> 1024 or
        # 96 -> 1536).  Keep the sparse coordinates dynamic instead of baking
        # the historical 1024 resolution into the encoder.
        return shape_slat, list(reversed(subdivisions))


def _import_o_voxel():
    try:
        import o_voxel
    except ImportError as exc:
        raise RuntimeError(
            "The o_voxel CUDA extension is required only for converting the external "
            "Hunyuan mesh. Install a wheel matching ComfyUI's Python, PyTorch and CUDA."
        ) from exc
    return o_voxel


@torch.no_grad()
def mesh_to_flexible_dual_grid(vertices, faces, resolution: int):
    """Voxelize one normalized Y-up mesh with the same settings as the encoder.

    This shared boundary is used both by the shape encoder and by the optional
    high-resolution sparse-voxel rebuild node.  ``o_voxel`` accepts an arbitrary
    grid size, but TRELLIS.2 requires a size divisible by its 16x latent stride.
    """
    resolution = int(resolution)
    if resolution < 16 or resolution % 16:
        raise ValueError(
            f"voxel resolution must be a positive multiple of 16; got {resolution}"
        )
    o_voxel = _import_o_voxel()
    return o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices.float().cpu(),
        faces.long().cpu(),
        grid_size=resolution,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        face_weight=1.0,
        boundary_weight=0.2,
        regularization_weight=1e-2,
        timing=True,
    )


@torch.no_grad()
def encode_external_mesh(patcher, vertices, faces, resolution: int = 1024):
    import comfy.latent_formats
    import comfy.model_management
    from comfy.ldm.trellis2.vae import SparseTensor

    resolution = int(resolution)
    if resolution not in (1024, 1536):
        raise ValueError(
            "external-mesh encoding supports voxel_resolution 1024 or 1536; "
            f"got {resolution}"
        )
    comfy.model_management.load_model_gpu(patcher)
    model = patcher.model
    device = next(model.parameters()).device

    voxel_indices, dual_vertices, intersected = mesh_to_flexible_dual_grid(
        vertices, faces, resolution
    )
    coords = torch.cat(
        [torch.zeros_like(voxel_indices[:, :1]), voxel_indices], dim=-1
    ).to(torch.int32)
    vertex_field = SparseTensor(
        feats=dual_vertices * resolution - voxel_indices,
        coords=coords,
    ).to(device)
    intersection_field = vertex_field.replace(intersected).to(device)
    shape_slat, subdivides = model(vertex_field, intersection_field)

    shape_format = comfy.latent_formats.Trellis2ShapeSLAT()
    normalized = shape_format.process_in(shape_slat.feats.float()).cpu()
    coords_cpu = shape_slat.coords.to(torch.int32).cpu()
    token_count = int(coords_cpu.shape[0])
    samples = normalized.transpose(0, 1).unsqueeze(0).unsqueeze(-1).contiguous()
    latent = {
        "samples": samples,
        "coords": coords_cpu,
        "coord_counts": torch.tensor([token_count], dtype=torch.int64),
        "coord_resolution": int(resolution) // 16,
        "type": "trellis2",
        "model_frame": "y_up",
    }
    subdivides_cpu = [
        SparseTensor(
            feats=sub.feats.cpu(),
            coords=sub.coords.to(torch.int32).cpu(),
            shape=sub.shape,
            scale=sub._scale,
        )
        for sub in subdivides
    ]
    return latent, subdivides_cpu


@dataclass
class ComfyMeshPayload:
    """Tensors ready for the ComfyUI v0.40.0 ``Types.MESH`` constructor."""

    vertices: torch.Tensor
    faces: torch.Tensor
    uvs: torch.Tensor | None = None
    vertex_colors: torch.Tensor | None = None
    texture: torch.Tensor | None = None
    metallic_roughness: torch.Tensor | None = None
    normals: torch.Tensor | None = None
    normal_map: torch.Tensor | None = None
    emissive: torch.Tensor | None = None
    material: dict[str, Any] | None = None
    occlusion_in_mr: bool = False


def _array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _unwrap_singleton_batch(value: np.ndarray, name: str) -> np.ndarray:
    if value.ndim == 3:
        if value.shape[0] != 1:
            raise ValueError(f"{name} contains a batch of {value.shape[0]}; expected one mesh")
        value = value[0]
    return value


def _triangulate_faces(faces: np.ndarray) -> np.ndarray:
    faces = _unwrap_singleton_batch(faces, "faces")
    if faces.ndim != 2 or faces.shape[1] < 3:
        raise ValueError(f"Expected faces [F,3+] but got shape {tuple(faces.shape)}")
    if faces.shape[1] == 3:
        return faces

    # Fan triangulation is deterministic and is only a fallback for mesh-like
    # objects. A normal trimesh.Trimesh already stores triangular faces.
    triangles = [faces[:, [0, index, index + 1]] for index in range(1, faces.shape[1] - 1)]
    return np.concatenate(triangles, axis=0)


def _normalize_color_array(value: Any, expected_rows: int) -> torch.Tensor | None:
    if value is None:
        return None
    array = _array(value)
    array = _unwrap_singleton_batch(array, "vertex_colors")
    if array.ndim != 2 or array.shape[0] != expected_rows or array.shape[1] not in (3, 4):
        return None
    if np.issubdtype(array.dtype, np.integer):
        maximum = float(np.iinfo(array.dtype).max)
        array = array.astype(np.float32) / maximum
    else:
        array = array.astype(np.float32)
        if array.size and float(np.nanmax(array)) > 1.0:
            array = array / 255.0
    if not np.isfinite(array).all():
        return None
    return torch.from_numpy(np.ascontiguousarray(array)).clamp_(0.0, 1.0)


def _image_tensor(value: Any) -> torch.Tensor | None:
    if value is None:
        return None
    try:
        array = _array(value)
    except Exception:
        return None
    if array.ndim == 4:
        if array.shape[0] != 1:
            return None
        array = array[0]
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.ndim != 3 or array.shape[-1] not in (1, 2, 3, 4):
        return None
    if array.shape[-1] < 3:
        array = np.repeat(array[..., :1], 3, axis=-1)
    else:
        array = array[..., :3]
    if np.issubdtype(array.dtype, np.integer):
        maximum = float(np.iinfo(array.dtype).max)
        array = array.astype(np.float32) / maximum
    else:
        array = array.astype(np.float32)
        if array.size and float(np.nanmax(array)) > 1.0:
            array = array / 255.0
    if not np.isfinite(array).all():
        return None
    return torch.from_numpy(np.ascontiguousarray(array)).clamp_(0.0, 1.0).unsqueeze(0)


def _material_value(material: Any, *names: str) -> Any:
    if material is None:
        return None
    for name in names:
        try:
            value = getattr(material, name)
        except Exception:
            value = None
        if value is not None:
            return value
    return None


def _factor(value: Any, count: int) -> list[float] | None:
    if value is None:
        return None
    array = _array(value).reshape(-1)
    if array.size < count:
        return None
    array = array[:count].astype(np.float32)
    if np.issubdtype(_array(value).dtype, np.integer) or (array.size and float(np.max(array)) > 1.0):
        array = array / 255.0
    if not np.isfinite(array).all():
        return None
    return np.clip(array, 0.0, 1.0).tolist()


def _material_overrides(material: Any) -> dict[str, Any] | None:
    if material is None:
        return None
    result: dict[str, Any] = {}
    base_color = _factor(_material_value(material, "baseColorFactor"), 4)
    emissive = _factor(_material_value(material, "emissiveFactor"), 3)
    metallic = _material_value(material, "metallicFactor")
    roughness = _material_value(material, "roughnessFactor")
    double_sided = _material_value(material, "doubleSided")
    if base_color is not None:
        result["base_color_factor"] = base_color
    if metallic is not None:
        result["metallic_factor"] = float(metallic)
    if roughness is not None:
        result["roughness_factor"] = float(roughness)
    if emissive is not None:
        result["emissive_factor"] = emissive
    if double_sided is not None:
        result["double_sided"] = bool(double_sided)
    return result or None


def _as_single_mesh(mesh: Any) -> Any:
    # Some pipelines wrap their only result in a list/tuple or a tiny result object.
    if isinstance(mesh, (list, tuple)):
        if len(mesh) != 1:
            raise ValueError(f"Expected one TRIMESH object, got a sequence of {len(mesh)}")
        mesh = mesh[0]
    if not hasattr(mesh, "vertices") and hasattr(mesh, "mesh"):
        mesh = mesh.mesh

    # trimesh.Scene does not expose faces. Collapse it with scene transforms applied.
    if not hasattr(mesh, "faces") and hasattr(mesh, "geometry") and hasattr(mesh, "graph"):
        converted = None
        if hasattr(mesh, "to_geometry"):
            try:
                converted = mesh.to_geometry()
            except Exception:
                converted = None
        if converted is None and hasattr(mesh, "dump"):
            try:
                converted = mesh.dump(concatenate=True)
            except TypeError:
                converted = mesh.dump()
        if isinstance(converted, (list, tuple)):
            try:
                import trimesh

                converted = trimesh.util.concatenate(converted)
            except Exception as exc:
                raise ValueError("Could not concatenate the geometries in the TRIMESH scene") from exc
        if converted is not None:
            mesh = converted

    if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
        raise TypeError(
            "Input is not a trimesh.Trimesh-compatible object: vertices and faces are required"
        )
    return mesh


def trimesh_to_payload(
    mesh: Any,
    *,
    copy_visuals: bool = True,
    copy_normals: bool = True,
    flip_uv_v: bool = False,
) -> ComfyMeshPayload:
    """Convert a Trimesh-like object without importing a particular Trimesh version."""

    mesh = _as_single_mesh(mesh)
    vertices_array = _unwrap_singleton_batch(_array(mesh.vertices), "vertices")
    faces_source = _array(mesh.faces)

    if vertices_array.ndim != 2 or vertices_array.shape[1] != 3:
        raise ValueError(f"Expected vertices [N,3] but got shape {tuple(vertices_array.shape)}")
    if vertices_array.shape[0] == 0:
        raise ValueError("Cannot convert an empty TRIMESH")
    vertices_array = vertices_array.astype(np.float32)
    if not np.isfinite(vertices_array).all():
        raise ValueError("TRIMESH vertices contain NaN or infinity")

    if np.issubdtype(faces_source.dtype, np.floating):
        if not np.isfinite(faces_source).all() or not np.equal(faces_source, np.floor(faces_source)).all():
            raise ValueError("TRIMESH face indices must be finite integers")
    faces_array = _triangulate_faces(faces_source).astype(np.int64)
    if faces_array.shape[0] == 0:
        raise ValueError("Cannot convert a TRIMESH without faces")
    if int(faces_array.min()) < 0 or int(faces_array.max()) >= vertices_array.shape[0]:
        raise ValueError("TRIMESH contains a face index outside the vertex array")

    vertices = torch.from_numpy(np.ascontiguousarray(vertices_array)).unsqueeze(0)
    faces = torch.from_numpy(np.ascontiguousarray(faces_array)).to(torch.int32).unsqueeze(0)
    payload = ComfyMeshPayload(vertices=vertices, faces=faces)

    if copy_normals:
        try:
            normals_array = _unwrap_singleton_batch(_array(mesh.vertex_normals), "vertex_normals")
        except Exception:
            normals_array = None
        if (
            normals_array is not None
            and normals_array.ndim == 2
            and normals_array.shape == vertices_array.shape
            and np.isfinite(normals_array).all()
        ):
            normals = torch.from_numpy(
                np.ascontiguousarray(normals_array.astype(np.float32))
            )
            lengths = torch.linalg.vector_norm(normals, dim=-1, keepdim=True)
            if bool(torch.all(lengths > 1e-12)):
                payload.normals = (normals / lengths).unsqueeze(0)

    if not copy_visuals:
        return payload

    visual = getattr(mesh, "visual", None)
    uv_value = getattr(visual, "uv", None) if visual is not None else None
    if uv_value is not None:
        uv_array = _unwrap_singleton_batch(_array(uv_value), "uvs")
        if (
            uv_array.ndim == 2
            and uv_array.shape == (vertices_array.shape[0], 2)
            and np.isfinite(uv_array).all()
        ):
            uv_array = uv_array.astype(np.float32)
            if flip_uv_v:
                uv_array = uv_array.copy()
                uv_array[:, 1] = 1.0 - uv_array[:, 1]
            payload.uvs = torch.from_numpy(np.ascontiguousarray(uv_array)).unsqueeze(0)

    visual_kind = getattr(visual, "kind", None) if visual is not None else None
    if visual_kind in ("vertex", None):
        payload.vertex_colors = _normalize_color_array(
            getattr(visual, "vertex_colors", None) if visual is not None else None,
            vertices_array.shape[0],
        )
        if payload.vertex_colors is not None:
            payload.vertex_colors = payload.vertex_colors.unsqueeze(0)

    material = getattr(visual, "material", None) if visual is not None else None
    payload.texture = _image_tensor(_material_value(material, "baseColorTexture", "image"))
    payload.metallic_roughness = _image_tensor(
        _material_value(material, "metallicRoughnessTexture")
    )
    payload.normal_map = _image_tensor(_material_value(material, "normalTexture"))
    payload.emissive = _image_tensor(_material_value(material, "emissiveTexture"))
    payload.material = _material_overrides(material)
    return payload


ENCODER_FOLDER = "trellis2_encoders"


def _ensure_encoder_folder():
    import folder_paths

    path = os.path.join(folder_paths.models_dir, ENCODER_FOLDER)
    os.makedirs(path, exist_ok=True)
    folder_paths.add_model_folder_path(ENCODER_FOLDER, path)
    return folder_paths


def _single_mesh_tensors(mesh) -> tuple[torch.Tensor, torch.Tensor]:
    vertices = mesh.vertices
    faces = mesh.faces
    if isinstance(vertices, list) or isinstance(faces, list):
        if len(vertices) != 1 or len(faces) != 1:
            raise ValueError("The adapter processes exactly one mesh per execution")
        vertices, faces = vertices[0], faces[0]
    else:
        if vertices.ndim == 3:
            if vertices.shape[0] != 1:
                raise ValueError(f"The adapter processes one mesh; got batch={vertices.shape[0]}")
            vertices = vertices[0]
        if faces.ndim == 3:
            if faces.shape[0] != 1:
                raise ValueError(f"The adapter processes one mesh; got batch={faces.shape[0]}")
            faces = faces[0]
    if vertices.ndim != 2 or vertices.shape[-1] != 3:
        raise ValueError(f"Expected mesh vertices [N,3], got {tuple(vertices.shape)}")
    if faces.ndim != 2 or faces.shape[-1] != 3:
        raise ValueError(f"Expected triangular faces [M,3], got {tuple(faces.shape)}")
    if vertices.numel() == 0 or faces.numel() == 0:
        raise ValueError("Cannot texture an empty mesh")
    return vertices.detach().float().cpu(), faces.detach().to(torch.int64).cpu()


def rotation_y(degrees: float, *, dtype=torch.float32) -> torch.Tensor:
    theta = math.radians(float(degrees))
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=dtype)


MASK_MODES = ("ignore", "foreground", "transparency")
NAF_MODES = ("stable_fp32", "dino_fallback", "comfy_v040_native")


def stable_na2d_pure_v040(
    q: torch.Tensor,
    k_lr: torch.Tensor,
    v_lr: torch.Tensor,
    kernel_size: tuple[int, int],
    dilation: tuple[int, int],
    scale: float,
    tile: int = 128,
    v_chunk: int = 64,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Numerically stable replacement for ComfyUI 0.40's NAF attention.

    ComfyUI 0.40 materializes nearest-upsampled K/V tiles and evaluates the
    attention logits in the model dtype.  The current upstream implementation
    instead groups the HR queries belonging to each LR cell and evaluates QK +
    softmax in fp32.  This function keeps the 0.40 call signature so the fix can
    be applied only while this adapter's conditioning node runs.

    ``tile`` and ``v_chunk`` are retained for signature compatibility; memory is
    bounded by chunks of LR rows, as in the newer upstream algorithm.
    """
    del tile, v_chunk
    if q.ndim != 5 or k_lr.ndim != 5 or v_lr.ndim != 5:
        raise ValueError("NAF attention expects q/k/v tensors with five dimensions")

    batch, height, width, heads, qk_dim = q.shape
    if k_lr.shape[:1] != (batch,) or k_lr.shape[3] != heads or k_lr.shape[4] != qk_dim:
        raise ValueError("NAF Q/K shapes are incompatible")
    if v_lr.shape[:1] != (batch,) or v_lr.shape[1:3] != k_lr.shape[1:3] or v_lr.shape[3] != heads:
        raise ValueError("NAF K/V shapes are incompatible")

    lr_height, lr_width = int(k_lr.shape[1]), int(k_lr.shape[2])
    dilation_h, dilation_w = int(dilation[0]), int(dilation[1])
    if height != lr_height * dilation_h or width != lr_width * dilation_w:
        raise ValueError(
            "NAF output size must be an integer multiple of the LR feature grid; "
            f"got HR={height}x{width}, LR={lr_height}x{lr_width}, dilation={dilation}"
        )

    value_dim = int(v_lr.shape[-1])
    kernel_h, kernel_w = int(kernel_size[0]), int(kernel_size[1])
    kernel_tokens = kernel_h * kernel_w
    queries_per_cell = dilation_h * dilation_w

    # Convert the v0.40 BHWHC layouts to the BCHW layouts used by the stable
    # cell-wise implementation.
    q_heads = q.permute(0, 3, 4, 1, 2).contiguous()
    k = k_lr.permute(0, 3, 4, 1, 2).reshape(
        batch, heads * qk_dim, lr_height, lr_width
    ).contiguous()
    v = v_lr.permute(0, 3, 4, 1, 2).reshape(
        batch, heads * value_dim, lr_height, lr_width
    ).contiguous()

    out = output if output is not None else torch.empty(
        (batch, heads, value_dim, height, width), device=q.device, dtype=q.dtype
    )
    padding = (kernel_w // 2, kernel_w // 2, kernel_h // 2, kernel_h // 2)
    k = torch.nn.functional.pad(k, padding)
    v = torch.nn.functional.pad(v, padding)

    # Keep live temporaries around 512 MiB, matching current upstream ComfyUI.
    element_size = q.element_size()
    row_bytes = batch * heads * lr_width * (
        kernel_tokens * (qk_dim * (element_size + 4) + value_dim * element_size)
        + queries_per_cell
        * (
            qk_dim * (element_size + 4)
            + kernel_tokens * (8 + element_size)
            + 2 * value_dim * element_size
        )
    )
    rows_per_chunk = max(1, (2**29) // max(1, row_bytes))

    for row_start in range(0, lr_height, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, lr_height)
        cell_count = (row_end - row_start) * lr_width
        k_windows = torch.nn.functional.unfold(
            k[:, :, row_start : row_end + kernel_h - 1], (kernel_h, kernel_w)
        ).view(batch, heads, qk_dim, kernel_tokens, cell_count).transpose(2, 4)
        v_windows = torch.nn.functional.unfold(
            v[:, :, row_start : row_end + kernel_h - 1], (kernel_h, kernel_w)
        ).view(batch, heads, value_dim, kernel_tokens, cell_count).transpose(2, 4)
        q_cells = (
            q_heads[..., row_start * dilation_h : row_end * dilation_h, :]
            .reshape(
                batch,
                heads,
                qk_dim,
                row_end - row_start,
                dilation_h,
                lr_width,
                dilation_w,
            )
            .permute(0, 1, 3, 5, 4, 6, 2)
            .reshape(batch, heads, cell_count, queries_per_cell, qk_dim)
        )
        attention = (
            torch.matmul(q_cells.float(), k_windows.float().transpose(-1, -2)) * scale
        ).softmax(dim=-1).to(v_windows.dtype)
        chunk = torch.matmul(attention, v_windows).view(
            batch,
            heads,
            row_end - row_start,
            lr_width,
            dilation_h,
            dilation_w,
            value_dim,
        )
        chunk = chunk.permute(0, 1, 6, 2, 4, 3, 5).reshape(
            batch,
            heads,
            value_dim,
            (row_end - row_start) * dilation_h,
            width,
        )
        out[..., row_start * dilation_h : row_end * dilation_h, :] = chunk

    return out


def apply_comfy_transparency(
    image: torch.Tensor,
    mask: torch.Tensor | None,
    name: str,
    mask_mode: str = "ignore",
):
    if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[-1] not in (3, 4):
        raise ValueError(f"{name} must be a ComfyUI IMAGE [B,H,W,3 or 4]")
    if mask_mode not in MASK_MODES:
        raise ValueError(f"mask_mode must be one of {MASK_MODES}, got {mask_mode!r}")
    rgb = image[..., :3].float().clamp(0, 1)
    alpha = image[..., 3:4].float().clamp(0, 1) if image.shape[-1] == 4 else None
    if mask is not None and mask_mode != "ignore":
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim != 3:
            raise ValueError(f"{name}_mask must be [B,H,W]")
        if mask.shape[0] not in (1, image.shape[0]):
            raise ValueError(f"{name}_mask batch must be 1 or match the image batch")
        if mask.shape[0] == 1 and image.shape[0] > 1:
            mask = mask.expand(image.shape[0], -1, -1)
        if tuple(mask.shape[1:]) != tuple(image.shape[1:3]):
            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(1).float(),
                size=image.shape[1:3],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
        mask = mask.float().clamp(0, 1)
        # ComfyUI does not impose one universal MASK polarity. RemoveBackground
        # emits 1=foreground, while LoadImage's alpha-derived MASK uses
        # 1=transparent. Keep the choice explicit instead of guessing.
        mask_alpha = mask if mask_mode == "foreground" else 1.0 - mask
        mask_alpha = mask_alpha.unsqueeze(-1)
        alpha = mask_alpha if alpha is None else alpha * mask_alpha
    return rgb if alpha is None else rgb * alpha


def make_prepared_views_preview(prepared: dict[str, torch.Tensor], size: int = 512):
    """Return front/left/right/back as a fixed-size IMAGE batch for visual QA."""
    items = []
    for name in ("front", "left", "right", "back"):
        image = prepared[name][:1, ..., :3].float().clamp(0, 1)
        if tuple(image.shape[1:3]) != (size, size):
            image = torch.nn.functional.interpolate(
                image.movedim(-1, 1),
                size=(size, size),
                mode="bilinear",
                align_corners=False,
            ).movedim(1, -1)
        items.append(image)
    return torch.cat(items, dim=0)


VIEW_NAMES = ("front", "left", "right", "back")
VIEW_AZIMUTHS = (0.0, 90.0, 270.0, 180.0)
VIEW_LAYER_ORDERS = tuple(
    " > ".join(order) for order in itertools.permutations(VIEW_NAMES)
)
DEFAULT_VIEW_LAYER_ORDER = "back > left > right > front"
BLEND_MODES = (
    "normal",
    "darken",
    "multiply",
    "color_burn",
    "linear_burn",
    "darker_color",
    "lighten",
    "screen",
    "color_dodge",
    "linear_dodge_add",
    "lighter_color",
    "overlay",
    "soft_light",
    "hard_light",
    "vivid_light",
    "linear_light",
    "pin_light",
    "hard_mix",
    "difference",
    "exclusion",
    "subtract",
    "divide",
    "hue",
    "saturation",
    "color",
    "luminosity",
)
GAUSSIAN_BLUR_TARGETS = ("rgb", "blend_mask", "rgb_and_mask")
COVERAGE_MODES = ("angular_sector", "all_camera_visible")
CAMERA_COMPOSITE_MODES = ("ordered_layers", "weighted_average")
GEOMETRY_ALIGN_FIT_MODES = ("contain", "stretch", "none")
GEOMETRY_ALIGN_MASK_MODES = ("target", "intersection", "warped_source")
HUMANOID_GLOBAL_MODELS = ("similarity", "affine")


@dataclass(frozen=True)
class ReprojectionViewSettings:
    view: str
    enabled: bool
    weight: float
    coverage_degrees: float
    edge_softness_percent: float
    background_threshold: float
    blend_mode: str
    gaussian_blur_target: str
    gaussian_blur_sigma_pixels: float
    flip_horizontal: bool
    edge_feather_pixels: float
    crop_left_percent: float = 0.0
    crop_right_percent: float = 0.0
    crop_top_percent: float = 0.0
    crop_bottom_percent: float = 0.0
    coverage_mode: str = "angular_sector"
    camera_feather_pixels: float = 0.0
    camera_feather_inset_pixels: float = 0.0


@dataclass(frozen=True)
class GeometryAlignSettings:
    view: str
    global_fit_mode: str
    global_alignment_strength: float
    local_alignment_strength: float
    max_local_shift_pixels: float
    boundary_lock_pixels: float
    analysis_size: int
    control_grid_size: int
    optimization_steps: int
    depth_weight: float
    edge_weight: float
    silhouette_weight: float
    rigidity: float
    confidence_threshold: float
    source_depth_near_is_white: bool
    target_depth_near_is_white: bool
    background_threshold: float
    output_mask_mode: str
    confidence_gate_strength: float
    source_depth_blur_pixels: float
    fold_prevention_strength: float


@dataclass(frozen=True)
class HumanoidAlignSettings:
    view: str
    global_model: str
    score_threshold: float
    global_alignment_strength: float
    torso_strength: float
    arm_strength: float
    leg_strength: float
    head_strength: float
    hand_strength: float
    foot_strength: float
    tps_regularization: float
    max_local_shift_pixels: float
    outer_stability: float
    fold_protection: bool
    max_folded_area_percent: float
    background_threshold: float
    output_mask_mode: str
    source_person_index: int
    target_person_index: int
def parse_view_layer_order(value) -> tuple[str, str, str, str]:
    if isinstance(value, str):
        result = tuple(part.strip().lower() for part in value.split(">"))
    else:
        result = tuple(str(part).strip().lower() for part in value)
    if len(result) != 4 or set(result) != set(VIEW_NAMES):
        raise ValueError(
            "layer_order_bottom_to_top must contain front, left, right and back "
            f"exactly once; got {value!r}"
        )
    return result


def _four_view_values(value, name: str) -> tuple[float, float, float, float]:
    if isinstance(value, (tuple, list)):
        if len(value) != 4:
            raise ValueError(f"{name} must contain four F/L/R/B values")
        return tuple(float(item) for item in value)
    scalar = float(value)
    return (scalar, scalar, scalar, scalar)


def _validate_view_settings(settings, expected_view: str) -> ReprojectionViewSettings:
    if not isinstance(settings, ReprojectionViewSettings):
        raise ValueError(
            f"{expected_view}_settings must come from Pixal3D Reprojection View Settings"
        )
    if settings.view != expected_view:
        raise ValueError(
            f"{expected_view}_settings received a {settings.view!r} configuration. "
            f"Set its view to {expected_view!r} or connect it to the matching socket."
        )
    if settings.blend_mode not in BLEND_MODES:
        raise ValueError(f"Unknown blend mode: {settings.blend_mode!r}")
    if settings.gaussian_blur_target not in GAUSSIAN_BLUR_TARGETS:
        raise ValueError(
            f"Unknown Gaussian blur target: {settings.gaussian_blur_target!r}"
        )
    if settings.coverage_mode not in COVERAGE_MODES:
        raise ValueError(f"Unknown coverage mode: {settings.coverage_mode!r}")
    return settings


def _world_from_grid_coords(spatial: torch.Tensor, resolution: int) -> torch.Tensor:
    """Convert TRELLIS grid coordinates to the world convention used by Pixal3D."""
    norm = spatial.float() / float(max(1, resolution - 1)) * 2.0 - 1.0
    return torch.stack([norm[:, 0], -norm[:, 2], norm[:, 1]], dim=-1) / 2.0


def _sparse_grid_keys(spatial: torch.Tensor, resolution: int) -> torch.Tensor:
    """Pack XYZ integer coordinates into collision-free int64 keys."""
    spatial = spatial.to(torch.int64)
    resolution = int(resolution)
    return spatial[:, 0] + resolution * (
        spatial[:, 1] + resolution * spatial[:, 2]
    )


def _lookup_sparse_keys(
    sorted_keys: torch.Tensor, query_keys: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return source indices and an exact-hit mask for packed sparse keys."""
    flat = query_keys.reshape(-1).contiguous()
    positions = torch.searchsorted(sorted_keys, flat)
    safe_positions = positions.clamp_max(max(0, int(sorted_keys.numel()) - 1))
    hits = positions < sorted_keys.numel()
    if sorted_keys.numel():
        hits &= sorted_keys[safe_positions] == flat
    source_indices = safe_positions.reshape(query_keys.shape)
    return source_indices, hits.reshape(query_keys.shape)


def _grid_offset_shell(radius: int, device: torch.device) -> torch.Tensor:
    offsets = [
        (x, y, z)
        for x in range(-radius, radius + 1)
        for y in range(-radius, radius + 1)
        for z in range(-radius, radius + 1)
        if radius == 1 or max(abs(x), abs(y), abs(z)) == radius
    ]
    offsets.sort(key=lambda value: value[0] ** 2 + value[1] ** 2 + value[2] ** 2)
    return torch.tensor(offsets, device=device, dtype=torch.int64)


@torch.no_grad()
def transfer_sparse_voxel_features(
    source_coords: torch.Tensor,
    source_features: torch.Tensor,
    source_resolution: int,
    target_coords: torch.Tensor,
    target_resolution: int,
    transfer_mode: str = "edge_aware",
    max_search_radius_voxels: int = 4,
    rgb_edge_threshold: float = 0.20,
    pbr_edge_threshold: float = 0.15,
    chunk_size: int = 32768,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Resample every sparse voxel channel without crossing sharp material edges.

    The primary 3x3x3 neighbourhood uses inverse-distance interpolation.  The
    spatially nearest source voxel is the anchor, and neighbours with a large RGB
    or PBR discontinuity are rejected before weights are normalized.  Wider search
    shells are used only for holes and copy their nearest value, preventing a
    distant opposite surface from being averaged into a thin part.
    """
    if transfer_mode not in ("edge_aware", "nearest"):
        raise ValueError("transfer_mode must be 'edge_aware' or 'nearest'")
    if not isinstance(source_coords, torch.Tensor) or source_coords.ndim != 2:
        raise ValueError("source_coords must be a tensor shaped [N,3] or [N,4]")
    if source_coords.shape[1] not in (3, 4):
        raise ValueError(f"source_coords must be [N,3/4], got {tuple(source_coords.shape)}")
    if not isinstance(target_coords, torch.Tensor) or target_coords.ndim != 2:
        raise ValueError("target_coords must be a tensor shaped [N,3] or [N,4]")
    if target_coords.shape[1] not in (3, 4):
        raise ValueError(f"target_coords must be [N,3/4], got {tuple(target_coords.shape)}")
    if source_features.ndim != 2 or source_features.shape[0] != source_coords.shape[0]:
        raise ValueError("source_features must be [N,C] and match source_coords")
    if source_features.shape[0] == 0 or source_features.shape[1] == 0:
        raise ValueError("cannot transfer an empty sparse voxel field")

    source_resolution = int(source_resolution)
    target_resolution = int(target_resolution)
    if source_resolution <= 1 or target_resolution <= 1:
        raise ValueError("source and target voxel resolutions must be greater than one")
    max_search_radius_voxels = max(1, int(max_search_radius_voxels))
    chunk_size = max(256, int(chunk_size))

    source_spatial = source_coords[:, -3:]
    target_spatial = target_coords[:, -3:]
    if source_coords.shape[1] == 4 and torch.any(source_coords[:, 0] != 0):
        raise ValueError("sparse voxel rebuild currently supports one mesh per execution")
    if target_coords.shape[1] == 4 and torch.any(target_coords[:, 0] != 0):
        raise ValueError("target sparse grid must describe one mesh")
    if torch.any(source_spatial < 0) or torch.any(source_spatial >= source_resolution):
        raise ValueError("source sparse coordinates lie outside voxel.resolution")
    if torch.any(target_spatial < 0) or torch.any(target_spatial >= target_resolution):
        raise ValueError("target sparse coordinates lie outside target_resolution")

    device = source_features.device
    source_spatial = source_spatial.to(device=device, dtype=torch.int64)
    source_features = source_features.contiguous()
    keys = _sparse_grid_keys(source_spatial, source_resolution)
    sorted_keys, order = keys.sort()
    sorted_features = source_features[order]

    channel_count = int(source_features.shape[1])
    target_count = int(target_spatial.shape[0])
    transferred = torch.empty(
        (target_count, channel_count), device=device, dtype=source_features.dtype
    )
    fallback_value = source_features.float().mean(dim=0).to(source_features.dtype)
    primary_offsets = _grid_offset_shell(1, device)
    stats = {"primary": 0, "fallback": 0, "unmatched": 0, "edge_rejected": 0}
    # o_voxel divides the fixed AABB by grid_size, so integer coordinates name
    # cells, not endpoint samples.  Map cell centres between the two grids.
    scale = float(source_resolution) / float(target_resolution)

    for start in range(0, target_count, chunk_size):
        end = min(start + chunk_size, target_count)
        target_chunk = target_spatial[start:end].to(device=device, dtype=torch.float32)
        mapped = (target_chunk + 0.5) * scale - 0.5
        centers = mapped.round().to(torch.int64)
        candidates = centers[:, None, :] + primary_offsets[None, :, :]
        valid_bounds = ((candidates >= 0) & (candidates < source_resolution)).all(dim=-1)
        candidate_keys = _sparse_grid_keys(
            candidates.reshape(-1, 3).clamp(0, source_resolution - 1),
            source_resolution,
        ).reshape(candidates.shape[:2])
        candidate_indices, occupied = _lookup_sparse_keys(sorted_keys, candidate_keys)
        occupied &= valid_bounds
        distances_sq = (candidates.float() - mapped[:, None, :]).square().sum(dim=-1)
        distances_sq = distances_sq.masked_fill(~occupied, float("inf"))
        nearest_slots = distances_sq.argmin(dim=1)
        nearest_distances = distances_sq.gather(1, nearest_slots[:, None]).squeeze(1)
        found = torch.isfinite(nearest_distances)
        output_chunk = fallback_value.expand(end - start, -1).clone()

        if torch.any(found):
            rows = torch.arange(end - start, device=device)
            anchor_indices = candidate_indices[rows, nearest_slots]
            anchor_features = sorted_features[anchor_indices]
            if transfer_mode == "nearest":
                output_chunk[found] = anchor_features[found]
            else:
                candidate_features = sorted_features[candidate_indices].float()
                anchor_float = anchor_features.float()[:, None, :]
                edge_keep = occupied.clone()
                if channel_count >= 3:
                    rgb_delta = (candidate_features[..., :3] - anchor_float[..., :3]).abs().amax(dim=-1)
                    edge_keep &= rgb_delta <= float(rgb_edge_threshold)
                if channel_count > 3:
                    pbr_delta = (candidate_features[..., 3:] - anchor_float[..., 3:]).abs().amax(dim=-1)
                    edge_keep &= pbr_delta <= float(pbr_edge_threshold)
                edge_keep[rows, nearest_slots] = found
                stats["edge_rejected"] += int((occupied & ~edge_keep).sum().item())
                weights = torch.where(
                    edge_keep,
                    1.0 / (distances_sq + 1.0e-4),
                    torch.zeros_like(distances_sq),
                )
                weight_sum = weights.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
                interpolated = (candidate_features * weights[..., None]).sum(dim=1) / weight_sum
                output_chunk[found] = interpolated[found].to(source_features.dtype)
            stats["primary"] += int(found.sum().item())

        missing = ~found
        for radius in range(2, max_search_radius_voxels + 1):
            if not torch.any(missing):
                break
            missing_rows = torch.nonzero(missing, as_tuple=False).squeeze(1)
            shell = _grid_offset_shell(radius, device)
            shell_candidates = centers[missing_rows, None, :] + shell[None, :, :]
            valid_shell = (
                (shell_candidates >= 0) & (shell_candidates < source_resolution)
            ).all(dim=-1)
            shell_keys = _sparse_grid_keys(
                shell_candidates.reshape(-1, 3).clamp(0, source_resolution - 1),
                source_resolution,
            ).reshape(shell_candidates.shape[:2])
            shell_indices, shell_occupied = _lookup_sparse_keys(sorted_keys, shell_keys)
            shell_occupied &= valid_shell
            shell_distances = (
                shell_candidates.float() - mapped[missing_rows, None, :]
            ).square().sum(dim=-1)
            shell_distances = shell_distances.masked_fill(~shell_occupied, float("inf"))
            shell_slots = shell_distances.argmin(dim=1)
            shell_best = shell_distances.gather(1, shell_slots[:, None]).squeeze(1)
            shell_found = torch.isfinite(shell_best)
            if torch.any(shell_found):
                resolved_rows = missing_rows[shell_found]
                resolved_indices = shell_indices[
                    torch.arange(missing_rows.numel(), device=device), shell_slots
                ][shell_found]
                output_chunk[resolved_rows] = sorted_features[resolved_indices]
                missing[resolved_rows] = False
                stats["fallback"] += int(shell_found.sum().item())

        stats["unmatched"] += int(missing.sum().item())
        transferred[start:end] = output_chunk

    return transferred, stats


def _project_world_points(
    world: torch.Tensor,
    width: int,
    height: int,
    fov: float,
    azimuth: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project world points with the exact four-view Pixal3D camera convention."""
    fov_radians = math.radians(float(fov))
    distance = 1.1 * 0.5 / math.tan(fov_radians / 2.0)
    focal_pixels = width / (2.0 * math.tan(fov_radians / 2.0))
    az = math.radians(float(azimuth))
    back = world.new_tensor([math.sin(az), -math.cos(az), 0.0])
    right = world.new_tensor([math.cos(az), math.sin(az), 0.0])
    up = torch.cross(back, right, dim=0)
    rotation = torch.stack([right, up, back], dim=-1)
    translation = back * distance
    camera = (world - translation) @ rotation
    x_camera, y_camera, z_camera = camera.unbind(dim=-1)
    depth = -z_camera
    x_pixel = focal_pixels * x_camera / (depth + 1e-8) + width / 2.0
    y_pixel = -focal_pixels * y_camera / (depth + 1e-8) + height / 2.0
    valid = (
        (x_pixel >= 0)
        & (x_pixel < width)
        & (y_pixel >= 0)
        & (y_pixel < height)
        & (depth > 0)
    )
    return x_pixel, y_pixel, depth, valid, back


def _foreground_bbox(image: torch.Tensor, threshold: float) -> tuple[float, float, float, float, bool]:
    """Return inclusive foreground bounds and whether a real foreground was found."""
    mask = image[..., :3].amax(dim=-1) > float(threshold)
    if image.shape[-1] >= 4:
        mask &= image[..., 3] > float(threshold)
    rows, cols = torch.nonzero(mask, as_tuple=True)
    height, width = int(image.shape[0]), int(image.shape[1])
    if rows.numel() == 0:
        return 0.0, 0.0, float(width - 1), float(height - 1), False
    return (
        float(cols.min()),
        float(rows.min()),
        float(cols.max()),
        float(rows.max()),
        True,
    )


def _warp_image_to_canvas(
    image: torch.Tensor,
    target_size: int,
    scale: float,
    source_center: tuple[float, float],
    target_center: tuple[float, float],
    x_deformation_percent: float = 0.0,
    y_deformation_percent: float = 0.0,
) -> torch.Tensor:
    """Place an image on a square canvas with scale, translation and axis deformation."""
    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise ValueError("Warp input must be an RGB or RGBA image [H,W,3/4]")
    # Keep alpha when it is available.  Auto Align used to discard it here,
    # turning transparent padding into ordinary black pixels before reprojection.
    source = image.float().clamp(0.0, 1.0)
    source_height, source_width = int(source.shape[0]), int(source.shape[1])
    dtype = source.dtype
    device = source.device
    out_axis = torch.arange(target_size, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(out_axis, out_axis, indexing="ij")
    scale_x = float(scale) * (1.0 + float(x_deformation_percent) / 100.0)
    scale_y = float(scale) * (1.0 + float(y_deformation_percent) / 100.0)
    if scale_x <= 0.0 or scale_y <= 0.0:
        raise ValueError("Axis deformation must keep both image scales greater than zero")
    source_x = (xx - float(target_center[0])) / scale_x + float(source_center[0])
    source_y = (yy - float(target_center[1])) / scale_y + float(source_center[1])
    grid_x = (source_x + 0.5) / float(source_width) * 2.0 - 1.0
    grid_y = (source_y + 0.5) / float(source_height) * 2.0 - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
    return torch.nn.functional.grid_sample(
        source.movedim(-1, 0).unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).movedim(1, -1)


def _geometry_single_nchw(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 4 or value.shape[0] != 1:
        raise ValueError(f"{name} must be a single-item IMAGE batch [1,H,W,C]")
    if value.shape[-1] < 1:
        raise ValueError(f"{name} has no channels")
    # ComfyUI executes nodes under torch.inference_mode().  A clone made while
    # Geometry-Aware Align has locally disabled that mode converts incoming
    # inference tensors into ordinary tensors that autograd may safely save for
    # grid_sample's backward pass.
    return value.detach().float().clone().movedim(-1, 1)


def _geometry_optional_mask_nchw(
    value: torch.Tensor | None,
    name: str,
    height: int,
    width: int,
) -> torch.Tensor | None:
    if value is None:
        return None
    if not isinstance(value, torch.Tensor) or value.ndim != 3 or value.shape[0] != 1:
        raise ValueError(f"{name} must be a single-item MASK batch [1,H,W]")
    mask = value.detach().float().clone().unsqueeze(1).clamp(0.0, 1.0)
    if tuple(mask.shape[-2:]) != (height, width):
        mask = torch.nn.functional.interpolate(mask, size=(height, width), mode="nearest")
    return mask


def _geometry_gray_nchw(value: torch.Tensor) -> torch.Tensor:
    if value.shape[1] == 1:
        return value[:, :1]
    rgb = value[:, :3]
    weights = rgb.new_tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1)
    return (rgb * weights).sum(dim=1, keepdim=True)


def _geometry_normalize_depth(
    depth: torch.Tensor,
    mask: torch.Tensor,
    near_is_white: bool,
) -> tuple[torch.Tensor, float, float]:
    value = _geometry_gray_nchw(depth).clamp(0.0, 1.0)
    if not bool(near_is_white):
        value = 1.0 - value
    selected = value[mask > 0.5]
    if selected.numel() < 16:
        raise ValueError("Geometry-Aware Align depth mask contains too few valid pixels")
    low = float(torch.quantile(selected, 0.01))
    high = float(torch.quantile(selected, 0.99))
    if high - low < 1e-5:
        low = float(selected.amin())
        high = float(selected.amax())
    if high - low < 1e-5:
        raise ValueError("Geometry-Aware Align depth has no usable variation inside its mask")
    normalized = ((value - low) / (high - low)).clamp(0.0, 1.0) * mask
    return normalized, low, high


def _geometry_masked_blur(
    value: torch.Tensor,
    mask: torch.Tensor,
    sigma_pixels: float,
) -> torch.Tensor:
    """Blur depth inside a mask without pulling black background into its edge."""
    sigma = max(0.0, float(sigma_pixels))
    if sigma <= 1e-6:
        return value * mask
    numerator = _gaussian_blur_nchw(value * mask, sigma, padding_mode="constant")
    denominator = _gaussian_blur_nchw(mask, sigma, padding_mode="constant")
    return (numerator / denominator.clamp_min(1e-5)).clamp(0.0, 1.0) * mask


def _geometry_mask_bbox(mask: torch.Tensor) -> tuple[float, float, float, float]:
    rows, cols = torch.nonzero(mask[0, 0] > 0.5, as_tuple=True)
    if rows.numel() < 16:
        raise ValueError("Geometry-Aware Align could not find a usable foreground mask")
    return float(cols.min()), float(rows.min()), float(cols.max()), float(rows.max())


def _geometry_base_grid(
    source_mask: torch.Tensor,
    target_mask: torch.Tensor,
    fit_mode: str,
    strength: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if fit_mode not in GEOMETRY_ALIGN_FIT_MODES:
        raise ValueError(f"Unknown geometry align fit mode: {fit_mode!r}")
    height, width = int(target_mask.shape[-2]), int(target_mask.shape[-1])
    sx0, sy0, sx1, sy1 = _geometry_mask_bbox(source_mask)
    tx0, ty0, tx1, ty1 = _geometry_mask_bbox(target_mask)
    source_width = max(1.0, sx1 - sx0 + 1.0)
    source_height = max(1.0, sy1 - sy0 + 1.0)
    target_width = max(1.0, tx1 - tx0 + 1.0)
    target_height = max(1.0, ty1 - ty0 + 1.0)
    source_center_x = (sx0 + sx1) * 0.5
    source_center_y = (sy0 + sy1) * 0.5
    target_center_x = (tx0 + tx1) * 0.5
    target_center_y = (ty0 + ty1) * 0.5
    if fit_mode == "none":
        requested_scale_x = requested_scale_y = 1.0
        target_center_x = source_center_x
        target_center_y = source_center_y
    elif fit_mode == "contain":
        uniform = min(target_width / source_width, target_height / source_height)
        requested_scale_x = requested_scale_y = uniform
    else:
        requested_scale_x = target_width / source_width
        requested_scale_y = target_height / source_height
    amount = min(1.0, max(0.0, float(strength)))
    scale_x = 1.0 + (requested_scale_x - 1.0) * amount
    scale_y = 1.0 + (requested_scale_y - 1.0) * amount
    placed_center_x = source_center_x + (target_center_x - source_center_x) * amount
    placed_center_y = source_center_y + (target_center_y - source_center_y) * amount
    axis_x = torch.arange(width, device=target_mask.device, dtype=torch.float32)
    axis_y = torch.arange(height, device=target_mask.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(axis_y, axis_x, indexing="ij")
    source_x = (xx - placed_center_x) / max(scale_x, 1e-6) + source_center_x
    source_y = (yy - placed_center_y) / max(scale_y, 1e-6) + source_center_y
    grid = torch.stack(
        [
            (source_x + 0.5) / float(width) * 2.0 - 1.0,
            (source_y + 0.5) / float(height) * 2.0 - 1.0,
        ],
        dim=-1,
    ).unsqueeze(0)
    return grid, {
        "scale_x": float(scale_x),
        "scale_y": float(scale_y),
        "offset_x": float(placed_center_x - source_center_x),
        "offset_y": float(placed_center_y - source_center_y),
    }


def _geometry_edge_magnitude(value: torch.Tensor) -> torch.Tensor:
    sobel_x = value.new_tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]
    ).unsqueeze(0)
    sobel_y = value.new_tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]
    ).unsqueeze(0)
    gx = torch.nn.functional.conv2d(value, sobel_x, padding=1)
    gy = torch.nn.functional.conv2d(value, sobel_y, padding=1)
    magnitude = torch.sqrt(gx.square() + gy.square() + 1e-8)
    return magnitude / magnitude.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)


def _geometry_grid_with_flow(base_grid: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    height, width = int(flow.shape[-2]), int(flow.shape[-1])
    grid = base_grid.clone()
    grid[..., 0] += flow[:, 0] * (2.0 / float(max(1, width - 1)))
    grid[..., 1] += flow[:, 1] * (2.0 / float(max(1, height - 1)))
    return grid


def _geometry_warp(value: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.grid_sample(
        value,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )


def _geometry_deformation_penalties(
    flow: torch.Tensor,
    base_scale_x: float,
    base_scale_y: float,
    max_shift_pixels: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return elastic strain, fold-over and bending penalties for a pixel flow.

    A uniform translation has zero strain and bending.  Symmetric derivatives
    penalise local stretching/shear while allowing rigid rotation.  The Jacobian
    determinant term prevents the backward sampling map from folding over itself,
    which otherwise produces the characteristic minced/smeared texture.
    """
    if flow.shape[-2] < 3 or flow.shape[-1] < 3:
        zero = flow.new_zeros(())
        return zero, zero, zero
    u = flow[:, 0]
    v = flow[:, 1]
    du_dx = u[:, :-1, 1:] - u[:, :-1, :-1]
    du_dy = u[:, 1:, :-1] - u[:, :-1, :-1]
    dv_dx = v[:, :-1, 1:] - v[:, :-1, :-1]
    dv_dy = v[:, 1:, :-1] - v[:, :-1, :-1]

    normal_x = du_dx
    normal_y = dv_dy
    shear = 0.5 * (du_dy + dv_dx)
    strain = (
        normal_x.square().mean()
        + normal_y.square().mean()
        + 2.0 * shear.square().mean()
    )

    base_j00 = 1.0 / max(float(base_scale_x), 1e-6)
    base_j11 = 1.0 / max(float(base_scale_y), 1e-6)
    determinant = (
        (base_j00 + du_dx) * (base_j11 + dv_dy) - du_dy * dv_dx
    )
    minimum_determinant = 0.20 * base_j00 * base_j11
    fold = torch.relu(minimum_determinant - determinant).square().mean()

    dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
    dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]
    dxx = dx[:, :, :, 1:] - dx[:, :, :, :-1]
    dyy = dy[:, :, 1:, :] - dy[:, :, :-1, :]
    bending = dxx.square().mean() + dyy.square().mean()
    magnitude = flow.square().mean() / max(float(max_shift_pixels) ** 2, 1.0)
    elastic = strain + 0.25 * bending + 0.0025 * magnitude
    return elastic, fold, determinant


@torch.inference_mode(False)
def geometry_aware_view_align(
    image: torch.Tensor,
    source_depth: torch.Tensor,
    target_depth: torch.Tensor,
    settings: GeometryAlignSettings,
    source_mask: torch.Tensor | None = None,
    target_mask: torch.Tensor | None = None,
):
    """Align one RGB view to rendered depth with a locally trainable flow grid.

    ComfyUI wraps normal execution in ``torch.inference_mode()``.  The explicit
    decorator is required because ``torch.enable_grad()`` alone only overrides
    ``no_grad`` and cannot re-enable graph construction inside inference mode.
    """
    if not isinstance(settings, GeometryAlignSettings):
        raise ValueError("settings must come from Pixal3D Geometry Align Settings")
    source_image = _geometry_single_nchw(image, "image")
    source_depth_nchw = _geometry_single_nchw(source_depth, "source_depth")
    target_depth_nchw = _geometry_single_nchw(target_depth, "target_depth")
    target_height, target_width = (
        int(target_depth_nchw.shape[-2]),
        int(target_depth_nchw.shape[-1]),
    )
    if source_image.shape[1] < 3:
        source_image = source_image.repeat(1, 3, 1, 1)
    source_alpha = source_image[:, 3:4] if source_image.shape[1] >= 4 else None
    source_rgb = source_image[:, :3].clamp(0.0, 1.0)
    source_height, source_width = int(source_rgb.shape[-2]), int(source_rgb.shape[-1])
    source_mask_nchw = _geometry_optional_mask_nchw(
        source_mask, "source_mask", source_height, source_width
    )
    if source_mask_nchw is None:
        if source_alpha is not None:
            source_mask_nchw = source_alpha.clamp(0.0, 1.0)
        else:
            source_mask_nchw = (
                source_rgb.amax(dim=1, keepdim=True)
                > float(settings.background_threshold)
            ).float()
    target_mask_nchw = _geometry_optional_mask_nchw(
        target_mask, "target_mask", target_height, target_width
    )
    if target_mask_nchw is None:
        target_gray_raw = _geometry_gray_nchw(target_depth_nchw)
        target_mask_nchw = (
            target_gray_raw > float(settings.background_threshold)
        ).float()

    if tuple(source_rgb.shape[-2:]) != (target_height, target_width):
        source_rgb = torch.nn.functional.interpolate(
            source_rgb, size=(target_height, target_width), mode="bilinear", align_corners=False
        )
        source_depth_nchw = torch.nn.functional.interpolate(
            source_depth_nchw,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )
        source_mask_nchw = torch.nn.functional.interpolate(
            source_mask_nchw, size=(target_height, target_width), mode="nearest"
        )
    elif tuple(source_depth_nchw.shape[-2:]) != (target_height, target_width):
        source_depth_nchw = torch.nn.functional.interpolate(
            source_depth_nchw,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )

    if source_mask_nchw.sum() < 16 or target_mask_nchw.sum() < 16:
        raise ValueError("Geometry-Aware Align requires non-empty source and target masks")
    if source_rgb.is_cuda:
        device = source_rgb.device
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = source_rgb.device
    source_rgb = source_rgb.to(device)
    source_depth_nchw = source_depth_nchw.to(device)
    target_depth_nchw = target_depth_nchw.to(device)
    source_mask_nchw = source_mask_nchw.to(device)
    target_mask_nchw = target_mask_nchw.to(device)
    source_depth_normalized, source_low, source_high = _geometry_normalize_depth(
        source_depth_nchw,
        source_mask_nchw,
        settings.source_depth_near_is_white,
    )
    target_depth_normalized, target_low, target_high = _geometry_normalize_depth(
        target_depth_nchw,
        target_mask_nchw,
        settings.target_depth_near_is_white,
    )
    source_depth_normalized = _geometry_masked_blur(
        source_depth_normalized,
        source_mask_nchw,
        settings.source_depth_blur_pixels,
    )

    longest = max(target_height, target_width)
    analysis_scale = min(1.0, float(settings.analysis_size) / float(longest))
    analysis_height = max(32, int(round(target_height * analysis_scale)))
    analysis_width = max(32, int(round(target_width * analysis_scale)))
    source_depth_analysis = torch.nn.functional.interpolate(
        source_depth_normalized,
        size=(analysis_height, analysis_width),
        mode="bilinear",
        align_corners=False,
    )
    target_depth_analysis = torch.nn.functional.interpolate(
        target_depth_normalized,
        size=(analysis_height, analysis_width),
        mode="bilinear",
        align_corners=False,
    )
    source_mask_analysis = torch.nn.functional.interpolate(
        source_mask_nchw, size=(analysis_height, analysis_width), mode="nearest"
    )
    target_mask_analysis = torch.nn.functional.interpolate(
        target_mask_nchw, size=(analysis_height, analysis_width), mode="nearest"
    )
    base_grid_analysis, _analysis_transform = _geometry_base_grid(
        source_mask_analysis,
        target_mask_analysis,
        settings.global_fit_mode,
        settings.global_alignment_strength,
    )
    boundary_width = float(settings.boundary_lock_pixels) * analysis_scale
    if boundary_width > 1e-6:
        interior_gate = _inner_edge_feather_mask(target_mask_analysis, boundary_width)
        interior_gate = ((interior_gate - 0.5) * 2.0).clamp(0.0, 1.0)
    else:
        # Do not cut the displacement field off at the silhouette.  A hard
        # target-mask multiplication creates a flow discontinuity exactly where
        # arms and other outline parts must move, and can smear their texture.
        interior_gate = torch.ones_like(target_mask_analysis)
    target_edge = _geometry_edge_magnitude(target_depth_analysis)
    max_shift_analysis = max(0.0, float(settings.max_local_shift_pixels) * analysis_scale)
    grid_size = min(
        int(settings.control_grid_size), analysis_height, analysis_width
    )
    raw_flow = torch.zeros(
        (1, 2, grid_size, grid_size), device=device, dtype=torch.float32, requires_grad=True
    )

    def evaluated_flow() -> torch.Tensor:
        if max_shift_analysis <= 1e-6:
            return torch.zeros(
                (1, 2, analysis_height, analysis_width), device=device, dtype=torch.float32
            )
        control = torch.tanh(raw_flow) * max_shift_analysis
        dense = torch.nn.functional.interpolate(
            control,
            size=(analysis_height, analysis_width),
            mode="bicubic",
            align_corners=False,
        )
        return dense * interior_gate

    def alignment_loss(flow: torch.Tensor):
        grid = _geometry_grid_with_flow(base_grid_analysis, flow)
        warped_depth = _geometry_warp(source_depth_analysis, grid)
        warped_mask = _geometry_warp(source_mask_analysis, grid).clamp(0.0, 1.0)
        overlap = target_mask_analysis * warped_mask
        denominator = overlap.sum().clamp_min(1.0)
        residual = warped_depth - target_depth_analysis
        depth_loss = (torch.sqrt(residual.square() + 1e-6) * overlap).sum() / denominator
        warped_edge = _geometry_edge_magnitude(warped_depth)
        edge_loss = (
            (warped_edge - target_edge).abs() * target_mask_analysis
        ).sum() / target_mask_analysis.sum().clamp_min(1.0)
        # Full-image averaging made small limbs nearly irrelevant on padded
        # 1024/2048 canvases.  Union normalisation gives hands, ears and feet a
        # useful influence independent of the amount of surrounding padding.
        silhouette_error = (warped_mask - target_mask_analysis).abs()
        silhouette_union = torch.maximum(
            warped_mask.detach(), target_mask_analysis
        ).sum().clamp_min(1.0)
        silhouette_loss = silhouette_error.sum() / silhouette_union
        regularization, fold_penalty, _determinant = _geometry_deformation_penalties(
            flow,
            _analysis_transform["scale_x"],
            _analysis_transform["scale_y"],
            max_shift_analysis,
        )
        total = (
            float(settings.depth_weight) * depth_loss
            + float(settings.edge_weight) * edge_loss
            + float(settings.silhouette_weight) * silhouette_loss
            + float(settings.rigidity) * regularization
            + float(settings.fold_prevention_strength) * fold_penalty
        )
        return (
            total,
            depth_loss,
            edge_loss,
            silhouette_loss,
            regularization,
            fold_penalty,
            warped_depth,
            warped_mask,
        )

    with torch.enable_grad():
        initial = alignment_loss(evaluated_flow())
        initial_loss = float(initial[0].detach())
        if (
            int(settings.optimization_steps) > 0
            and float(settings.local_alignment_strength) > 0.0
            and max_shift_analysis > 1e-6
        ):
            optimizer = torch.optim.Adam((raw_flow,), lr=0.035)
            for _ in range(int(settings.optimization_steps)):
                optimizer.zero_grad(set_to_none=True)
                loss = alignment_loss(evaluated_flow())[0]
                loss.backward()
                torch.nn.utils.clip_grad_norm_((raw_flow,), 5.0)
                optimizer.step()
        optimized_flow = evaluated_flow()
        final_terms = alignment_loss(optimized_flow)

    with torch.no_grad():
        final_loss = float(final_terms[0])
        warped_depth_analysis = final_terms[6]
        warped_mask_analysis = final_terms[7]
        residual = (warped_depth_analysis - target_depth_analysis).abs()
        confidence = (
            torch.exp(-6.0 * residual)
            * target_mask_analysis
            * warped_mask_analysis
        ).clamp(0.0, 1.0)
        confidence = _gaussian_blur_nchw(confidence, 1.0).clamp(0.0, 1.0)
        threshold = min(0.95, max(0.0, float(settings.confidence_threshold)))
        if threshold > 0.0:
            confidence = ((confidence - threshold) / (1.0 - threshold)).clamp(0.0, 1.0)
        gate_strength = min(
            1.0, max(0.0, float(settings.confidence_gate_strength))
        )
        # Confidence is primarily diagnostic.  The old unconditional
        # multiplication suppressed precisely the large-mismatch areas that
        # needed deformation most.  This optional gate blends continuously
        # from full solved flow (0) to the old conservative behaviour (1).
        confidence_gate = 1.0 - gate_strength * (1.0 - confidence)
        effective_flow_analysis = (
            optimized_flow
            * min(1.0, max(0.0, float(settings.local_alignment_strength)))
            * confidence_gate
        )
        flow_full = torch.nn.functional.interpolate(
            effective_flow_analysis,
            size=(target_height, target_width),
            mode="bicubic",
            align_corners=False,
        )
        flow_full[:, 0] *= float(target_width) / float(analysis_width)
        flow_full[:, 1] *= float(target_height) / float(analysis_height)
        confidence_full = torch.nn.functional.interpolate(
            confidence,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
        base_grid_full, transform = _geometry_base_grid(
            source_mask_nchw,
            target_mask_nchw,
            settings.global_fit_mode,
            settings.global_alignment_strength,
        )
        final_grid = _geometry_grid_with_flow(base_grid_full, flow_full)
        aligned_rgb = _geometry_warp(source_rgb, final_grid).clamp(0.0, 1.0)
        aligned_source_mask = _geometry_warp(source_mask_nchw, final_grid).clamp(0.0, 1.0)
        aligned_depth = _geometry_warp(source_depth_normalized, final_grid).clamp(0.0, 1.0)
        if settings.output_mask_mode == "target":
            output_mask = target_mask_nchw
        elif settings.output_mask_mode == "intersection":
            output_mask = target_mask_nchw * aligned_source_mask
        elif settings.output_mask_mode == "warped_source":
            output_mask = aligned_source_mask
        else:
            raise ValueError(f"Unknown geometry output mask mode: {settings.output_mask_mode!r}")
        output_mask = output_mask.clamp(0.0, 1.0)
        aligned_rgb = aligned_rgb * output_mask
        aligned_rgba = torch.cat((aligned_rgb, output_mask), dim=1)
        aligned_depth_image = aligned_depth.repeat(1, 3, 1, 1) * output_mask
        max_shift = max(float(settings.max_local_shift_pixels), 1.0)
        flow_preview = torch.stack(
            (
                (0.5 + flow_full[:, 0] / (2.0 * max_shift)).clamp(0.0, 1.0),
                (0.5 + flow_full[:, 1] / (2.0 * max_shift)).clamp(0.0, 1.0),
                confidence_full[:, 0],
            ),
            dim=1,
        ) * target_mask_nchw
        warped_edge_full = _geometry_edge_magnitude(aligned_depth)
        target_edge_full = _geometry_edge_magnitude(target_depth_normalized)
        overlay = (aligned_rgb * 0.35).clamp(0.0, 1.0)
        overlay[:, 0:1] = torch.maximum(overlay[:, 0:1], warped_edge_full)
        overlay[:, 1:2] = torch.maximum(overlay[:, 1:2], target_edge_full)
        overlay[:, 2:3] = torch.maximum(overlay[:, 2:3], warped_edge_full)
        flow_magnitude = torch.linalg.vector_norm(flow_full, dim=1)
        _elastic_full, _fold_full, determinant_full = _geometry_deformation_penalties(
            flow_full,
            transform["scale_x"],
            transform["scale_y"],
            float(settings.max_local_shift_pixels),
        )
        active = target_mask_nchw[:, 0] > 0.5
        mean_flow = float(flow_magnitude[active].mean()) if torch.any(active) else 0.0
        max_flow = float(flow_magnitude.max())
        mean_confidence = float(confidence_full[:, 0][active].mean()) if torch.any(active) else 0.0
        target_coverage = float(
            (aligned_source_mask * target_mask_nchw).sum()
            / target_mask_nchw.sum().clamp_min(1.0)
        )
        source_mask_area = float((source_mask_nchw > 0.5).float().mean())
        target_mask_area = float((target_mask_nchw > 0.5).float().mean())
        determinant_active = determinant_full[active[:, :-1, :-1]]
        folded_fraction = (
            float((determinant_active <= 0.0).float().mean())
            if determinant_active.numel()
            else 0.0
        )
        mask_warning = (
            "WARNING: source_mask covers over 85% of the frame; verify that 1=object, not transparent background"
            if source_mask_area > 0.85
            else "source_mask polarity looks plausible"
        )
        report = "\n".join(
            (
                "Pixal3D geometry-aware view alignment",
                f"view={settings.view}, output={target_width}x{target_height}, analysis={analysis_width}x{analysis_height}, grid={grid_size}x{grid_size}",
                f"global_fit={settings.global_fit_mode}, strength={settings.global_alignment_strength:.3f}, scale={transform['scale_x']:.5f}/{transform['scale_y']:.5f}, offset={transform['offset_x']:.2f}/{transform['offset_y']:.2f}px",
                f"local_strength={settings.local_alignment_strength:.3f}, max_shift={settings.max_local_shift_pixels:.2f}px, boundary_lock={settings.boundary_lock_pixels:.2f}px, steps={settings.optimization_steps}",
                f"loss={initial_loss:.6f}->{final_loss:.6f}, depth={float(final_terms[1]):.6f}, edge={float(final_terms[2]):.6f}, silhouette={float(final_terms[3]):.6f}",
                f"deformation: elastic={float(final_terms[4]):.6f}, fold_penalty={float(final_terms[5]):.6f}, folded_pixels={folded_fraction:.4%}",
                f"flow_mean={mean_flow:.3f}px, flow_max={max_flow:.3f}px, target_coverage={target_coverage:.2%}",
                f"confidence_mean={mean_confidence:.4f}, confidence_threshold={settings.confidence_threshold:.3f}, confidence_gate_strength={gate_strength:.3f}",
                f"mask_area: source={source_mask_area:.2%}, target={target_mask_area:.2%}; {mask_warning}",
                f"source_depth_range={source_low:.5f}..{source_high:.5f}, target_depth_range={target_low:.5f}..{target_high:.5f}, output_mask={settings.output_mask_mode}",
                "overlay: warped source depth edges=magenta, target model depth edges=green; white means agreement",
            )
        )
    return (
        aligned_rgb.movedim(1, -1).cpu(),
        aligned_rgba.movedim(1, -1).cpu(),
        output_mask[:, 0].cpu(),
        aligned_depth_image.movedim(1, -1).cpu(),
        confidence_full[:, 0].cpu(),
        flow_preview.movedim(1, -1).cpu(),
        overlay.movedim(1, -1).cpu(),
        report,
    )


_HUMANOID_BODY_NAMES = (
    "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist", "right_hip", "right_knee",
    "right_ankle", "left_hip", "left_knee", "left_ankle", "right_eye",
    "left_eye", "right_ear", "left_ear",
)
_HUMANOID_BODY_GROUPS = (
    "head", "torso", "arms", "arms", "arms", "arms", "arms", "arms",
    "torso", "legs", "legs", "torso", "legs", "legs", "head", "head",
    "head", "head",
)


def _humanoid_pose_person(frame, requested_index: int, threshold: float, label: str):
    if not isinstance(frame, dict):
        raise ValueError(f"{label} POSE_KEYPOINT frame must be a dictionary")
    people = frame.get("people") or []
    if not people:
        raise ValueError(f"{label} POSE_KEYPOINT contains no detected people")
    index = int(requested_index)
    if index >= 0:
        if index >= len(people):
            raise ValueError(
                f"{label} person_index={index} is outside the {len(people)} detected people"
            )
        return people[index], index

    best_index = 0
    best_area = -1.0
    for candidate_index, person in enumerate(people):
        raw = person.get("pose_keypoints_2d") or []
        if len(raw) < 3:
            continue
        points = torch.as_tensor(raw, dtype=torch.float32).reshape(-1, 3)
        valid = (
            (points[:, 2] >= float(threshold))
            & (points[:, 0] >= 0.0)
            & (points[:, 1] >= 0.0)
        )
        selected = points[valid, :2]
        if selected.shape[0] < 2:
            area = 0.0
        else:
            span = selected.amax(dim=0) - selected.amin(dim=0)
            area = float(span[0] * span[1])
        if area > best_area:
            best_area = area
            best_index = candidate_index
    return people[best_index], best_index


def _humanoid_pose_array(person, key: str, count: int) -> torch.Tensor:
    raw = person.get(key) or []
    if not raw:
        return torch.zeros((count, 3), dtype=torch.float32)
    value = torch.as_tensor(raw, dtype=torch.float32).reshape(-1, 3)
    if value.shape[0] < count:
        value = torch.cat((value, value.new_zeros((count - value.shape[0], 3))), dim=0)
    return value[:count]


def _humanoid_pose_frame(value, label: str):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(
            f"{label} must be POSE_KEYPOINT from ComfyUI SDPose Keypoint Extractor"
        )
    frame = value[0]
    width = int(frame.get("canvas_width", 0)) if isinstance(frame, dict) else 0
    height = int(frame.get("canvas_height", 0)) if isinstance(frame, dict) else 0
    if width <= 0 or height <= 0:
        raise ValueError(f"{label} POSE_KEYPOINT has an invalid canvas size")
    return frame, width, height


def _humanoid_common_landmarks(
    source_pose,
    target_pose,
    settings: HumanoidAlignSettings,
):
    source_frame, source_width, source_height = _humanoid_pose_frame(
        source_pose, "source_pose"
    )
    target_frame, target_width, target_height = _humanoid_pose_frame(
        target_pose, "target_pose"
    )
    source_person, source_index = _humanoid_pose_person(
        source_frame,
        settings.source_person_index,
        settings.score_threshold,
        "source",
    )
    target_person, target_index = _humanoid_pose_person(
        target_frame,
        settings.target_person_index,
        settings.score_threshold,
        "target",
    )
    threshold = float(settings.score_threshold)
    source_points = []
    target_points = []
    names = []
    groups = []

    def append_common(source_array, target_array, point_names, point_groups, indices=None):
        use_indices = range(len(point_names)) if indices is None else indices
        for index in use_indices:
            source_point = source_array[index]
            target_point = target_array[index]
            valid = (
                float(source_point[2]) >= threshold
                and float(target_point[2]) >= threshold
                and bool(torch.isfinite(source_point).all())
                and bool(torch.isfinite(target_point).all())
                and float(source_point[0]) >= 0.0
                and float(source_point[1]) >= 0.0
                and float(target_point[0]) >= 0.0
                and float(target_point[1]) >= 0.0
            )
            if valid:
                source_points.append(
                    torch.tensor(
                        (
                            float(source_point[0]) / max(1.0, source_width - 1.0),
                            float(source_point[1]) / max(1.0, source_height - 1.0),
                        )
                    )
                )
                target_points.append(
                    torch.tensor(
                        (
                            float(target_point[0]) / max(1.0, target_width - 1.0),
                            float(target_point[1]) / max(1.0, target_height - 1.0),
                        )
                    )
                )
                names.append(point_names[index])
                groups.append(point_groups[index])

    source_body = _humanoid_pose_array(source_person, "pose_keypoints_2d", 18)
    target_body = _humanoid_pose_array(target_person, "pose_keypoints_2d", 18)
    append_common(
        source_body,
        target_body,
        _HUMANOID_BODY_NAMES,
        _HUMANOID_BODY_GROUPS,
    )

    foot_names = tuple(f"foot_{index}" for index in range(6))
    foot_groups = ("feet",) * 6
    append_common(
        _humanoid_pose_array(source_person, "foot_keypoints_2d", 6),
        _humanoid_pose_array(target_person, "foot_keypoints_2d", 6),
        foot_names,
        foot_groups,
    )

    # A compact hand subset keeps fingers from over-constraining the body warp.
    hand_indices = (5, 9, 13, 17, 8, 12, 16, 20)
    for side, key in (
        ("right", "hand_right_keypoints_2d"),
        ("left", "hand_left_keypoints_2d"),
    ):
        hand_names = tuple(f"{side}_hand_{index}" for index in range(21))
        append_common(
            _humanoid_pose_array(source_person, key, 21),
            _humanoid_pose_array(target_person, key, 21),
            hand_names,
            ("hands",) * 21,
            hand_indices,
        )

    if len(source_points) < 3:
        raise ValueError(
            "Humanoid Pose Align needs at least three common source/target body "
            "keypoints. Lower score_threshold or improve the target character render."
        )
    source = torch.stack(source_points).float()
    target = torch.stack(target_points).float()
    return source, target, names, groups, source_index, target_index


def _humanoid_fit_affine(
    target: torch.Tensor,
    source: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """Return [3,2] coefficients mapping normalized target points to source."""
    if mode == "affine":
        design = torch.cat((torch.ones((target.shape[0], 1)), target), dim=1)
        return torch.linalg.lstsq(design, source).solution[:3]
    if mode != "similarity":
        raise ValueError(f"Unknown humanoid global model: {mode!r}")
    target_mean = target.mean(dim=0)
    source_mean = source.mean(dim=0)
    target_centered = target - target_mean
    source_centered = source - source_mean
    u, _singular, vh = torch.linalg.svd(target_centered.T @ source_centered)
    rotation = u @ vh
    if float(torch.linalg.det(rotation)) < 0.0:
        u = u.clone()
        u[:, -1] *= -1.0
        rotation = u @ vh
    rotated = target_centered @ rotation
    scale = float(
        (rotated * source_centered).sum()
        / target_centered.square().sum().clamp_min(1e-8)
    )
    linear = rotation * scale
    offset = source_mean - target_mean @ linear
    return torch.cat((offset.view(1, 2), linear), dim=0)


def _humanoid_apply_affine(points: torch.Tensor, coefficients: torch.Tensor):
    return torch.cat((torch.ones((points.shape[0], 1)), points), dim=1) @ coefficients


def _humanoid_mask_base_coefficients(
    source_mask: torch.Tensor,
    target_mask: torch.Tensor,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    mode: str,
    strength: float,
) -> torch.Tensor:
    """Stable target->source affine fit derived from complete silhouettes.

    Pose points are deliberately not used for global framing: a side-view detector
    can return only a face or a foot, whose tiny bounding box would otherwise crop
    and magnify the entire character before TPS even starts.
    """
    sx0, sy0, sx1, sy1 = _geometry_mask_bbox(source_mask)
    tx0, ty0, tx1, ty1 = _geometry_mask_bbox(target_mask)
    source_bbox_width = max(1.0, sx1 - sx0 + 1.0)
    source_bbox_height = max(1.0, sy1 - sy0 + 1.0)
    target_bbox_width = max(1.0, tx1 - tx0 + 1.0)
    target_bbox_height = max(1.0, ty1 - ty0 + 1.0)
    if mode == "similarity":
        uniform = min(
            target_bbox_width / source_bbox_width,
            target_bbox_height / source_bbox_height,
        )
        scale_x = scale_y = uniform
    elif mode == "affine":
        scale_x = target_bbox_width / source_bbox_width
        scale_y = target_bbox_height / source_bbox_height
    else:
        raise ValueError(f"Unknown humanoid global model: {mode!r}")
    source_center_x = 0.5 * (sx0 + sx1)
    source_center_y = 0.5 * (sy0 + sy1)
    target_center_x = 0.5 * (tx0 + tx1)
    target_center_y = 0.5 * (ty0 + ty1)
    coefficients = torch.tensor(
        [
            [
                (source_center_x - target_center_x / scale_x)
                / max(1.0, source_width - 1.0),
                (source_center_y - target_center_y / scale_y)
                / max(1.0, source_height - 1.0),
            ],
            [
                max(1.0, target_width - 1.0)
                / (scale_x * max(1.0, source_width - 1.0)),
                0.0,
            ],
            [
                0.0,
                max(1.0, target_height - 1.0)
                / (scale_y * max(1.0, source_height - 1.0)),
            ],
        ],
        dtype=torch.float32,
    )
    identity = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32
    )
    amount = min(1.0, max(0.0, float(strength)))
    return identity + amount * (coefficients - identity)


def _humanoid_tps_solve(
    controls: torch.Tensor,
    residuals: torch.Tensor,
    regularization: float,
):
    count = controls.shape[0]
    squared = torch.cdist(controls, controls).square()
    kernel = squared * torch.log(squared.clamp_min(1e-8))
    kernel.diagonal().zero_()
    affine = torch.cat((torch.ones((count, 1)), controls), dim=1)
    system = torch.zeros((count + 3, count + 3), dtype=torch.float32)
    system[:count, :count] = kernel
    system[:count, :count] += torch.eye(count) * max(0.0, float(regularization))
    system[:count, count:] = affine
    system[count:, :count] = affine.T
    right = torch.cat((residuals, torch.zeros((3, 2))), dim=0)
    try:
        solution = torch.linalg.solve(system, right)
    except RuntimeError:
        solution = torch.linalg.lstsq(system, right).solution
    return solution[:count], solution[count:]


def _humanoid_tps_evaluate(
    points: torch.Tensor,
    controls: torch.Tensor,
    radial_weights: torch.Tensor,
    affine_weights: torch.Tensor,
):
    squared = torch.cdist(points, controls).square()
    kernel = squared * torch.log(squared.clamp_min(1e-8))
    design = torch.cat((torch.ones((points.shape[0], 1)), points), dim=1)
    return kernel @ radial_weights + design @ affine_weights


def _humanoid_frame_anchors(amount: float) -> torch.Tensor:
    if float(amount) <= 1e-6:
        return torch.empty((0, 2), dtype=torch.float32)
    per_edge = 2 + int(round(4.0 * min(1.0, max(0.0, float(amount)))))
    values = torch.linspace(0.0, 1.0, per_edge + 2)
    points = []
    for value in values:
        scalar = float(value)
        points.extend(((scalar, 0.0), (scalar, 1.0)))
    for value in values[1:-1]:
        scalar = float(value)
        points.extend(((0.0, scalar), (1.0, scalar)))
    return torch.tensor(points, dtype=torch.float32)


def _humanoid_mapping_grid(
    target_height: int,
    target_width: int,
    source_height: int,
    source_width: int,
    base_coefficients: torch.Tensor,
    controls: torch.Tensor,
    radial_weights: torch.Tensor,
    affine_weights: torch.Tensor,
    max_local_shift_pixels: float,
):
    rows = []
    x_values = torch.linspace(0.0, 1.0, target_width)
    for start in range(0, target_height, 128):
        end = min(target_height, start + 128)
        y_values = torch.linspace(0.0, 1.0, target_height)[start:end]
        yy, xx = torch.meshgrid(y_values, x_values, indexing="ij")
        target_points = torch.stack((xx, yy), dim=-1).reshape(-1, 2)
        base = _humanoid_apply_affine(target_points, base_coefficients)
        residual = _humanoid_tps_evaluate(
            target_points, controls, radial_weights, affine_weights
        )
        if float(max_local_shift_pixels) > 0.0:
            residual_pixels = residual * residual.new_tensor(
                (max(1, source_width - 1), max(1, source_height - 1))
            )
            magnitude = torch.linalg.vector_norm(residual_pixels, dim=1, keepdim=True)
            residual *= (
                float(max_local_shift_pixels) / magnitude.clamp_min(1e-8)
            ).clamp(max=1.0)
        rows.append(torch.cat((base, residual), dim=1).reshape(end - start, target_width, 4))
    packed = torch.cat(rows, dim=0)
    return packed[..., :2], packed[..., 2:]


def _humanoid_fold_fraction(
    source_coordinates: torch.Tensor,
    target_mask: torch.Tensor,
    source_width: int,
    source_height: int,
):
    source_x = source_coordinates[..., 0] * max(1, source_width - 1)
    source_y = source_coordinates[..., 1] * max(1, source_height - 1)
    dx_x = source_x[:-1, 1:] - source_x[:-1, :-1]
    dy_x = source_y[:-1, 1:] - source_y[:-1, :-1]
    dx_y = source_x[1:, :-1] - source_x[:-1, :-1]
    dy_y = source_y[1:, :-1] - source_y[:-1, :-1]
    determinant = dx_x * dy_y - dx_y * dy_x
    active = target_mask[0, 0, :-1, :-1] > 0.05
    if not bool(active.any()):
        active = torch.ones_like(determinant, dtype=torch.bool)
    folded = float((determinant[active] <= 0.0).float().mean())
    return folded, determinant


def _humanoid_landmark_overlay(
    target_reference: torch.Tensor,
    target_points: torch.Tensor,
    source_points: torch.Tensor,
    base_coefficients: torch.Tensor,
):
    import numpy as np
    from PIL import Image, ImageDraw

    image = target_reference[0, ..., :3].detach().float().cpu().clamp(0.0, 1.0)
    array = (image.numpy() * 255.0 + 0.5).astype(np.uint8)
    canvas = Image.fromarray(array, mode="RGB")
    draw = ImageDraw.Draw(canvas)
    height, width = array.shape[:2]
    linear = base_coefficients[1:]
    try:
        inverse = torch.linalg.inv(linear)
        initial_target = (source_points - base_coefficients[0]) @ inverse
    except RuntimeError:
        initial_target = target_points
    radius = max(2, int(round(max(height, width) / 256.0)))
    for predicted, target in zip(initial_target, target_points):
        px = float(predicted[0]) * max(1, width - 1)
        py = float(predicted[1]) * max(1, height - 1)
        tx = float(target[0]) * max(1, width - 1)
        ty = float(target[1]) * max(1, height - 1)
        draw.line((px, py, tx, ty), fill=(255, 210, 0), width=max(1, radius // 2))
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=(255, 0, 255))
        draw.ellipse((tx - radius, ty - radius, tx + radius, ty + radius), fill=(0, 255, 96))
    output = torch.from_numpy(np.asarray(canvas, dtype=np.float32).copy()).div_(255.0)
    return output.unsqueeze(0)


def humanoid_pose_view_align(
    image: torch.Tensor,
    source_pose,
    target_reference: torch.Tensor,
    target_pose,
    settings: HumanoidAlignSettings,
    source_mask: torch.Tensor | None = None,
    target_mask: torch.Tensor | None = None,
):
    """Warp a character view from explicit source/target anatomical landmarks."""
    if not isinstance(settings, HumanoidAlignSettings):
        raise ValueError("settings must come from Pixal3D Humanoid Align Settings")
    source_nchw = _geometry_single_nchw(image, "image")
    reference_nchw = _geometry_single_nchw(target_reference, "target_reference")
    if source_nchw.shape[1] < 3 or reference_nchw.shape[1] < 3:
        raise ValueError("Humanoid Pose Align requires RGB source and target_reference images")
    source_rgb = source_nchw[:, :3].clamp(0.0, 1.0)
    source_height, source_width = int(source_rgb.shape[-2]), int(source_rgb.shape[-1])
    target_height, target_width = int(reference_nchw.shape[-2]), int(reference_nchw.shape[-1])
    source_mask_nchw = _geometry_optional_mask_nchw(
        source_mask, "source_mask", source_height, source_width
    )
    if source_mask_nchw is None:
        source_mask_nchw = (
            source_nchw[:, 3:4].clamp(0.0, 1.0)
            if source_nchw.shape[1] >= 4
            else (source_rgb.amax(dim=1, keepdim=True) > settings.background_threshold).float()
        )
    target_mask_nchw = _geometry_optional_mask_nchw(
        target_mask, "target_mask", target_height, target_width
    )
    if target_mask_nchw is None:
        target_mask_nchw = (
            reference_nchw[:, 3:4].clamp(0.0, 1.0)
            if reference_nchw.shape[1] >= 4
            else (
                reference_nchw[:, :3].amax(dim=1, keepdim=True)
                > settings.background_threshold
            ).float()
        )
    source_points, target_points, names, groups, source_person, target_person = (
        _humanoid_common_landmarks(source_pose, target_pose, settings)
    )
    counts = {
        name: groups.count(name)
        for name in ("head", "torso", "arms", "legs", "hands", "feet")
    }
    if settings.view in ("left", "right"):
        anatomy_valid = counts["torso"] >= 2 and counts["arms"] >= 2 and counts["legs"] >= 2
    else:
        anatomy_valid = counts["torso"] >= 2 and counts["arms"] >= 4 and counts["legs"] >= 4
    base_coefficients = _humanoid_mask_base_coefficients(
        source_mask_nchw,
        target_mask_nchw,
        source_width,
        source_height,
        target_width,
        target_height,
        settings.global_model,
        settings.global_alignment_strength,
    )
    base_at_landmarks = _humanoid_apply_affine(target_points, base_coefficients)
    group_strength = {
        "torso": settings.torso_strength,
        "arms": settings.arm_strength,
        "legs": settings.leg_strength,
        "head": settings.head_strength,
        "hands": settings.hand_strength,
        "feet": settings.foot_strength,
    }
    strengths = torch.tensor([float(group_strength[group]) for group in groups]).unsqueeze(1)
    if not anatomy_valid:
        # A partial face/foot hallucination is not an anatomical target.  Keep
        # the full silhouette fit but disable local TPS instead of distorting
        # or cropping the source around the few detected points.
        strengths.zero_()
    landmark_residuals = (source_points - base_at_landmarks) * strengths
    frame_anchors = _humanoid_frame_anchors(settings.outer_stability)
    if frame_anchors.numel():
        controls = torch.cat((target_points, frame_anchors), dim=0)
        residuals = torch.cat(
            (landmark_residuals, torch.zeros_like(frame_anchors)), dim=0
        )
    else:
        controls = target_points
        residuals = landmark_residuals
    radial_weights, affine_weights = _humanoid_tps_solve(
        controls, residuals, settings.tps_regularization
    )
    base_map, residual_map = _humanoid_mapping_grid(
        target_height,
        target_width,
        source_height,
        source_width,
        base_coefficients,
        controls,
        radial_weights,
        affine_weights,
        settings.max_local_shift_pixels,
    )
    applied_local_scale = 1.0
    source_coordinates = base_map + residual_map
    folded_fraction, determinant = _humanoid_fold_fraction(
        source_coordinates, target_mask_nchw, source_width, source_height
    )
    allowed_fold = float(settings.max_folded_area_percent) / 100.0
    if settings.fold_protection:
        while folded_fraction > allowed_fold and applied_local_scale > 0.05:
            applied_local_scale *= 0.82
            source_coordinates = base_map + residual_map * applied_local_scale
            folded_fraction, determinant = _humanoid_fold_fraction(
                source_coordinates, target_mask_nchw, source_width, source_height
            )

    grid = torch.empty((1, target_height, target_width, 2), dtype=torch.float32)
    grid[0, ..., 0] = (
        (source_coordinates[..., 0] * max(1, source_width - 1) + 0.5)
        / float(source_width)
        * 2.0
        - 1.0
    )
    grid[0, ..., 1] = (
        (source_coordinates[..., 1] * max(1, source_height - 1) + 0.5)
        / float(source_height)
        * 2.0
        - 1.0
    )
    device = source_rgb.device
    grid_device = grid.to(device)
    aligned_rgb = _geometry_warp(source_rgb, grid_device).clamp(0.0, 1.0)
    aligned_source_mask = _geometry_warp(source_mask_nchw.to(device), grid_device).clamp(0.0, 1.0)
    target_mask_device = target_mask_nchw.to(device)
    if settings.output_mask_mode == "target":
        output_mask = target_mask_device
    elif settings.output_mask_mode == "intersection":
        output_mask = target_mask_device * aligned_source_mask
    elif settings.output_mask_mode == "warped_source":
        output_mask = aligned_source_mask
    else:
        raise ValueError(f"Unknown humanoid output mask mode: {settings.output_mask_mode!r}")
    output_mask = output_mask.clamp(0.0, 1.0)
    aligned_rgb = aligned_rgb * output_mask
    aligned_rgba = torch.cat((aligned_rgb, output_mask), dim=1)

    displacement_pixels = residual_map.clone()
    displacement_pixels[..., 0] *= max(1, source_width - 1) * applied_local_scale
    displacement_pixels[..., 1] *= max(1, source_height - 1) * applied_local_scale
    preview_limit = max(1.0, float(settings.max_local_shift_pixels))
    determinant_preview = torch.zeros((target_height, target_width))
    determinant_preview[:-1, :-1] = determinant
    positive_scale = torch.quantile(determinant_preview[determinant_preview > 0], 0.9) if bool((determinant_preview > 0).any()) else torch.tensor(1.0)
    deformation_preview = torch.stack(
        (
            (0.5 + displacement_pixels[..., 0] / (2.0 * preview_limit)).clamp(0.0, 1.0),
            (0.5 + displacement_pixels[..., 1] / (2.0 * preview_limit)).clamp(0.0, 1.0),
            (determinant_preview / positive_scale.clamp_min(1e-6)).clamp(0.0, 1.0),
        ),
        dim=-1,
    ).unsqueeze(0) * target_mask_nchw.movedim(1, -1)
    landmark_overlay = _humanoid_landmark_overlay(
        target_reference, target_points, source_points, base_coefficients
    )

    final_landmark_residuals = _humanoid_tps_evaluate(
        target_points, controls, radial_weights, affine_weights
    )
    if float(settings.max_local_shift_pixels) > 0.0:
        landmark_residual_pixels = final_landmark_residuals * source_points.new_tensor(
            (max(1, source_width - 1), max(1, source_height - 1))
        )
        landmark_magnitude = torch.linalg.vector_norm(
            landmark_residual_pixels, dim=1, keepdim=True
        )
        final_landmark_residuals *= (
            float(settings.max_local_shift_pixels) / landmark_magnitude.clamp_min(1e-8)
        ).clamp(max=1.0)
    mapped_landmarks = base_at_landmarks + final_landmark_residuals * applied_local_scale
    errors = (mapped_landmarks - source_points) * source_points.new_tensor(
        (max(1, source_width - 1), max(1, source_height - 1))
    )
    rms_error = float(torch.sqrt(errors.square().sum(dim=1).mean()))
    warnings = []
    if counts["arms"] < 4:
        warnings.append("WARNING: fewer than four arm joints matched; arms cannot be aligned reliably")
    if settings.hand_strength > 0.0 and counts["hands"] == 0:
        warnings.append("hands were not detected in both views; body wrists remain active")
    if not anatomy_valid:
        warnings.append(
            "SAFE FALLBACK: target anatomy is incomplete; local TPS was disabled and only silhouette framing was applied"
        )
    report_lines = [
        "Pixal3D humanoid pose view alignment",
        f"view={settings.view}, output={target_width}x{target_height}, global_model={settings.global_model}, global_strength={settings.global_alignment_strength:.3f}",
        f"people: source={source_person}, target={target_person}; common_landmarks={len(names)}, score_threshold={settings.score_threshold:.3f}",
        "landmarks: " + ", ".join(f"{key}={value}" for key, value in counts.items()),
        f"anatomy_valid={anatomy_valid}; global framing source=silhouette masks",
        f"strengths: torso={settings.torso_strength:.2f}, arms={settings.arm_strength:.2f}, legs={settings.leg_strength:.2f}, head={settings.head_strength:.2f}, hands={settings.hand_strength:.2f}, feet={settings.foot_strength:.2f}",
        f"TPS: regularization={settings.tps_regularization:.6f}, max_shift={settings.max_local_shift_pixels:.1f}px, outer_stability={settings.outer_stability:.2f}, applied_local_scale={applied_local_scale:.4f}",
        f"landmark_rms={rms_error:.3f}px, folded_area={folded_fraction:.4%}, output_mask={settings.output_mask_mode}",
        "overlay: magenta=source landmark after global fit, green=target landmark, yellow=required local correction",
    ]
    report_lines.extend(warnings)
    return (
        aligned_rgb.movedim(1, -1).cpu(),
        aligned_rgba.movedim(1, -1).cpu(),
        output_mask[:, 0].cpu(),
        deformation_preview.cpu(),
        landmark_overlay.cpu(),
        "\n".join(report_lines),
    )


def make_humanoid_pose_reference(
    depth_map: torch.Tensor,
    mesh_mask: torch.Tensor,
    relief_strength: float,
    edge_strength: float,
    background_level: float,
):
    """Turn a geometric depth map into a neutral character render for pose detection."""
    depth = _geometry_gray_nchw(_geometry_single_nchw(depth_map, "depth_map"))
    height, width = int(depth.shape[-2]), int(depth.shape[-1])
    mask = _geometry_optional_mask_nchw(mesh_mask, "mesh_mask", height, width)
    if mask is None:
        raise ValueError("mesh_mask is required")
    sobel_x = depth.new_tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]
    ).unsqueeze(0) / 8.0
    sobel_y = depth.new_tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]
    ).unsqueeze(0) / 8.0
    gradient_x = torch.nn.functional.conv2d(depth, sobel_x, padding=1)
    gradient_y = torch.nn.functional.conv2d(depth, sobel_y, padding=1)
    relief = max(0.0, float(relief_strength))
    normal = torch.cat(
        (-gradient_x * relief, -gradient_y * relief, torch.ones_like(depth)), dim=1
    )
    normal = torch.nn.functional.normalize(normal, dim=1)
    lights = normal.new_tensor(
        [[-0.45, -0.35, 0.82], [0.45, -0.25, 0.86], [0.0, 0.35, 0.94]]
    )
    lights = torch.nn.functional.normalize(lights, dim=1)
    light_values = torch.einsum("bchw,kc->bkhw", normal, lights).clamp_min(0.0)
    shading = 0.18 + 0.82 * light_values.mean(dim=1, keepdim=True)
    shading *= 0.72 + 0.28 * depth.clamp(0.0, 1.0)
    silhouette_edge = _geometry_edge_magnitude(mask)
    shading = (shading - silhouette_edge * max(0.0, float(edge_strength))).clamp(0.0, 1.0)
    background = min(1.0, max(0.0, float(background_level)))
    output = shading * mask + background * (1.0 - mask)
    return output.repeat(1, 3, 1, 1).movedim(1, -1).cpu()
_SAM3D_OPENPOSE18_TO_MHR70 = (
    0, 69, 6, 8, 41, 5, 7, 62, 10, 12, 14, 9, 11, 13, 2, 1, 4, 3,
)
_SAM3D_HAND21_TO_MHR70_RIGHT = (
    41, 24, 23, 22, 21, 28, 27, 26, 25, 32, 31, 30, 29, 36, 35, 34, 33,
    40, 39, 38, 37,
)
_SAM3D_HAND21_TO_MHR70_LEFT = (
    62, 45, 44, 43, 42, 49, 48, 47, 46, 53, 52, 51, 50, 57, 56, 55, 54,
    61, 60, 59, 58,
)
_OPENPOSE_BODY_EDGES = (
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
    (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
)


def _sam3d_select_person(pose_data, requested_index: int):
    if not isinstance(pose_data, dict) or not pose_data.get("frames"):
        raise ValueError("mhr_pose_data contains no frames")
    people = pose_data["frames"][0]
    if not people:
        raise ValueError("mhr_pose_data contains no detected people")
    index = int(requested_index)
    if index >= 0:
        if index >= len(people):
            raise ValueError(
                f"person_index={index} is outside the {len(people)} SAM3D people"
            )
        return people[index], index
    best_index = 0
    best_area = -1.0
    for candidate_index, person in enumerate(people):
        points = person.get("pred_keypoints_2d")
        if points is None:
            continue
        value = torch.as_tensor(points, dtype=torch.float32)
        finite = torch.isfinite(value).all(dim=1)
        value = value[finite]
        if value.shape[0] < 2:
            area = 0.0
        else:
            span = value.amax(dim=0) - value.amin(dim=0)
            area = float(span[0] * span[1])
        if area > best_area:
            best_area = area
            best_index = candidate_index
    return people[best_index], best_index


def _sam3d_project_points(
    points: torch.Tensor,
    source_view: str,
    target_view: str,
    target_center_x: float,
    target_center_y: float,
    scale: float,
    horizontal_scale: float,
    body_center_xz: torch.Tensor,
    body_center_y: float,
    mirror_side_projection: bool,
):
    centered = points.float().clone()
    centered[:, 0] -= body_center_xz[0]
    centered[:, 2] -= body_center_xz[1]
    centered[:, 1] -= float(body_center_y)
    delta = math.radians(
        VIEW_AZIMUTHS[VIEW_NAMES.index(target_view)]
        - VIEW_AZIMUTHS[VIEW_NAMES.index(source_view)]
    )
    cosine, sine = math.cos(delta), math.sin(delta)
    horizontal = cosine * centered[:, 0] + sine * centered[:, 2]
    if bool(mirror_side_projection) and target_view in ("left", "right"):
        horizontal = -horizontal
    x = float(target_center_x) + horizontal * float(scale) * float(horizontal_scale)
    y = float(target_center_y) + centered[:, 1] * float(scale)
    return torch.stack((x, y), dim=1)


def _sam3d_pose_preview(mask: torch.Tensor, body: torch.Tensor, scores: torch.Tensor):
    import numpy as np
    from PIL import Image, ImageDraw

    value = mask[0].detach().float().cpu().clamp(0.0, 1.0)
    background = (value.unsqueeze(-1).repeat(1, 1, 3) * 80.0).byte().numpy()
    canvas = Image.fromarray(background, mode="RGB")
    draw = ImageDraw.Draw(canvas)
    height, width = value.shape
    line_width = max(2, int(round(max(height, width) / 384.0)))
    radius = max(3, int(round(max(height, width) / 256.0)))
    for first, second in _OPENPOSE_BODY_EDGES:
        if float(scores[first]) <= 0.0 or float(scores[second]) <= 0.0:
            continue
        draw.line(
            (
                float(body[first, 0]), float(body[first, 1]),
                float(body[second, 0]), float(body[second, 1]),
            ),
            fill=(0, 220, 255),
            width=line_width,
        )
    for point, score in zip(body, scores):
        if float(score) <= 0.0:
            continue
        x, y = float(point[0]), float(point[1])
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 210, 0))
    return torch.from_numpy(np.asarray(canvas, dtype=np.float32).copy()).div_(255.0).unsqueeze(0)


def project_sam3d_humanoid_pose(
    mhr_pose_data,
    target_mask: torch.Tensor,
    source_view: str,
    target_view: str,
    skeleton_height_percent: float,
    horizontal_scale_percent: float,
    horizontal_offset_pixels: float,
    vertical_offset_pixels: float,
    mirror_side_projection: bool,
    include_hands: bool,
    person_index: int,
):
    """Project a reliable source SAM3D skeleton into any Pixal3D camera view."""
    if source_view not in VIEW_NAMES or target_view not in VIEW_NAMES:
        raise ValueError(f"source_view and target_view must be one of {VIEW_NAMES}")
    if not isinstance(target_mask, torch.Tensor) or target_mask.ndim != 3 or target_mask.shape[0] != 1:
        raise ValueError("target_mask must be a single-item MASK [1,H,W]")
    mask = target_mask.detach().float().cpu().clamp(0.0, 1.0)
    target_height, target_width = int(mask.shape[1]), int(mask.shape[2])
    mask_nchw = mask.unsqueeze(1)
    x0, y0, x1, y1 = _geometry_mask_bbox(mask_nchw)
    person, selected_person = _sam3d_select_person(mhr_pose_data, int(person_index))
    raw = person.get("pred_keypoints_3d")
    if raw is None:
        raise ValueError("Selected SAM3D person has no pred_keypoints_3d")
    keypoints = torch.as_tensor(raw, dtype=torch.float32).cpu()
    if keypoints.ndim != 2 or keypoints.shape[0] < 70 or keypoints.shape[1] < 3:
        raise ValueError(
            f"SAM3D pred_keypoints_3d must contain at least [70,3], got {tuple(keypoints.shape)}"
        )
    body_3d = keypoints[torch.tensor(_SAM3D_OPENPOSE18_TO_MHR70)]
    finite_body = torch.isfinite(body_3d).all(dim=1)
    if int(finite_body.sum()) < 14:
        raise ValueError("SAM3D body contains too few finite joints")
    hip_center = 0.5 * (body_3d[8] + body_3d[11])
    raw_y = body_3d[finite_body, 1]
    body_height = float(raw_y.amax() - raw_y.amin())
    if not math.isfinite(body_height) or body_height <= 1e-6:
        raise ValueError("SAM3D body has invalid vertical extent")
    target_bbox_height = max(1.0, y1 - y0 + 1.0)
    desired_height = target_bbox_height * min(1.2, max(0.5, float(skeleton_height_percent) / 100.0))
    scale = desired_height / body_height
    target_center_x = 0.5 * (x0 + x1) + float(horizontal_offset_pixels)
    target_center_y = 0.5 * (y0 + y1) + float(vertical_offset_pixels)
    horizontal_scale = min(2.0, max(0.25, float(horizontal_scale_percent) / 100.0))
    center_xz = torch.stack((hip_center[0], hip_center[2]))
    body_center_y = 0.5 * float(raw_y.amin() + raw_y.amax())
    body_2d = _sam3d_project_points(
        body_3d,
        source_view,
        target_view,
        target_center_x,
        target_center_y,
        scale,
        horizontal_scale,
        center_xz,
        body_center_y,
        bool(mirror_side_projection),
    )
    body_scores = finite_body.float()
    margin_x, margin_y = target_width * 0.25, target_height * 0.25
    in_canvas = (
        (body_2d[:, 0] >= -margin_x)
        & (body_2d[:, 0] <= target_width - 1 + margin_x)
        & (body_2d[:, 1] >= -margin_y)
        & (body_2d[:, 1] <= target_height - 1 + margin_y)
    )
    body_scores *= in_canvas.float()

    def project_hand(mapping):
        points_3d = keypoints[torch.tensor(mapping)]
        points_2d = _sam3d_project_points(
            points_3d,
            source_view,
            target_view,
            target_center_x,
            target_center_y,
            scale,
            horizontal_scale,
            center_xz,
            body_center_y,
            bool(mirror_side_projection),
        )
        scores = torch.isfinite(points_3d).all(dim=1).float()
        return points_2d, scores

    if bool(include_hands):
        right_hand, right_scores = project_hand(_SAM3D_HAND21_TO_MHR70_RIGHT)
        left_hand, left_scores = project_hand(_SAM3D_HAND21_TO_MHR70_LEFT)
    else:
        right_hand = left_hand = torch.zeros((21, 2), dtype=torch.float32)
        right_scores = left_scores = torch.zeros(21, dtype=torch.float32)

    def flatten(points, scores):
        return torch.cat((points, scores.unsqueeze(1)), dim=1).reshape(-1).tolist()

    person_output = {
        "pose_keypoints_2d": flatten(body_2d, body_scores),
        "foot_keypoints_2d": [0.0] * 18,
        "face_keypoints_2d": [0.0] * 210,
        "hand_right_keypoints_2d": flatten(right_hand, right_scores),
        "hand_left_keypoints_2d": flatten(left_hand, left_scores),
    }
    pose = [{
        "canvas_width": target_width,
        "canvas_height": target_height,
        "people": [person_output],
    }]
    preview = _sam3d_pose_preview(mask, body_2d, body_scores)
    report = "\n".join(
        (
            "Pixal3D SAM3D humanoid pose projection",
            f"source_view={source_view}, target_view={target_view}, output={target_width}x{target_height}, person={selected_person}",
            f"body_joints={int(body_scores.sum())}/18, include_hands={bool(include_hands)}",
            f"skeleton_height={float(skeleton_height_percent):.2f}%, horizontal_scale={float(horizontal_scale_percent):.2f}%",
            f"offset={float(horizontal_offset_pixels):.1f}/{float(vertical_offset_pixels):.1f}px, mirror_side={bool(mirror_side_projection)}",
            "preview: cyan=projected bones, yellow=projected joints, gray=target mesh mask",
        )
    )
    return pose, preview, report


def _rasterize_projected_triangles(
    x_pixel: torch.Tensor,
    y_pixel: torch.Tensor,
    depth: torch.Tensor,
    triangles: torch.Tensor,
    size: int,
) -> torch.Tensor:
    """Rasterize a binary silhouette; depth ordering is irrelevant for a union mask."""
    import numpy as np
    from PIL import Image, ImageDraw

    points = torch.stack([x_pixel, y_pixel], dim=-1).detach().float().cpu().numpy()
    depths = depth.detach().float().cpu().numpy()
    faces = triangles.detach().to(torch.int64).cpu().numpy()
    canvas = Image.new("L", (int(size), int(size)), 0)
    draw = ImageDraw.Draw(canvas)
    coordinate_limit = float(size) * 16.0
    for face in faces:
        face_depth = depths[face]
        polygon = points[face]
        if not np.isfinite(face_depth).all() or not np.isfinite(polygon).all():
            continue
        if (face_depth <= 0.0).any():
            continue
        if (
            polygon[:, 0].max() < 0.0
            or polygon[:, 1].max() < 0.0
            or polygon[:, 0].min() >= size
            or polygon[:, 1].min() >= size
        ):
            continue
        polygon = np.clip(polygon, -coordinate_limit, coordinate_limit)
        draw.polygon([(float(x), float(y)) for x, y in polygon], fill=255)
    return torch.from_numpy(np.asarray(canvas, dtype=np.float32).copy()).div_(255.0)


def _rasterize_projected_depth(
    x_pixel: torch.Tensor,
    y_pixel: torch.Tensor,
    depth: torch.Tensor,
    triangles: torch.Tensor,
    size: int,
) -> torch.Tensor:
    """Rasterize perspective-correct camera depth with a hard z-buffer."""
    width = height = int(size)
    xy = torch.stack([x_pixel, y_pixel], dim=-1).detach().float().cpu()
    vertex_depth = depth.detach().float().cpu()
    faces = triangles.detach().to(torch.int64).cpu()
    face_xy = xy[faces]
    face_depth = vertex_depth[faces]
    finite = torch.isfinite(face_xy).all(dim=(1, 2)) & torch.isfinite(face_depth).all(dim=1)
    in_front = (face_depth > 1e-6).all(dim=1)
    min_x = face_xy[..., 0].amin(dim=1).floor().long().clamp(0, width)
    max_x = (face_xy[..., 0].amax(dim=1).ceil().long() + 1).clamp(0, width)
    min_y = face_xy[..., 1].amin(dim=1).floor().long().clamp(0, height)
    max_y = (face_xy[..., 1].amax(dim=1).ceil().long() + 1).clamp(0, height)
    span_x = max_x - min_x
    span_y = max_y - min_y
    keep = torch.nonzero(
        finite & in_front & (span_x > 0) & (span_y > 0), as_tuple=False
    ).flatten()
    z_buffer = torch.full((width * height,), float("inf"), dtype=torch.float32)
    if keep.numel() == 0:
        return z_buffer.reshape(height, width)

    bbox_size = torch.maximum(span_x, span_y)
    pixel_budget = 1_000_000
    max_chunk = 4096
    lower = 0
    limit = 1
    max_extent = max(width, height)
    while lower < max_extent:
        candidates = keep[(bbox_size[keep] > lower) & (bbox_size[keep] <= limit)]
        chunk_size = min(max_chunk, max(1, pixel_budget // max(1, limit * limit)))
        for start in range(0, int(candidates.numel()), chunk_size):
            selected = candidates[start : start + chunk_size]
            if selected.numel() == 0:
                continue
            sx = span_x[selected]
            sy = span_y[selected]
            grid_width = int(sx.max())
            grid_height = int(sy.max())
            x_offset = torch.arange(grid_width)
            y_offset = torch.arange(grid_height)
            sample_x = min_x[selected, None, None].float() + x_offset[None, None, :].float() + 0.5
            sample_y = min_y[selected, None, None].float() + y_offset[None, :, None].float() + 0.5
            in_bbox = (
                (x_offset[None, None, :] < sx[:, None, None])
                & (y_offset[None, :, None] < sy[:, None, None])
            )

            vertices_2d = face_xy[selected]
            ax = vertices_2d[:, 0, 0, None, None]
            ay = vertices_2d[:, 0, 1, None, None]
            bx = vertices_2d[:, 1, 0, None, None]
            by = vertices_2d[:, 1, 1, None, None]
            cx = vertices_2d[:, 2, 0, None, None]
            cy = vertices_2d[:, 2, 1, None, None]
            area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
            edge_a = (bx - sample_x) * (cy - sample_y) - (by - sample_y) * (cx - sample_x)
            edge_b = (cx - sample_x) * (ay - sample_y) - (cy - sample_y) * (ax - sample_x)
            edge_c = (ax - sample_x) * (by - sample_y) - (ay - sample_y) * (bx - sample_x)
            nondegenerate = area.abs() > 1e-8
            inside = (
                (edge_a * area >= 0.0)
                & (edge_b * area >= 0.0)
                & (edge_c * area >= 0.0)
                & in_bbox
                & nondegenerate
            )
            face_index, pixel_y, pixel_x = inside.nonzero(as_tuple=True)
            if face_index.numel() == 0:
                continue
            inverse_area = torch.where(
                nondegenerate, area.reciprocal(), torch.zeros_like(area)
            )
            weight_a = edge_a * inverse_area
            weight_b = edge_b * inverse_area
            weight_c = edge_c * inverse_area
            depths = face_depth[selected]
            reciprocal_depth = (
                weight_a / depths[:, 0, None, None]
                + weight_b / depths[:, 1, None, None]
                + weight_c / depths[:, 2, None, None]
            )
            fragment_depth = reciprocal_depth[face_index, pixel_y, pixel_x].reciprocal()
            absolute_x = min_x[selected][face_index] + pixel_x
            absolute_y = min_y[selected][face_index] + pixel_y
            pixel_index = absolute_y * width + absolute_x
            z_buffer.scatter_reduce_(
                0, pixel_index.long(), fragment_depth.float(), reduce="amin", include_self=True
            )
        lower = limit
        limit = min(limit * 2, max_extent)
    return z_buffer.reshape(height, width)


def _voxel_projection_geometry(
    spatial_coords: torch.Tensor,
    resolution: int,
    target_size: int,
    fov: float,
    azimuth: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return projected cube vertices and triangles for sparse occupancy tokens."""
    offsets = torch.tensor(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [-0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
        ],
        dtype=torch.float32,
    )
    cube_triangles = torch.tensor(
        [
            [0, 2, 1], [1, 2, 3],
            [4, 5, 6], [5, 7, 6],
            [0, 1, 4], [1, 5, 4],
            [2, 6, 3], [3, 6, 7],
            [0, 4, 2], [2, 4, 6],
            [1, 3, 5], [3, 7, 5],
        ],
        dtype=torch.int64,
    )
    cube_coords = spatial_coords.detach().float().cpu().unsqueeze(1) + offsets.unsqueeze(0)
    world = _world_from_grid_coords(cube_coords.reshape(-1, 3), int(resolution))
    x_pixel, y_pixel, depth, _valid, _back = _project_world_points(
        world, int(target_size), int(target_size), float(fov), float(azimuth)
    )
    base = torch.arange(cube_coords.shape[0], dtype=torch.int64).view(-1, 1, 1) * 8
    triangles = (cube_triangles.view(1, -1, 3) + base).reshape(-1, 3)
    return x_pixel, y_pixel, depth, triangles


def _mesh_projection_geometry(
    mesh,
    target_size: int,
    fov: float,
    azimuth: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return projected normalized-mesh vertices and triangles."""
    vertices, faces = _single_mesh_tensors(mesh)
    world = torch.stack([vertices[:, 0], -vertices[:, 2], vertices[:, 1]], dim=-1)
    x_pixel, y_pixel, depth, _valid, _back = _project_world_points(
        world, int(target_size), int(target_size), float(fov), float(azimuth)
    )
    return x_pixel, y_pixel, depth, faces


def _project_voxel_occupancy_mask(
    spatial_coords: torch.Tensor,
    resolution: int,
    target_size: int,
    fov: float,
    azimuth: float,
) -> torch.Tensor:
    """Render the sparse shape tokens as occupied cubes into a solid silhouette."""
    x_pixel, y_pixel, depth, triangles = _voxel_projection_geometry(
        spatial_coords, resolution, target_size, fov, azimuth
    )
    return _rasterize_projected_triangles(
        x_pixel, y_pixel, depth, triangles, int(target_size)
    )


def _project_mesh_silhouette_mask(
    mesh,
    target_size: int,
    fov: float,
    azimuth: float,
) -> torch.Tensor:
    """Render a normalized Hunyuan mesh in the exact Pixal3D camera convention."""
    x_pixel, y_pixel, depth, faces = _mesh_projection_geometry(
        mesh, target_size, fov, azimuth
    )
    return _rasterize_projected_triangles(
        x_pixel, y_pixel, depth, faces, int(target_size)
    )


def _postprocess_shape_mask(
    mask: torch.Tensor,
    expand_pixels: int,
    feather_pixels: float,
) -> torch.Tensor:
    """Expand/erode a silhouette, then optionally produce a soft Gaussian edge."""
    result = mask.detach().float().clamp(0.0, 1.0)[None, None]
    amount = int(expand_pixels)
    if amount != 0:
        kernel = 2 * abs(amount) + 1
        if amount > 0:
            result = torch.nn.functional.max_pool2d(
                result, kernel_size=kernel, stride=1, padding=abs(amount)
            )
        else:
            result = 1.0 - torch.nn.functional.max_pool2d(
                1.0 - result, kernel_size=kernel, stride=1, padding=abs(amount)
            )
    if float(feather_pixels) > 0.0:
        result = _gaussian_blur_nchw(result, float(feather_pixels))
    return result[0].clamp(0.0, 1.0)


def _normalize_depth_map(
    depth: torch.Tensor,
    near_is_white: bool,
    gamma: float,
) -> tuple[torch.Tensor, float | None, float | None]:
    """Normalize finite camera depth to a black-background 0..1 conditioning map."""
    visible = torch.isfinite(depth)
    output = torch.zeros_like(depth, dtype=torch.float32)
    if not visible.any():
        return output, None, None
    near = float(depth[visible].amin())
    far = float(depth[visible].amax())
    if far - near <= 1e-8:
        normalized = torch.ones_like(depth[visible])
    else:
        normalized = (depth[visible] - near) / (far - near)
        if bool(near_is_white):
            normalized = 1.0 - normalized
    normalized = normalized.clamp(0.0, 1.0).pow(max(0.05, float(gamma)))
    # Keep every visible surface distinguishable from the zero-valued background.
    normalized = normalized * (254.0 / 255.0) + (1.0 / 255.0)
    output[visible] = normalized
    return output, near, far


def _canny_conditioning_image(
    grayscale: torch.Tensor,
    low_threshold: float,
    high_threshold: float,
    preblur_pixels: float,
    dilate_pixels: int,
) -> torch.Tensor:
    """Create a Canny IMAGE, using ComfyUI's Kornia path with a torch fallback."""
    low = float(low_threshold)
    high = float(high_threshold)
    if not 0.0 <= low < high <= 1.0:
        raise ValueError("Canny thresholds must satisfy 0 <= low < high <= 1")
    source = grayscale.detach().float().clamp(0.0, 1.0)
    if source.ndim == 2:
        source = source.unsqueeze(0)
    source_nchw = source.unsqueeze(1)
    if float(preblur_pixels) > 0.0:
        source_nchw = _gaussian_blur_nchw(source_nchw, float(preblur_pixels))
    try:
        from kornia.filters import canny

        edges = canny(source_nchw, low, high)[1]
    except ImportError:
        sobel_x = source_nchw.new_tensor(
            [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]
        ).unsqueeze(0)
        sobel_y = source_nchw.new_tensor(
            [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]
        ).unsqueeze(0)
        gradient_x = torch.nn.functional.conv2d(source_nchw, sobel_x, padding=1)
        gradient_y = torch.nn.functional.conv2d(source_nchw, sobel_y, padding=1)
        magnitude = torch.sqrt(gradient_x.square() + gradient_y.square())
        magnitude = magnitude / magnitude.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        weak = magnitude >= low
        connected = magnitude >= high
        for _ in range(8):
            connected = weak & (
                torch.nn.functional.max_pool2d(
                    connected.float(), kernel_size=3, stride=1, padding=1
                ) > 0.0
            )
        edges = connected.float()
    dilation = max(0, int(dilate_pixels))
    if dilation:
        edges = torch.nn.functional.max_pool2d(
            edges,
            kernel_size=2 * dilation + 1,
            stride=1,
            padding=dilation,
        )
    return edges[:, :1].repeat(1, 3, 1, 1).movedim(1, -1).cpu().clamp(0.0, 1.0)


def make_shape_view_maps(
    shape_latent: dict,
    mesh,
    view: str,
    fov: float,
    target_size: int,
    mask_expand_pixels: int,
    mask_feather_pixels: float,
    depth_source: str,
    depth_near_is_white: bool,
    depth_gamma: float,
    canny_source: str,
    canny_low_threshold: float,
    canny_high_threshold: float,
    canny_preblur_pixels: float,
    canny_dilate_pixels: int,
):
    """Render masks, geometric depth and Canny for one semantic Pixal3D view."""
    if view not in VIEW_NAMES:
        raise ValueError(f"view must be one of {VIEW_NAMES}, got {view!r}")
    if depth_source not in ("mesh", "voxel"):
        raise ValueError("depth_source must be mesh or voxel")
    if canny_source not in ("depth_map", "mesh_mask", "voxel_mask"):
        raise ValueError("canny_source must be depth_map, mesh_mask or voxel_mask")
    if not 1.0 <= float(fov) <= 170.0:
        raise ValueError(f"fov must be in 1..170 degrees, got {fov}")
    if int(target_size) < 64:
        raise ValueError("target_size must be at least 64")
    _features, coords = _flatten_sparse_latent(shape_latent)
    if coords.shape[0] == 0:
        raise ValueError("Shape View Maps requires a non-empty shape latent")
    if torch.unique(coords[:, 0]).numel() != 1:
        raise ValueError("Shape View Maps supports exactly one shape per execution")
    resolution = int(shape_latent.get("coord_resolution", int(coords[:, 1:].amax()) + 1))
    azimuth = VIEW_AZIMUTHS[VIEW_NAMES.index(view)]

    voxel_mask_raw = _project_voxel_occupancy_mask(
        coords[:, 1:], resolution, int(target_size), float(fov), azimuth
    )
    mesh_mask_raw = _project_mesh_silhouette_mask(
        mesh, int(target_size), float(fov), azimuth
    )
    voxel_mask = _postprocess_shape_mask(
        voxel_mask_raw, int(mask_expand_pixels), float(mask_feather_pixels)
    )
    mesh_mask = _postprocess_shape_mask(
        mesh_mask_raw, int(mask_expand_pixels), float(mask_feather_pixels)
    )

    if depth_source == "mesh":
        x_pixel, y_pixel, raw_depth, triangles = _mesh_projection_geometry(
            mesh, int(target_size), float(fov), azimuth
        )
    else:
        x_pixel, y_pixel, raw_depth, triangles = _voxel_projection_geometry(
            coords[:, 1:], resolution, int(target_size), float(fov), azimuth
        )
    depth_buffer = _rasterize_projected_depth(
        x_pixel, y_pixel, raw_depth, triangles, int(target_size)
    )
    depth_gray, depth_near, depth_far = _normalize_depth_map(
        depth_buffer, bool(depth_near_is_white), float(depth_gamma)
    )
    depth_image = depth_gray.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3)
    canny_input = {
        "depth_map": depth_gray.unsqueeze(0),
        "mesh_mask": mesh_mask,
        "voxel_mask": voxel_mask,
    }[canny_source]
    canny_image = _canny_conditioning_image(
        canny_input,
        float(canny_low_threshold),
        float(canny_high_threshold),
        float(canny_preblur_pixels),
        int(canny_dilate_pixels),
    )
    depth_range = (
        "empty"
        if depth_near is None
        else f"{depth_near:.6f}..{depth_far:.6f}"
    )
    report = "\n".join(
        (
            "Pixal3D shape view maps",
            f"view={view}, target={int(target_size)}x{int(target_size)}, fov={float(fov):.2f}",
            f"voxel_mask={int((voxel_mask >= 0.5).sum()):,}px, mesh_mask={int((mesh_mask >= 0.5).sum()):,}px",
            f"mask_expand={int(mask_expand_pixels):+d}px, mask_feather={float(mask_feather_pixels):.2f}px",
            f"depth_source={depth_source}, raw_visible_range={depth_range}, near_is_white={bool(depth_near_is_white)}, gamma={float(depth_gamma):.3f}",
            f"canny_source={canny_source}, thresholds={float(canny_low_threshold):.3f}/{float(canny_high_threshold):.3f}, preblur={float(canny_preblur_pixels):.2f}px, dilate={int(canny_dilate_pixels)}px",
        )
    )
    return voxel_mask, mesh_mask, depth_image, canny_image, report


def auto_align_single_view(
    shape_latent: dict,
    image: torch.Tensor,
    mask: torch.Tensor | None,
    view: str,
    fov: float,
    target_size: int,
    mask_mode: str,
    foreground_threshold: float,
    silhouette_scale_percent: float,
    alignment_strength: float,
    horizontal_offset_pixels: int,
    vertical_offset_pixels: int,
    x_deformation_percent: float,
    y_deformation_percent: float,
    overlay_point_size: int,
    overlay_point_density_percent: float,
    fit_mode: str,
    auto_align_enabled: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Uniformly align one foreground to one semantic Pixal3D camera view."""
    if view not in VIEW_NAMES:
        raise ValueError(f"view must be one of {VIEW_NAMES}, got {view!r}")
    _features, coords = _flatten_sparse_latent(shape_latent)
    if coords.shape[0] == 0:
        raise ValueError("Auto Align requires a non-empty shape latent")
    batch_ids = coords[:, 0]
    if torch.unique(batch_ids).numel() != 1:
        raise ValueError("Auto Align supports exactly one shape per execution")
    if not 1.0 <= float(fov) <= 170.0:
        raise ValueError(f"fov must be in 1..170 degrees, got {fov}")
    if int(target_size) < 64:
        raise ValueError("target_size must be at least 64")
    if fit_mode not in ("contain", "cover"):
        raise ValueError("fit_mode must be contain or cover")

    resolution = int(shape_latent.get("coord_resolution", int(coords[:, 1:].amax()) + 1))
    world = _world_from_grid_coords(coords[:, 1:].float(), resolution).cpu()
    preserve_source_alpha = image.shape[-1] >= 4 or (
        mask is not None and mask_mode != "ignore"
    )
    source_rgb = image[:1, ..., :3].detach().float().clamp(0.0, 1.0)
    source_alpha = (
        image[:1, ..., 3:4].detach().float().clamp(0.0, 1.0)
        if image.shape[-1] >= 4
        else torch.ones_like(source_rgb[..., :1])
    )
    if mask is not None and mask_mode != "ignore":
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim != 3 or mask.shape[0] not in (1, source_rgb.shape[0]):
            raise ValueError(f"{view}_mask must be [1,H,W] or match the image batch")
        if tuple(mask.shape[1:]) != tuple(source_rgb.shape[1:3]):
            mask = torch.nn.functional.interpolate(
                mask[:1].unsqueeze(1).float(),
                size=source_rgb.shape[1:3],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
        else:
            mask = mask[:1].float()
        mask_alpha = mask.clamp(0.0, 1.0)
        if mask_mode == "transparency":
            mask_alpha = 1.0 - mask_alpha
        source_alpha *= mask_alpha.unsqueeze(-1)
    source = (
        torch.cat((source_rgb, source_alpha), dim=-1)
        if preserve_source_alpha
        else source_rgb
    )[0].cpu()
    azimuth = VIEW_AZIMUTHS[VIEW_NAMES.index(view)]
    scale_multiplier = float(silhouette_scale_percent) / 100.0
    requested_strength = min(1.0, max(0.0, float(alignment_strength)))
    strength = requested_strength if bool(auto_align_enabled) else 0.0

    source_height, source_width = int(source.shape[0]), int(source.shape[1])
    sx0, sy0, sx1, sy1, found = _foreground_bbox(source, foreground_threshold)
    source_box_width = max(1.0, sx1 - sx0 + 1.0)
    source_box_height = max(1.0, sy1 - sy0 + 1.0)
    source_box_center = ((sx0 + sx1) / 2.0, (sy0 + sy1) / 2.0)

    px, py, _depth, valid, _back = _project_world_points(
        world, int(target_size), int(target_size), float(fov), azimuth
    )
    if not valid.any():
        raise ValueError(f"The projected shape is outside the {view} camera")
    tx0, tx1 = float(px[valid].amin()), float(px[valid].amax())
    ty0, ty1 = float(py[valid].amin()), float(py[valid].amax())
    target_box_width = max(1.0, tx1 - tx0 + 1.0)
    target_box_height = max(1.0, ty1 - ty0 + 1.0)
    ratios = (target_box_width / source_box_width, target_box_height / source_box_height)
    automatic_scale = min(ratios) if fit_mode == "contain" else max(ratios)

    base_scale = min(float(target_size) / source_width, float(target_size) / source_height)
    scale = math.exp(
        (1.0 - strength) * math.log(max(base_scale, 1e-8))
        + strength * math.log(max(automatic_scale, 1e-8))
    ) * scale_multiplier
    base_source_center = ((source_width - 1) / 2.0, (source_height - 1) / 2.0)
    base_target_center = ((target_size - 1) / 2.0, (target_size - 1) / 2.0)
    projected_center = ((tx0 + tx1) / 2.0, (ty0 + ty1) / 2.0)
    source_center = tuple(
        (1.0 - strength) * base + strength * automatic
        for base, automatic in zip(base_source_center, source_box_center)
    )
    target_center = (
        (1.0 - strength) * base_target_center[0]
        + strength * projected_center[0]
        + int(horizontal_offset_pixels),
        (1.0 - strength) * base_target_center[1]
        + strength * projected_center[1]
        + int(vertical_offset_pixels),
    )
    warped = _warp_image_to_canvas(
        source,
        int(target_size),
        scale,
        source_center,
        target_center,
        float(x_deformation_percent),
        float(y_deformation_percent),
    ).cpu()

    point_mask = torch.zeros((target_size, target_size), dtype=torch.float32)
    point_x = px[valid].round().long().clamp(0, target_size - 1)
    point_y = py[valid].round().long().clamp(0, target_size - 1)
    available_points = int(point_x.numel())
    density = min(100.0, max(0.0, float(overlay_point_density_percent)))
    kept_points = (
        max(1, int(round(available_points * density / 100.0)))
        if available_points and density > 0.0
        else 0
    )
    if 0 < kept_points < available_points:
        selection = torch.linspace(0, available_points - 1, kept_points).round().long()
        point_x = point_x[selection]
        point_y = point_y[selection]
    if kept_points:
        point_mask[point_y[:kept_points], point_x[:kept_points]] = 1.0
    point_size = max(1, int(overlay_point_size))
    if point_size % 2 == 0:
        point_size += 1
    point_mask = torch.nn.functional.max_pool2d(
        point_mask[None, None],
        kernel_size=point_size,
        stride=1,
        padding=point_size // 2,
    )[0, 0]
    overlay_alpha = point_mask.unsqueeze(-1) * 0.65
    overlay_color = torch.tensor([0.0, 1.0, 0.25]).view(1, 1, 3)
    overlay = warped[0, ..., :3] * (1.0 - overlay_alpha) + overlay_color * overlay_alpha
    report = "\n".join(
        [
            f"Pixal3D automatic alignment: {view}",
            f"target={int(target_size)}x{int(target_size)}, fov={float(fov):.2f}, fit={fit_mode}, auto_align={'enabled' if bool(auto_align_enabled) else 'disabled'}, strength={strength:.2f}",
            f"silhouette_scale={float(silhouette_scale_percent):.1f}%, offset=({int(horizontal_offset_pixels)}, {int(vertical_offset_pixels)}) px",
            f"deformation=(x={float(x_deformation_percent):+.1f}%, y={float(y_deformation_percent):+.1f}%), final_scale=(x={scale * (1.0 + float(x_deformation_percent) / 100.0):.4f}, y={scale * (1.0 + float(y_deformation_percent) / 100.0):.4f})",
            f"overlay_points={kept_points:,}/{available_points:,} ({density:.1f}%), point_size={point_size}px",
            f"source_bbox=({sx0:.0f},{sy0:.0f})..({sx1:.0f},{sy1:.0f}), mesh_bbox=({tx0:.1f},{ty0:.1f})..({tx1:.1f},{ty1:.1f}), scale={scale:.4f}"
            + ("" if found else " [foreground not found; used full image]"),
        ]
    )
    return warped, overlay.unsqueeze(0), report


def auto_align_four_views(
    shape_latent: dict,
    images: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    masks: tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None],
    fov: float,
    target_size: int,
    mask_mode: str,
    foreground_threshold: float,
    silhouette_scale_percent: float,
    alignment_strength: float,
    fit_mode: str,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor, str]:
    """Backward-compatible four-view wrapper around the single-view aligner."""
    results = [
        auto_align_single_view(
            shape_latent,
            image,
            mask,
            view,
            fov,
            target_size,
            mask_mode,
            foreground_threshold,
            silhouette_scale_percent,
            alignment_strength,
            0,
            0,
            0.0,
            0.0,
            5,
            100.0,
            fit_mode,
        )
        for image, mask, view in zip(images, masks, VIEW_NAMES)
    ]
    aligned = tuple(result[0] for result in results)
    overlays = torch.cat([result[1] for result in results], dim=0)
    report = "Pixal3D automatic four-view alignment\n" + "\n".join(
        f"{view}: {result[2].splitlines()[-1]}" for view, result in zip(VIEW_NAMES, results)
    )
    return aligned, torch.cat(aligned, dim=0), overlays, report


def batch_four_views(
    front: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    back: torch.Tensor,
) -> torch.Tensor:
    """Make an explicitly ordered F/L/R/B RGB(A) batch without resizing."""
    images = (front, left, right, back)
    first_shape = None
    validated = []
    preserve_alpha = False
    for name, image in zip(VIEW_NAMES, images):
        if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[-1] < 3:
            raise ValueError(f"{name} must be a ComfyUI IMAGE [B,H,W,3/4]")
        shape = (int(image.shape[1]), int(image.shape[2]))
        if first_shape is None:
            first_shape = shape
        elif shape != first_shape:
            raise ValueError(
                f"All four aligned views must have the same size; front={first_shape}, {name}={shape}"
            )
        preserve_alpha |= image.shape[-1] >= 4
        validated.append(image[:1].float().clamp(0.0, 1.0))
    prepared = []
    for image in validated:
        rgb = image[..., :3]
        if preserve_alpha:
            alpha = image[..., 3:4] if image.shape[-1] >= 4 else torch.ones_like(rgb[..., :1])
            prepared.append(torch.cat((rgb, alpha), dim=-1))
        else:
            prepared.append(rgb)
    return torch.cat(prepared, dim=0)


def voxel_diagnostic_text(voxel) -> str:
    coords = voxel.data
    colors = voxel.voxel_colors
    resolution = int(voxel.resolution)
    if not isinstance(coords, torch.Tensor) or not isinstance(colors, torch.Tensor):
        return "Pixal3D voxel diagnostic: invalid VOXEL tensors"
    if colors.ndim != 2 or colors.shape[0] == 0:
        return f"Pixal3D voxel diagnostic: empty/invalid colors {tuple(colors.shape)}"

    sample = colors.detach().float().cpu()
    if sample.shape[0] > 200_000:
        step = max(1, sample.shape[0] // 200_000)
        sample = sample[::step][:200_000]
    finite = torch.isfinite(sample)
    safe = torch.nan_to_num(sample, nan=0.0, posinf=0.0, neginf=0.0)
    mins = safe.amin(dim=0)
    maxs = safe.amax(dim=0)
    means = safe.mean(dim=0)
    stds = safe.std(dim=0, unbiased=False)
    base = safe[:, : min(3, safe.shape[1])]
    black = float((base.amax(dim=1) < 0.05).float().mean()) if base.numel() else 0.0
    gray = (
        float(((base.amax(dim=1) - base.amin(dim=1)) < 0.03).float().mean())
        if base.shape[1] == 3
        else 0.0
    )
    spatial = coords[:, -3:].detach().cpu() if coords.ndim == 2 and coords.shape[1] >= 3 else None
    coord_range = "unknown"
    if spatial is not None and spatial.numel():
        coord_range = f"{spatial.amin(dim=0).tolist()}..{spatial.amax(dim=0).tolist()}"

    def values(tensor):
        return "[" + ", ".join(f"{float(value):.4f}" for value in tensor) + "]"

    return "\n".join(
        [
            "Pixal3D decoded voxel diagnostics",
            f"resolution={resolution}, voxels={int(colors.shape[0]):,}, channels={int(colors.shape[1])}",
            f"coords={coord_range}",
            f"finite={float(finite.float().mean()):.6f}",
            f"min={values(mins)}",
            f"max={values(maxs)}",
            f"mean={values(means)}",
            f"std={values(stds)}",
            f"base_near_black={black:.4%}, base_near_gray={gray:.4%}",
        ]
    )


def _flatten_sparse_latent(latent: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sparse latent features as [N,C] and matching coords as [N,4]."""
    samples = latent.get("samples")
    coords = latent.get("coords")
    if not isinstance(samples, torch.Tensor) or not isinstance(coords, torch.Tensor):
        raise ValueError("LATENT must contain tensor entries 'samples' and 'coords'")
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"Expected sparse coords [N,4], got {tuple(coords.shape)}")

    if samples.ndim == 4 and samples.shape[-1] == 1:
        # External adapter format: [B,C,N,1].  coord_counts says how many of
        # the padded tokens belong to each batch item.
        counts = latent.get("coord_counts")
        if counts is None:
            if samples.shape[0] != 1:
                raise ValueError("Batched sparse LATENT requires coord_counts")
            counts = torch.tensor([coords.shape[0]], dtype=torch.int64)
        counts = torch.as_tensor(counts, dtype=torch.int64).cpu()
        if counts.numel() != samples.shape[0] or int(counts.sum()) != coords.shape[0]:
            raise ValueError("coord_counts does not match samples/coords")
        pieces = []
        for batch_index, count_value in enumerate(counts.tolist()):
            pieces.append(samples[batch_index, :, : int(count_value), 0].transpose(0, 1))
        features = torch.cat(pieces, dim=0) if pieces else samples.new_empty((0, samples.shape[1]))
    elif samples.ndim == 3:
        # Native sparse-stage format: [B,N,C].
        counts = latent.get("coord_counts")
        if counts is None:
            features = samples.reshape(-1, samples.shape[-1])
        else:
            counts = torch.as_tensor(counts, dtype=torch.int64).cpu()
            if counts.numel() != samples.shape[0] or int(counts.sum()) != coords.shape[0]:
                raise ValueError("coord_counts does not match samples/coords")
            features = torch.cat(
                [samples[i, : int(count)] for i, count in enumerate(counts.tolist())], dim=0
            )
    elif samples.ndim == 2:
        features = samples
    else:
        raise ValueError(
            "Expected sparse samples [B,C,N,1], [B,N,C], or [N,C], "
            f"got {tuple(samples.shape)}"
        )
    if features.shape[0] != coords.shape[0]:
        raise ValueError(
            f"Sparse token count mismatch: samples={features.shape[0]}, coords={coords.shape[0]}"
        )
    return features.detach().float().cpu(), coords.detach().to(dtype=torch.int64, device="cpu")


def latent_diagnostic_text(latent: dict, kind: str = "latent") -> str:
    """Describe sparse feature scale and spatial continuity before a VAE decode."""
    features, coords = _flatten_sparse_latent(latent)
    if features.shape[0] == 0:
        return f"Pixal3D {kind} latent diagnostics: no sparse tokens"

    finite = torch.isfinite(features)
    safe = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    channel_means = safe.mean(dim=0)
    channel_stds = safe.std(dim=0, unbiased=False)
    spatial = coords[:, 1:]
    resolution = int(latent.get("coord_resolution", int(spatial.amax()) + 1))

    # Compare only occupied direct neighbours.  An unexpectedly large value on
    # one axis is a compact numerical signature of the stripe direction.
    extent = max(resolution, int(spatial.amax()) + 1)
    keys = (((coords[:, 0] * extent + spatial[:, 0]) * extent + spatial[:, 1]) * extent + spatial[:, 2])
    sorted_keys, permutation = torch.sort(keys)
    del permutation
    deltas = (extent * extent, extent, 1)
    neighbour_lines = []
    for axis, (axis_name, delta) in enumerate(zip("xyz", deltas)):
        candidates = spatial[:, axis] < extent - 1
        source_indexes = torch.nonzero(candidates, as_tuple=False).flatten()
        target_keys = keys[source_indexes] + delta
        positions = torch.searchsorted(sorted_keys, target_keys)
        in_range = positions < sorted_keys.numel()
        matched = torch.zeros_like(in_range)
        matched[in_range] = sorted_keys[positions[in_range]] == target_keys[in_range]
        source_indexes = source_indexes[matched]
        if source_indexes.numel():
            target_positions = positions[matched]
            # sorted_keys came from keys[permutation]; recover original rows.
            sorted_indices = torch.argsort(keys)
            target_indexes = sorted_indices[target_positions]
            mad = (safe[source_indexes] - safe[target_indexes]).abs().mean()
            neighbour_lines.append(
                f"{axis_name}={float(mad):.6f} (pairs={int(source_indexes.numel()):,})"
            )
        else:
            neighbour_lines.append(f"{axis_name}=n/a (pairs=0)")

    coord_min = spatial.amin(dim=0).tolist()
    coord_max = spatial.amax(dim=0).tolist()
    shape_text = tuple(latent["samples"].shape)
    return "\n".join(
        [
            f"Pixal3D {kind} latent diagnostics",
            f"samples={shape_text}, tokens={features.shape[0]:,}, channels={features.shape[1]}, coord_resolution={resolution}",
            f"coords={coord_min}..{coord_max}",
            f"finite={float(finite.float().mean()):.6f}",
            f"global: min={float(safe.amin()):.6f}, max={float(safe.amax()):.6f}, mean={float(safe.mean()):.6f}, std={float(safe.std(unbiased=False)):.6f}",
            f"channel_mean_range={float(channel_means.amin()):.6f}..{float(channel_means.amax()):.6f}",
            f"channel_std_range={float(channel_stds.amin()):.6f}..{float(channel_stds.amax()):.6f}",
            "occupied-neighbour feature MAD: " + ", ".join(neighbour_lines),
        ]
    )


def make_projection_diagnostics(
    shape_latent: dict,
    prepared_views: torch.Tensor,
    fov: float,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Visualize the exact sparse-coordinate camera projection used by Pixal3D.

    The first output overlays projected sparse tokens on F/L/R/B.  The second
    directly samples RGB at those locations, averages the views, and renders the
    resulting coarse token field from four sides.  No DINO, diffusion, or VAE is
    involved, so this isolates framing/camera/axis mistakes.
    """
    _features, coords = _flatten_sparse_latent(shape_latent)
    if coords.shape[0] == 0:
        raise ValueError("Projection diagnostics requires a non-empty shape latent")
    if prepared_views.ndim != 4 or prepared_views.shape[0] < 4 or prepared_views.shape[-1] < 3:
        raise ValueError("prepared_views must contain F/L/R/B as IMAGE batch [4,H,W,3]")
    if not 1.0 <= float(fov) <= 170.0:
        raise ValueError(f"fov must be in 1..170 degrees, got {fov}")

    views = prepared_views[:4, ..., :3].detach().float().cpu().clamp(0.0, 1.0)
    height, width = int(views.shape[1]), int(views.shape[2])
    if height != width:
        raise ValueError(f"Projection diagnostics requires square views, got {width}x{height}")

    resolution = int(shape_latent.get("coord_resolution", int(coords[:, 1:].amax()) + 1))
    spatial = coords[:, 1:].float()
    norm = spatial / float(max(1, resolution - 1)) * 2.0 - 1.0
    # Same grid-to-Blender/world rotation used by ComfyUI's Pixal3D projector:
    # (x, y, z) -> (x, -z, y), followed by the training scale of 1/2.
    world = torch.stack([norm[:, 0], -norm[:, 2], norm[:, 1]], dim=-1) / 2.0

    fov_radians = math.radians(float(fov))
    distance = 1.1 * 0.5 / math.tan(fov_radians / 2.0)
    focal_pixels = width / (2.0 * math.tan(fov_radians / 2.0))
    azimuths = (0.0, 90.0, 270.0, 180.0)  # F / L / R / B
    overlays = []
    sampled_rgb = []
    report_lines = [
        "Pixal3D raw projection diagnostics",
        f"tokens={coords.shape[0]:,}, coord_resolution={resolution}, view_size={width}x{height}, fov={float(fov):.2f}",
    ]

    for view_index, (name, azimuth) in enumerate(zip(("front", "left", "right", "back"), azimuths)):
        az = math.radians(azimuth)
        back = torch.tensor([math.sin(az), -math.cos(az), 0.0], dtype=torch.float32)
        right = torch.tensor([math.cos(az), math.sin(az), 0.0], dtype=torch.float32)
        up = torch.cross(back, right, dim=0)
        rotation = torch.stack([right, up, back], dim=-1)
        translation = back * distance
        camera = (world - translation) @ rotation
        x_camera, y_camera, z_camera = camera.unbind(dim=-1)
        depth = -z_camera
        x_pixel = focal_pixels * x_camera / (depth + 1e-8) + width / 2.0
        y_pixel = -focal_pixels * y_camera / (depth + 1e-8) + height / 2.0
        valid = (
            (x_pixel >= 0)
            & (x_pixel < width)
            & (y_pixel >= 0)
            & (y_pixel < height)
            & (depth > 0)
        )

        # Bilinear sampling uses the same half-pixel conversion and border mode
        # as the native DINO/NAF projection path.
        uv = torch.stack([x_pixel, y_pixel], dim=-1)
        uv_ndc = (uv + 0.5) / float(width) * 2.0 - 1.0
        grid = uv_ndc.view(1, -1, 1, 2)
        sampled = torch.nn.functional.grid_sample(
            views[view_index].movedim(-1, 0).unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        ).squeeze(0).squeeze(-1).transpose(0, 1)
        sampled_rgb.append(sampled)

        foreground = sampled.amax(dim=1) > 0.03
        valid_fraction = float(valid.float().mean())
        foreground_fraction = float(foreground[valid].float().mean()) if valid.any() else 0.0
        report_lines.append(
            f"{name}: in_frame={valid_fraction:.2%}, projected_on_nonblack={foreground_fraction:.2%}"
        )

        point_mask = torch.zeros((height, width), dtype=torch.float32)
        px = x_pixel[valid].round().long().clamp(0, width - 1)
        py = y_pixel[valid].round().long().clamp(0, height - 1)
        point_mask[py, px] = 1.0
        point_mask = torch.nn.functional.max_pool2d(
            point_mask[None, None], kernel_size=5, stride=1, padding=2
        )[0, 0]
        overlay_color = torch.tensor([0.0, 1.0, 0.25]).view(1, 1, 3)
        alpha = point_mask.unsqueeze(-1) * 0.65
        overlays.append(views[view_index] * (1.0 - alpha) + overlay_color * alpha)

    token_rgb = torch.stack(sampled_rgb, dim=0).mean(dim=0)

    class _ProjectionVoxel:
        pass

    projected = _ProjectionVoxel()
    projected.data = coords.to(torch.int32)
    projected.voxel_colors = token_rgb
    projected.resolution = resolution
    token_projections = make_voxel_projection_preview(projected)
    return torch.stack(overlays, dim=0), token_projections, "\n".join(report_lines)


def make_coordinate_test_voxel(voxel):
    """Replace base color with an XYZ ramp while preserving coords/PBR channels.

    Baking this field must produce a smooth gradient.  It is deliberately derived
    from the decoded voxel coordinates, so it tests only voxel-to-mesh/UV mapping
    and cannot be affected by the diffusion model's predicted colors.
    """
    coords = voxel.data
    colors = voxel.voxel_colors
    resolution = int(voxel.resolution)
    if not isinstance(coords, torch.Tensor) or not isinstance(colors, torch.Tensor):
        raise ValueError("Invalid VOXEL tensors")
    if coords.ndim != 2 or coords.shape[-1] < 3 or colors.ndim != 2:
        raise ValueError("Coordinate test requires [N,3/4] coords and [N,C] colors")
    if colors.shape[0] != coords.shape[0] or colors.shape[1] < 3:
        raise ValueError("Coordinate/color counts do not match, or RGB is missing")
    if resolution <= 0:
        raise ValueError(f"Invalid voxel resolution: {resolution}")

    spatial = coords[:, -3:].to(device=colors.device, dtype=torch.float32)
    ramp = ((spatial + 0.5) / float(resolution)).clamp(0.0, 1.0)
    test_colors = colors.clone()
    test_colors[:, :3] = ramp.to(dtype=test_colors.dtype)
    out = copy.copy(voxel)
    out.voxel_colors = test_colors
    return out


def make_voxel_projection_preview(voxel) -> torch.Tensor:
    """Render four nearest-surface RGB projections before any UV processing.

    Batch order is Z-, X-, X+, Z+.  This is a diagnostic preview rather than a
    physically lit render: every pixel uses the outermost occupied voxel color.
    """
    coords = voxel.data
    colors = voxel.voxel_colors
    resolution = int(voxel.resolution)
    if not isinstance(coords, torch.Tensor) or not isinstance(colors, torch.Tensor):
        raise ValueError("Invalid VOXEL tensors")
    if coords.ndim != 2 or coords.shape[-1] < 3 or colors.ndim != 2 or colors.shape[1] < 3:
        raise ValueError("Voxel projections require [N,3/4] coords and RGB colors")
    if colors.shape[0] != coords.shape[0] or colors.shape[0] == 0:
        raise ValueError("Voxel projections require non-empty matching coords/colors")
    if resolution <= 0:
        raise ValueError(f"Invalid voxel resolution: {resolution}")

    # The texture decoder normally emits one batch.  If a batch column exists,
    # preview item zero so the result remains a normal IMAGE batch of four views.
    if coords.shape[-1] == 4:
        keep = coords[:, 0] == 0
        coords = coords[keep]
        colors = colors[keep]
    spatial = coords[:, -3:].detach().to(device="cpu", dtype=torch.int64)
    rgb = colors[:, :3].detach().to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0)
    if spatial.shape[0] == 0:
        return torch.zeros((4, resolution, resolution, 3), dtype=torch.float32)

    pixel_count = resolution * resolution

    def project(horizontal_axis: int, depth_axis: int, take_max: bool, flip_h: bool):
        horizontal = spatial[:, horizontal_axis].clamp(0, resolution - 1)
        vertical = spatial[:, 1].clamp(0, resolution - 1)
        if flip_h:
            horizontal = (resolution - 1) - horizontal
        row = (resolution - 1) - vertical
        flat = row * resolution + horizontal
        depth = spatial[:, depth_axis]

        fill = -1 if take_max else resolution
        extreme = torch.full((pixel_count,), fill, dtype=torch.int64)
        reduction = "amax" if take_max else "amin"
        extreme.scatter_reduce_(0, flat, depth, reduce=reduction, include_self=True)
        visible = depth == extreme[flat]
        visible_flat = flat[visible]
        visible_rgb = rgb[visible]

        accum = torch.zeros((pixel_count, 3), dtype=torch.float32)
        count = torch.zeros((pixel_count, 1), dtype=torch.float32)
        accum.index_add_(0, visible_flat, visible_rgb)
        count.index_add_(
            0,
            visible_flat,
            torch.ones((visible_flat.shape[0], 1), dtype=torch.float32),
        )
        return (accum / count.clamp_min(1.0)).reshape(resolution, resolution, 3)

    # Opposite views are mirrored horizontally so silhouettes have conventional
    # screen orientation.  Explicit axis names are used in the workflow title.
    return torch.stack(
        [
            project(0, 2, False, False),  # Z-
            project(2, 0, False, True),   # X-
            project(2, 0, True, False),   # X+
            project(0, 2, True, True),    # Z+
        ],
        dim=0,
    )


def make_voxel_camera_preview(voxel, fov: float) -> torch.Tensor:
    """Render the voxel field through the reprojection cameras in B/L/R/F order.

    Unlike the legacy axis preview, this uses the same perspective projection,
    screen-right basis and FOV as image-guided reprojection. It therefore shows
    the orientation that is actually stored in the voxel field and subsequently
    sampled by the mesh texture bake.
    """
    coords = voxel.data
    colors = voxel.voxel_colors
    resolution = int(voxel.resolution)
    if not isinstance(coords, torch.Tensor) or not isinstance(colors, torch.Tensor):
        raise ValueError("Invalid VOXEL tensors")
    if coords.ndim != 2 or coords.shape[-1] not in (3, 4):
        raise ValueError("Voxel camera preview requires coordinates [N,3/4]")
    if colors.ndim != 2 or colors.shape[0] != coords.shape[0] or colors.shape[1] < 3:
        raise ValueError("Voxel camera preview requires matching RGB colors")
    if resolution <= 0:
        raise ValueError(f"Invalid voxel resolution: {resolution}")
    if coords.shape[-1] == 4:
        keep = coords[:, 0] == 0
        coords = coords[keep]
        colors = colors[keep]
    if coords.shape[0] == 0:
        return torch.zeros((4, resolution, resolution, 3), dtype=torch.float32)

    device = colors.device
    spatial = coords[:, -3:].detach().to(device=device, dtype=torch.float32)
    rgb = colors[:, :3].detach().to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
    world = _world_from_grid_coords(spatial, resolution)
    pixel_count = resolution * resolution
    views = []
    for name in ("back", "left", "right", "front"):
        azimuth = VIEW_AZIMUTHS[VIEW_NAMES.index(name)]
        x_pixel, y_pixel, depth, valid, _camera_direction = _project_world_points(
            world, resolution, resolution, float(fov), azimuth
        )
        pixel_x = x_pixel.round().long().clamp(0, resolution - 1)
        pixel_y = y_pixel.round().long().clamp(0, resolution - 1)
        flat = pixel_y * resolution + pixel_x
        z_buffer = torch.full(
            (pixel_count,), float("inf"), device=device, dtype=torch.float32
        )
        z_buffer.scatter_reduce_(
            0, flat[valid], depth[valid].float(), reduce="amin", include_self=True
        )
        visible = valid & (depth <= z_buffer[flat] + 1e-7)
        visible_flat = flat[visible]
        accum = torch.zeros((pixel_count, 3), device=device, dtype=torch.float32)
        count = torch.zeros((pixel_count, 1), device=device, dtype=torch.float32)
        accum.index_add_(0, visible_flat, rgb[visible])
        count.index_add_(
            0,
            visible_flat,
            torch.ones((visible_flat.shape[0], 1), device=device, dtype=torch.float32),
        )
        views.append((accum / count.clamp_min(1.0)).reshape(resolution, resolution, 3).cpu())
    return torch.stack(views, dim=0)


def _angular_coverage_weight(
    world: torch.Tensor,
    camera_direction: torch.Tensor,
    coverage_degrees: float,
    edge_softness_percent: float,
) -> torch.Tensor:
    """Return a plateau-plus-cosine angular sector weight for one camera."""
    horizontal = world[:, :2]
    radius = torch.linalg.vector_norm(horizontal, dim=1)
    unit = horizontal / radius.clamp_min(1e-8).unsqueeze(1)
    direction = camera_direction[:2]
    direction = direction / torch.linalg.vector_norm(direction).clamp_min(1e-8)
    angle = torch.rad2deg(torch.acos((unit @ direction).clamp(-1.0, 1.0)))
    angle = torch.where(radius > 1e-8, angle, torch.zeros_like(angle))

    half_width = max(0.5, min(180.0, float(coverage_degrees)) / 2.0)
    softness = min(1.0, max(0.0, float(edge_softness_percent) / 100.0))
    plateau = half_width * (1.0 - softness)
    if softness <= 0.0:
        return (angle <= half_width).float()
    transition = ((angle - plateau) / max(half_width - plateau, 1e-6)).clamp(0.0, 1.0)
    fade = 0.5 * (1.0 + torch.cos(transition * math.pi))
    fade = torch.where(angle <= plateau, torch.ones_like(fade), fade).clamp_min(1e-6)
    return fade * (angle <= half_width)


def _gaussian_blur_nchw(
    image: torch.Tensor,
    sigma_pixels: float,
    padding_mode: str = "replicate",
) -> torch.Tensor:
    """Apply a channel-independent separable Gaussian blur to an NCHW tensor."""
    sigma = float(sigma_pixels)
    if sigma <= 1e-6:
        return image
    radius = max(1, int(math.ceil(3.0 * sigma)))
    axis = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    kernel = torch.exp(-(axis * axis) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum().clamp_min(torch.finfo(image.dtype).eps)
    channels = int(image.shape[1])
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    if padding_mode not in ("replicate", "constant"):
        raise ValueError("Gaussian padding_mode must be replicate or constant")
    result = torch.nn.functional.conv2d(
        torch.nn.functional.pad(image, (radius, radius, 0, 0), mode=padding_mode),
        horizontal,
        groups=channels,
    )
    return torch.nn.functional.conv2d(
        torch.nn.functional.pad(result, (0, 0, radius, radius), mode=padding_mode),
        vertical,
        groups=channels,
    )


def _inner_edge_feather_mask(mask: torch.Tensor, width_pixels: float) -> torch.Tensor:
    """Feather only the inside edge of an NCHW visibility mask.

    ``width_pixels`` describes the approximate complete transition width rather
    than Gaussian sigma. RGB is deliberately untouched, so fine texture detail
    away from the silhouette remains sharp.
    """
    width = max(0.0, float(width_pixels))
    source = mask.float().clamp(0.0, 1.0)
    if width <= 1e-6:
        return source
    binary = (source > 1e-6).to(source.dtype)
    blurred = _gaussian_blur_nchw(
        binary,
        max(width / 3.0, 0.25),
        padding_mode="constant",
    )
    return (source * blurred).clamp(0.0, 1.0)


def _erode_mask_nchw(mask: torch.Tensor, radius_pixels: float) -> torch.Tensor:
    """Erode an NCHW mask by a square pixel radius using separable max-pooling."""
    radius = max(0, int(round(float(radius_pixels))))
    source = mask.float().clamp(0.0, 1.0)
    if radius == 0:
        return source
    kernel = radius * 2 + 1
    outside = 1.0 - source
    outside = torch.nn.functional.max_pool2d(
        torch.nn.functional.pad(
            outside, (radius, radius, 0, 0), mode="constant", value=1.0
        ),
        kernel_size=(1, kernel),
        stride=1,
    )
    outside = torch.nn.functional.max_pool2d(
        torch.nn.functional.pad(
            outside, (0, 0, radius, radius), mode="constant", value=1.0
        ),
        kernel_size=(kernel, 1),
        stride=1,
    )
    return (1.0 - outside).clamp(0.0, 1.0)


def _camera_visibility_feather_weights(
    visible: torch.Tensor,
    flat_pixels: torch.Tensor,
    uv_ndc: torch.Tensor,
    height: int,
    width: int,
    feather_pixels: float,
    inset_pixels: float = 0.0,
) -> torch.Tensor:
    """Sample an inward feather from the actual z-buffer-visible silhouette.

    The mask is built from the same rounded camera pixels used by the z-buffer.
    ``inset_pixels`` removes a fully transparent inner rim before
    ``feather_pixels`` transitions toward full opacity. A one-pixel dilation
    stabilizes sparse boundaries without making hidden voxels eligible: the
    returned values are still multiplied by ``visible`` later.
    """
    feather = max(0.0, float(feather_pixels))
    inset = max(0.0, float(inset_pixels))
    if feather <= 1e-6 and inset <= 1e-6:
        return torch.ones_like(visible, dtype=torch.float32)
    mask_flat = torch.zeros(
        (int(height) * int(width),), device=visible.device, dtype=torch.float32
    )
    if torch.any(visible):
        mask_flat.scatter_(0, flat_pixels[visible], 1.0)
    mask = mask_flat.reshape(1, 1, int(height), int(width))
    mask = torch.nn.functional.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    # Compensate the one-pixel stabilization dilation so the user-facing inset
    # remains measured from the original visible silhouette.
    mask = _erode_mask_nchw(mask, inset + 1.0 if inset > 1e-6 else 0.0)
    if feather > 1e-6:
        mask = _inner_edge_feather_mask(mask, feather)
        # A raw Gaussian step is still about 0.5 at the boundary. Remap that
        # half-range to 0..1 so overlapping cameras can make a complete transition
        # instead of retaining an opaque rim at their visible silhouettes.
        mask = ((mask - 0.5) * 2.0).clamp(0.0, 1.0)
    return torch.nn.functional.grid_sample(
        mask,
        uv_ndc.view(1, -1, 1, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).flatten().clamp(0.0, 1.0)


def _view_crop_mask(
    height: int,
    width: int,
    crop_left_percent: float,
    crop_right_percent: float,
    crop_top_percent: float,
    crop_bottom_percent: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    """Build a camera-space crop mask without resizing the remaining image.

    Percentages are measured from the displayed source after ``flip_horizontal``.
    The returned pixel tuple is left/right/top/bottom for diagnostics.
    """
    values = tuple(
        min(95.0, max(0.0, float(value)))
        for value in (
            crop_left_percent,
            crop_right_percent,
            crop_top_percent,
            crop_bottom_percent,
        )
    )
    left_percent, right_percent, top_percent, bottom_percent = values
    if left_percent + right_percent >= 100.0:
        raise ValueError("crop_left_percent + crop_right_percent must be below 100")
    if top_percent + bottom_percent >= 100.0:
        raise ValueError("crop_top_percent + crop_bottom_percent must be below 100")

    left = int(round(width * left_percent / 100.0))
    right = int(round(width * right_percent / 100.0))
    top = int(round(height * top_percent / 100.0))
    bottom = int(round(height * bottom_percent / 100.0))
    x_end = max(left + 1, width - right)
    y_end = max(top + 1, height - bottom)
    mask = torch.zeros((1, 1, height, width), device=device, dtype=dtype)
    mask[:, :, top:y_end, left:x_end] = 1.0
    return mask, (left, right, top, bottom)


def _parse_rgb_color(color: str) -> tuple[float, float, float]:
    """Parse a ComfyUI COLOR widget value into normalized RGB."""
    value = str(color).strip()
    if len(value) not in (7, 9) or not value.startswith("#"):
        raise ValueError("uncovered_color must be #RRGGBB or #RRGGBBAA")
    try:
        packed = int(value[1:7], 16)
    except ValueError as exc:
        raise ValueError("uncovered_color must be #RRGGBB or #RRGGBBAA") from exc
    return (
        ((packed >> 16) & 255) / 255.0,
        ((packed >> 8) & 255) / 255.0,
        (packed & 255) / 255.0,
    )


def _rgb_to_hsl(rgb: torch.Tensor) -> torch.Tensor:
    """Convert [...,3] RGB in 0..1 to HSL in 0..1 without leaving torch."""
    red, green, blue = rgb.unbind(dim=-1)
    maximum, maximum_index = rgb.max(dim=-1)
    minimum = rgb.min(dim=-1).values
    delta = maximum - minimum
    lightness = (maximum + minimum) * 0.5
    saturation = torch.where(
        delta > 1e-8,
        delta / (1.0 - (2.0 * lightness - 1.0).abs()).clamp_min(1e-8),
        torch.zeros_like(delta),
    )
    hue_red = torch.remainder((green - blue) / delta.clamp_min(1e-8), 6.0)
    hue_green = (blue - red) / delta.clamp_min(1e-8) + 2.0
    hue_blue = (red - green) / delta.clamp_min(1e-8) + 4.0
    hue = torch.where(
        maximum_index == 0,
        hue_red,
        torch.where(maximum_index == 1, hue_green, hue_blue),
    ) / 6.0
    hue = torch.where(delta > 1e-8, hue, torch.zeros_like(hue))
    return torch.stack([hue, saturation, lightness], dim=-1).clamp(0.0, 1.0)


def _hsl_to_rgb(hsl: torch.Tensor) -> torch.Tensor:
    """Convert [...,3] HSL in 0..1 to RGB in 0..1."""
    hue, saturation, lightness = hsl.unbind(dim=-1)
    chroma = (1.0 - (2.0 * lightness - 1.0).abs()) * saturation
    sector = torch.remainder(hue, 1.0) * 6.0
    x_value = chroma * (1.0 - torch.remainder(sector, 2.0).sub(1.0).abs())
    zeros = torch.zeros_like(chroma)
    index = torch.floor(sector).to(torch.int64).clamp(0, 5)
    red = torch.where(
        (index == 0) | (index == 5),
        chroma,
        torch.where((index == 1) | (index == 4), x_value, zeros),
    )
    green = torch.where(
        (index == 1) | (index == 2),
        chroma,
        torch.where((index == 0) | (index == 3), x_value, zeros),
    )
    blue = torch.where(
        (index == 3) | (index == 4),
        chroma,
        torch.where((index == 2) | (index == 5), x_value, zeros),
    )
    offset = lightness - chroma * 0.5
    return torch.stack([red + offset, green + offset, blue + offset], dim=-1).clamp(0.0, 1.0)


def blend_rgb(base: torch.Tensor, layer: torch.Tensor, mode: str) -> torch.Tensor:
    """Apply a Photoshop-style RGB blend function before layer opacity."""
    if mode not in BLEND_MODES:
        raise ValueError(f"blend_mode must be one of {BLEND_MODES}, got {mode!r}")
    base = base.float().clamp(0.0, 1.0)
    layer = layer.float().clamp(0.0, 1.0)
    epsilon = 1e-6
    if mode == "normal":
        result = layer
    elif mode == "darken":
        result = torch.minimum(base, layer)
    elif mode == "multiply":
        result = base * layer
    elif mode == "color_burn":
        result = torch.where(
            layer <= epsilon,
            torch.zeros_like(base),
            1.0 - (1.0 - base) / layer.clamp_min(epsilon),
        )
    elif mode == "linear_burn":
        result = base + layer - 1.0
    elif mode == "darker_color":
        result = torch.where(
            (layer.sum(dim=-1) < base.sum(dim=-1)).unsqueeze(-1), layer, base
        )
    elif mode == "lighten":
        result = torch.maximum(base, layer)
    elif mode == "screen":
        result = 1.0 - (1.0 - base) * (1.0 - layer)
    elif mode == "color_dodge":
        result = torch.where(
            layer >= 1.0 - epsilon,
            torch.ones_like(base),
            base / (1.0 - layer).clamp_min(epsilon),
        )
    elif mode == "linear_dodge_add":
        result = base + layer
    elif mode == "lighter_color":
        result = torch.where(
            (layer.sum(dim=-1) > base.sum(dim=-1)).unsqueeze(-1), layer, base
        )
    elif mode == "overlay":
        result = torch.where(
            base <= 0.5,
            2.0 * base * layer,
            1.0 - 2.0 * (1.0 - base) * (1.0 - layer),
        )
    elif mode == "soft_light":
        curve = torch.where(
            base <= 0.25,
            ((16.0 * base - 12.0) * base + 4.0) * base,
            torch.sqrt(base.clamp_min(0.0)),
        )
        result = torch.where(
            layer <= 0.5,
            base - (1.0 - 2.0 * layer) * base * (1.0 - base),
            base + (2.0 * layer - 1.0) * (curve - base),
        )
    elif mode == "hard_light":
        result = torch.where(
            layer <= 0.5,
            2.0 * base * layer,
            1.0 - 2.0 * (1.0 - base) * (1.0 - layer),
        )
    elif mode == "vivid_light":
        burn = torch.where(
            layer <= epsilon,
            torch.zeros_like(base),
            1.0 - (1.0 - base) / (2.0 * layer).clamp_min(epsilon),
        )
        dodge = torch.where(
            layer >= 1.0 - epsilon,
            torch.ones_like(base),
            base / (2.0 * (1.0 - layer)).clamp_min(epsilon),
        )
        result = torch.where(layer <= 0.5, burn, dodge)
    elif mode == "linear_light":
        result = base + 2.0 * layer - 1.0
    elif mode == "pin_light":
        result = torch.where(
            layer <= 0.5,
            torch.minimum(base, 2.0 * layer),
            torch.maximum(base, 2.0 * layer - 1.0),
        )
    elif mode == "hard_mix":
        vivid = blend_rgb(base, layer, "vivid_light")
        result = (vivid >= 0.5).to(base.dtype)
    elif mode == "difference":
        result = (base - layer).abs()
    elif mode == "exclusion":
        result = base + layer - 2.0 * base * layer
    elif mode == "subtract":
        result = base - layer
    elif mode == "divide":
        result = base / layer.clamp_min(epsilon)
    else:
        base_hsl = _rgb_to_hsl(base)
        layer_hsl = _rgb_to_hsl(layer)
        if mode == "hue":
            result_hsl = torch.stack(
                [layer_hsl[..., 0], base_hsl[..., 1], base_hsl[..., 2]], dim=-1
            )
        elif mode == "saturation":
            result_hsl = torch.stack(
                [base_hsl[..., 0], layer_hsl[..., 1], base_hsl[..., 2]], dim=-1
            )
        elif mode == "color":
            result_hsl = torch.stack(
                [layer_hsl[..., 0], layer_hsl[..., 1], base_hsl[..., 2]], dim=-1
            )
        else:  # luminosity
            result_hsl = torch.stack(
                [base_hsl[..., 0], base_hsl[..., 1], layer_hsl[..., 2]], dim=-1
            )
        result = _hsl_to_rgb(result_hsl)
    return result.clamp(0.0, 1.0)


@torch.no_grad()
def reproject_single_view_rgb(
    voxel,
    image: torch.Tensor,
    view: str,
    fov: float,
    blend_mode: str,
    opacity: float,
    coverage_degrees: float,
    edge_softness_percent: float,
    visibility_tolerance_voxels: float,
    background_threshold: float,
    gaussian_blur_target: str,
    gaussian_blur_sigma_pixels: float,
):
    """Project one image as one ordered RGB layer while preserving PBR channels."""
    if view not in VIEW_NAMES:
        raise ValueError(f"view must be one of {VIEW_NAMES}, got {view!r}")
    if blend_mode not in BLEND_MODES:
        raise ValueError(f"blend_mode must be one of {BLEND_MODES}, got {blend_mode!r}")
    if gaussian_blur_target not in GAUSSIAN_BLUR_TARGETS:
        raise ValueError(
            f"gaussian_blur_target must be one of {GAUSSIAN_BLUR_TARGETS}, "
            f"got {gaussian_blur_target!r}"
        )
    coords = voxel.data
    colors = voxel.voxel_colors
    resolution = int(voxel.resolution)
    if not isinstance(coords, torch.Tensor) or not isinstance(colors, torch.Tensor):
        raise ValueError("Single-View Reprojection requires tensor VOXEL data")
    if coords.ndim != 2 or coords.shape[1] not in (3, 4):
        raise ValueError(f"Expected voxel coordinates [N,3/4], got {tuple(coords.shape)}")
    if colors.ndim != 2 or colors.shape[0] != coords.shape[0] or colors.shape[1] < 3:
        raise ValueError("Voxel colors must be [N,C], C>=3, and match the coordinates")
    if colors.shape[0] == 0 or resolution <= 0:
        raise ValueError("Cannot reproject an empty or invalid voxel field")
    if coords.shape[1] == 4 and torch.unique(coords[:, 0]).numel() != 1:
        raise ValueError("Single-View Reprojection supports one voxel object per run")
    if not isinstance(image, torch.Tensor) or image.ndim != 4:
        raise ValueError("image must be an IMAGE [B,H,W,3/4]")
    if image.shape[0] < 1 or image.shape[-1] not in (3, 4):
        raise ValueError("image must contain RGB or RGBA channels")
    height, width = int(image.shape[1]), int(image.shape[2])
    if height != width:
        raise ValueError(f"Reprojection requires a square image, got {width}x{height}")
    if not 1.0 <= float(fov) <= 170.0:
        raise ValueError(f"fov must be in 1..170 degrees, got {fov}")

    device = colors.device
    spatial = coords[:, -3:].detach().to(device=device, dtype=torch.float32)
    world = _world_from_grid_coords(spatial, resolution)
    base_rgb = colors[:, :3].detach().float()
    azimuth = VIEW_AZIMUTHS[VIEW_NAMES.index(view)]
    x_pixel, y_pixel, depth, in_frame, camera_direction = _project_world_points(
        world, width, height, float(fov), azimuth
    )
    rounded_x = x_pixel.round().long().clamp(0, width - 1)
    rounded_y = y_pixel.round().long().clamp(0, height - 1)
    flat = rounded_y * width + rounded_x
    z_buffer = torch.full((width * height,), float("inf"), device=device, dtype=torch.float32)
    z_buffer.scatter_reduce_(0, flat[in_frame], depth[in_frame], reduce="amin", include_self=True)
    tolerance = max(0.0, float(visibility_tolerance_voxels)) / float(max(1, resolution - 1))
    visible = in_frame & (depth <= z_buffer[flat] + tolerance)

    source = image[:1].detach().to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
    source_rgb = source[..., :3].movedim(-1, 1)
    if source.shape[-1] == 4:
        source_mask = source[..., 3:4].movedim(-1, 1)
    elif float(background_threshold) <= 0.0:
        source_mask = torch.ones((1, 1, height, width), device=device, dtype=torch.float32)
    else:
        source_mask = (
            source[..., :3].amax(dim=-1, keepdim=True) > float(background_threshold)
        ).movedim(-1, 1).float()
    sigma = max(0.0, float(gaussian_blur_sigma_pixels))
    if sigma > 0.0 and gaussian_blur_target in ("rgb", "rgb_and_mask"):
        source_rgb = _gaussian_blur_nchw(source_rgb, sigma)
    if sigma > 0.0 and gaussian_blur_target in ("blend_mask", "rgb_and_mask"):
        source_mask = _gaussian_blur_nchw(source_mask, sigma)

    uv = torch.stack([x_pixel, y_pixel], dim=-1)
    uv_ndc = torch.empty_like(uv)
    uv_ndc[:, 0] = (uv[:, 0] + 0.5) / float(width) * 2.0 - 1.0
    uv_ndc[:, 1] = (uv[:, 1] + 0.5) / float(height) * 2.0 - 1.0
    sampling_grid = uv_ndc.view(1, -1, 1, 2)
    sampled_rgb = torch.nn.functional.grid_sample(
        source_rgb,
        sampling_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).squeeze(0).squeeze(-1).transpose(0, 1)
    sampled_mask = torch.nn.functional.grid_sample(
        source_mask,
        sampling_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).flatten().clamp(0.0, 1.0)
    angular = _angular_coverage_weight(
        world, camera_direction, float(coverage_degrees), float(edge_softness_percent)
    )
    alpha = (
        visible.float()
        * sampled_mask
        * angular
        * min(1.0, max(0.0, float(opacity)))
    ).clamp(0.0, 1.0)
    layer_rgb = blend_rgb(base_rgb, sampled_rgb, blend_mode)
    final_rgb = torch.lerp(base_rgb, layer_rgb, alpha.unsqueeze(1)).clamp(0.0, 1.0)
    output_colors = colors.clone()
    output_colors[:, :3] = final_rgb.to(dtype=colors.dtype)
    result = copy.copy(voxel)
    result.voxel_colors = output_colors
    used = alpha > 1e-8
    report = "\n".join(
        (
            "Pixal3D single-view image-guided voxel reprojection",
            f"view={view}, image={width}x{height}, fov={float(fov):.2f}",
            f"blend_mode={blend_mode}, opacity={float(opacity):.3f}, coverage={float(coverage_degrees):.1f}°, edge_softness={float(edge_softness_percent):.1f}%",
            f"gaussian_blur={sigma:.2f}px target={gaussian_blur_target}, visibility_tolerance={float(visibility_tolerance_voxels):.2f} voxels",
            f"visible={int(visible.sum()):,}, used={int(used.sum()):,}/{colors.shape[0]:,}, mean_alpha={float(alpha.mean()):.4f}; PBR channels 3+ preserved",
        )
    )
    return result, report


@torch.no_grad()
def reproject_voxel_rgb(
    voxel,
    prepared_views: torch.Tensor,
    fov: float,
    reprojection_strength: float,
    view_weights: tuple[float, float, float, float],
    view_coverages: tuple[float, float, float, float],
    edge_softness_percent: float | tuple[float, float, float, float],
    visibility_tolerance_voxels: float,
    background_threshold: float | tuple[float, float, float, float],
    view_blend_modes: tuple[str, str, str, str] = ("normal",) * 4,
    view_blur_targets: tuple[str, str, str, str] = ("blend_mask",) * 4,
    view_blur_sigmas: tuple[float, float, float, float] = (0.0,) * 4,
    generated_texture_strength_percent: float = 100.0,
    uncovered_color: str = "#808080",
    view_flip_horizontal: tuple[bool, bool, bool, bool] = (False,) * 4,
    view_edge_feather_pixels: tuple[float, float, float, float] = (0.0,) * 4,
    layer_order_bottom_to_top: str | tuple[str, str, str, str] = VIEW_NAMES,
    view_crop_left_percent: tuple[float, float, float, float] = (0.0,) * 4,
    view_crop_right_percent: tuple[float, float, float, float] = (0.0,) * 4,
    view_crop_top_percent: tuple[float, float, float, float] = (0.0,) * 4,
    view_crop_bottom_percent: tuple[float, float, float, float] = (0.0,) * 4,
    view_coverage_modes: tuple[str, str, str, str] = ("angular_sector",) * 4,
    view_camera_feather_pixels: tuple[float, float, float, float] = (0.0,) * 4,
    view_camera_feather_inset_pixels: tuple[float, float, float, float] = (0.0,) * 4,
    camera_composite_mode: str = "ordered_layers",
):
    """Blend high-resolution F/L/R/B RGB views into a PBR voxel field.

    ``ordered_layers`` preserves the original Photoshop-style compositing.
    ``weighted_average`` normalizes all camera contributions per voxel, making
    overlaps independent of layer order and preventing the top view from
    darkening or dominating a seam.
    """
    coords = voxel.data
    colors = voxel.voxel_colors
    resolution = int(voxel.resolution)
    if not isinstance(coords, torch.Tensor) or not isinstance(colors, torch.Tensor):
        raise ValueError("Image-Guided Voxel Reprojection requires tensor VOXEL data")
    if coords.ndim != 2 or coords.shape[1] not in (3, 4):
        raise ValueError(f"Expected voxel coordinates [N,3/4], got {tuple(coords.shape)}")
    if colors.ndim != 2 or colors.shape[0] != coords.shape[0] or colors.shape[1] < 3:
        raise ValueError("Voxel colors must be [N,C], C>=3, and match the coordinates")
    if colors.shape[0] == 0 or resolution <= 0:
        raise ValueError("Cannot reproject an empty or invalid voxel field")
    if coords.shape[1] == 4 and torch.unique(coords[:, 0]).numel() != 1:
        raise ValueError("Image-Guided Voxel Reprojection supports one voxel object per run")
    if not isinstance(prepared_views, torch.Tensor) or prepared_views.ndim != 4:
        raise ValueError("prepared_views must be an IMAGE batch [4,H,W,3/4]")
    if prepared_views.shape[0] < 4 or prepared_views.shape[-1] not in (3, 4):
        raise ValueError("prepared_views must contain F/L/R/B with RGB or RGBA channels")
    height, width = int(prepared_views.shape[1]), int(prepared_views.shape[2])
    if height != width:
        raise ValueError(f"Reprojection requires square views, got {width}x{height}")
    if not 1.0 <= float(fov) <= 170.0:
        raise ValueError(f"fov must be in 1..170 degrees, got {fov}")
    if not (
        len(view_blend_modes)
        == len(view_blur_targets)
        == len(view_blur_sigmas)
        == len(view_flip_horizontal)
        == len(view_edge_feather_pixels)
        == len(view_coverage_modes)
        == len(view_camera_feather_pixels)
        == len(view_camera_feather_inset_pixels)
        == 4
    ):
        raise ValueError(
            "Blend modes, blur settings, horizontal flips and edge feather settings "
            "coverage modes, camera feather and inset settings must contain F/L/R/B values"
        )
    invalid_modes = [mode for mode in view_blend_modes if mode not in BLEND_MODES]
    if invalid_modes:
        raise ValueError(f"Unknown blend mode: {invalid_modes[0]!r}")
    invalid_targets = [target for target in view_blur_targets if target not in GAUSSIAN_BLUR_TARGETS]
    if invalid_targets:
        raise ValueError(f"Unknown Gaussian blur target: {invalid_targets[0]!r}")
    invalid_coverage_modes = [
        mode for mode in view_coverage_modes if mode not in COVERAGE_MODES
    ]
    if invalid_coverage_modes:
        raise ValueError(f"Unknown coverage mode: {invalid_coverage_modes[0]!r}")
    if camera_composite_mode not in CAMERA_COMPOSITE_MODES:
        raise ValueError(
            f"camera_composite_mode must be one of {CAMERA_COMPOSITE_MODES}, "
            f"got {camera_composite_mode!r}"
        )
    view_softnesses = _four_view_values(edge_softness_percent, "edge_softness_percent")
    view_background_thresholds = _four_view_values(
        background_threshold, "background_threshold"
    )
    view_crop_lefts = _four_view_values(
        view_crop_left_percent, "view_crop_left_percent"
    )
    view_crop_rights = _four_view_values(
        view_crop_right_percent, "view_crop_right_percent"
    )
    view_crop_tops = _four_view_values(
        view_crop_top_percent, "view_crop_top_percent"
    )
    view_crop_bottoms = _four_view_values(
        view_crop_bottom_percent, "view_crop_bottom_percent"
    )
    layer_order = parse_view_layer_order(layer_order_bottom_to_top)
    layer_indices = tuple(VIEW_NAMES.index(name) for name in layer_order)

    device = colors.device
    spatial = coords[:, -3:].detach().to(device=device, dtype=torch.float32)
    world = _world_from_grid_coords(spatial, resolution)
    base_rgb = colors[:, :3].detach().float()
    generated_strength = min(
        1.0, max(0.0, float(generated_texture_strength_percent) / 100.0)
    )
    fill_rgb = base_rgb.new_tensor(_parse_rgb_color(uncovered_color)).view(1, 3)
    underlay_rgb = torch.lerp(fill_rgb.expand_as(base_rgb), base_rgb, generated_strength)
    projected_rgb = underlay_rgb.clone()
    weight_sum = torch.zeros((colors.shape[0],), device=device, dtype=torch.float32)
    debug_rgb = torch.zeros_like(base_rgb)
    weighted_rgb_sum = torch.zeros_like(base_rgb)
    weighted_debug_sum = torch.zeros_like(base_rgb)
    tolerance = max(0.0, float(visibility_tolerance_voxels)) / float(max(1, resolution - 1))
    palette = (
        (1.0, 0.15, 0.0),
        (0.1, 1.0, 0.2),
        (0.0, 0.35, 1.0),
        (1.0, 0.0, 0.85),
    )
    report_lines = [
        "Pixal3D image-guided voxel reprojection",
        f"voxels={colors.shape[0]:,}, views={width}x{height}, fov={float(fov):.2f}, strength={float(reprojection_strength):.2f}",
        f"camera_composite_mode={camera_composite_mode}",
        f"generated_texture={generated_strength:.2%}, uncovered_color={uncovered_color}",
        f"layer_order_bottom_to_top={' > '.join(layer_order)}",
        f"visibility_tolerance={float(visibility_tolerance_voxels):.2f} voxels; softness/background threshold are per-view",
    ]

    for layer_position, index in enumerate(layer_indices, start=1):
        name = VIEW_NAMES[index]
        azimuth = VIEW_AZIMUTHS[index]
        configured_weight = view_weights[index]
        coverage = view_coverages[index]
        blend_mode = view_blend_modes[index]
        blur_target = view_blur_targets[index]
        blur_sigma = view_blur_sigmas[index]
        flip_horizontal = view_flip_horizontal[index]
        edge_feather_pixels = view_edge_feather_pixels[index]
        crop_left = view_crop_lefts[index]
        crop_right = view_crop_rights[index]
        crop_top = view_crop_tops[index]
        crop_bottom = view_crop_bottoms[index]
        coverage_mode = view_coverage_modes[index]
        camera_feather_pixels = max(0.0, float(view_camera_feather_pixels[index]))
        camera_feather_inset_pixels = max(
            0.0, float(view_camera_feather_inset_pixels[index])
        )
        softness = view_softnesses[index]
        view_background_threshold = view_background_thresholds[index]
        configured_weight = max(0.0, float(configured_weight))
        blur_sigma = max(0.0, float(blur_sigma))
        if configured_weight == 0.0:
            report_lines.append(
                f"layer {layer_position}/4 {name}: disabled (weight=0)"
            )
            continue
        x_pixel, y_pixel, depth, in_frame, camera_direction = _project_world_points(
            world, width, height, float(fov), azimuth
        )
        rounded_x = x_pixel.round().long().clamp(0, width - 1)
        rounded_y = y_pixel.round().long().clamp(0, height - 1)
        flat = rounded_y * width + rounded_x
        z_buffer = torch.full(
            (width * height,), float("inf"), device=device, dtype=torch.float32
        )
        z_buffer.scatter_reduce_(
            0, flat[in_frame], depth[in_frame], reduce="amin", include_self=True
        )
        visible = in_frame & (depth <= z_buffer[flat] + tolerance)

        view = prepared_views[index : index + 1].detach().to(device=device, dtype=torch.float32)
        if bool(flip_horizontal):
            view = torch.flip(view, dims=(2,))
        rgb_view = view[..., :3].movedim(-1, 1).clamp(0.0, 1.0)
        crop_active = any(
            float(value) > 1e-6
            for value in (crop_left, crop_right, crop_top, crop_bottom)
        )
        if crop_active:
            crop_mask, crop_pixels = _view_crop_mask(
                height,
                width,
                crop_left,
                crop_right,
                crop_top,
                crop_bottom,
                device=device,
                dtype=torch.float32,
            )
        else:
            crop_mask = None
            crop_pixels = (0, 0, 0, 0)
        if view.shape[-1] == 4:
            mask_view = view[..., 3:4].movedim(-1, 1).clamp(0.0, 1.0)
        elif float(view_background_threshold) <= 0.0:
            mask_view = torch.ones((1, 1, height, width), device=device, dtype=torch.float32)
        else:
            mask_view = (
                view[..., :3].amax(dim=-1, keepdim=True)
                > float(view_background_threshold)
            ).movedim(-1, 1).float()
        if crop_mask is not None:
            mask_view *= crop_mask
        if blur_sigma > 0.0 and blur_target in ("rgb", "rgb_and_mask"):
            rgb_view = _gaussian_blur_nchw(rgb_view, blur_sigma)
        soft_mask = blur_sigma > 0.0 and blur_target in ("blend_mask", "rgb_and_mask")
        if soft_mask:
            mask_view = _gaussian_blur_nchw(mask_view, blur_sigma)
        edge_feather_pixels = max(0.0, float(edge_feather_pixels))
        edge_mask_view = None
        if edge_feather_pixels > 0.0:
            if view.shape[-1] == 4:
                edge_mask_view = view[..., 3:4].movedim(-1, 1).clamp(0.0, 1.0)
            else:
                edge_threshold = max(float(view_background_threshold), 1e-6)
                edge_mask_view = (
                    view[..., :3].amax(dim=-1, keepdim=True) > edge_threshold
                ).movedim(-1, 1).float()
            if crop_mask is not None:
                edge_mask_view *= crop_mask
            edge_mask_view = _inner_edge_feather_mask(
                edge_mask_view, edge_feather_pixels
            )
        uv = torch.stack([x_pixel, y_pixel], dim=-1)
        uv_ndc = torch.empty_like(uv)
        uv_ndc[:, 0] = (uv[:, 0] + 0.5) / float(width) * 2.0 - 1.0
        uv_ndc[:, 1] = (uv[:, 1] + 0.5) / float(height) * 2.0 - 1.0
        sampled = torch.nn.functional.grid_sample(
            rgb_view,
            uv_ndc.view(1, -1, 1, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).squeeze(0).squeeze(-1).transpose(0, 1)
        sampled_mask = torch.nn.functional.grid_sample(
            mask_view,
            uv_ndc.view(1, -1, 1, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).flatten().clamp(0.0, 1.0)
        sampled_edge = None
        if edge_mask_view is not None:
            sampled_edge = torch.nn.functional.grid_sample(
                edge_mask_view,
                uv_ndc.view(1, -1, 1, 2),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            ).flatten().clamp(0.0, 1.0)
        if soft_mask:
            foreground_strength = sampled_mask
            foreground = foreground_strength > 1e-6
        else:
            if view.shape[-1] == 4:
                foreground = sampled_mask > float(view_background_threshold)
            elif float(view_background_threshold) <= 0.0:
                foreground = sampled_mask > 1e-6
                foreground_strength = sampled_mask
            elif blur_sigma <= 0.0:
                # Preserve the original four-view node's threshold behaviour.
                foreground = sampled.amax(dim=1) > float(view_background_threshold)
            else:
                foreground = sampled_mask > 0.5
            foreground_strength = foreground.float()
        if sampled_edge is not None:
            foreground &= sampled_edge > 1e-6
            foreground_strength *= sampled_edge

        if coverage_mode == "all_camera_visible":
            # Camera visibility already excludes out-of-frame and occluded voxels.
            # Avoid rejecting silhouette parts merely because their positions lie
            # outside an origin-centred angular sector.
            angular = _camera_visibility_feather_weights(
                visible,
                flat,
                uv_ndc,
                height,
                width,
                camera_feather_pixels,
                camera_feather_inset_pixels,
            )
        else:
            angular = _angular_coverage_weight(
                world, camera_direction, float(coverage), float(softness)
            )
        eligible = visible & foreground & (angular > 0.0)
        coverage_weight = angular * visible.float() * foreground_strength
        blend_weight = coverage_weight * configured_weight
        layer_alpha = blend_weight.clamp(0.0, 1.0)
        if camera_composite_mode == "weighted_average":
            # Evaluate non-Normal modes against the common underlay, then
            # normalize camera evidence.  For the normal mode this is exactly a
            # weighted average of the unmodified source views.
            blended_view = blend_rgb(underlay_rgb, sampled, blend_mode)
            weighted_rgb_sum.add_(blended_view * blend_weight.unsqueeze(1))
            debug_color = world.new_tensor(palette[index]).view(1, 3)
            weighted_debug_sum.add_(debug_color * blend_weight.unsqueeze(1))
        else:
            # True bottom-to-top layer compositing. Weight is an opacity multiplier:
            # 1.0 is fully opaque in the camera centre, values below 1 mix with lower
            # layers, and values above 1 push soft/feathered coverage toward opacity.
            blended_view = blend_rgb(projected_rgb, sampled, blend_mode)
            projected_rgb = torch.lerp(
                projected_rgb, blended_view, layer_alpha.unsqueeze(1)
            ).clamp(0.0, 1.0)
            debug_color = world.new_tensor(palette[index]).view(1, 3)
            debug_rgb = torch.lerp(
                debug_rgb,
                debug_color.expand_as(debug_rgb),
                layer_alpha.unsqueeze(1),
            )
        weight_sum.add_(blend_weight)
        report_lines.append(
            f"layer {layer_position}/4 {name}: mode={blend_mode}, weight={configured_weight:.2f}, "
            f"coverage_mode={coverage_mode}, coverage={float(coverage):.1f}°, "
            f"softness={float(softness):.1f}%, camera_inset={camera_feather_inset_pixels:.1f}px, "
            f"camera_feather={camera_feather_pixels:.1f}px, "
            f"background={float(view_background_threshold):.4f}, blur={blur_sigma:.2f}px/{blur_target}, "
            f"edge_feather={edge_feather_pixels:.1f}px, flip_x={bool(flip_horizontal)}, "
            f"crop=L{float(crop_left):.1f}%/R{float(crop_right):.1f}%/T{float(crop_top):.1f}%/B{float(crop_bottom):.1f}% "
            f"({crop_pixels[0]}/{crop_pixels[1]}/{crop_pixels[2]}/{crop_pixels[3]}px), "
            f"visible={int(visible.sum()):,}, used={int(eligible.sum()):,}"
        )

    has_source = weight_sum > 1e-8
    if camera_composite_mode == "weighted_average":
        denominator = weight_sum.clamp_min(1e-8).unsqueeze(1)
        normalized_rgb = (weighted_rgb_sum / denominator).clamp(0.0, 1.0)
        projected_rgb = torch.where(has_source.unsqueeze(1), normalized_rgb, underlay_rgb)
        coverage_alpha = weight_sum.clamp(0.0, 1.0)
        normalized_debug = (weighted_debug_sum / denominator).clamp(0.0, 1.0)
        debug_rgb = normalized_debug * coverage_alpha.unsqueeze(1)
        blend_alpha = coverage_alpha * min(
            1.0, max(0.0, float(reprojection_strength))
        )
    else:
        blend_alpha = has_source.float() * min(
            1.0, max(0.0, float(reprojection_strength))
        )
    final_rgb = torch.lerp(
        underlay_rgb, projected_rgb, blend_alpha.unsqueeze(1)
    ).clamp(0.0, 1.0)
    output_colors = colors.clone()
    output_colors[:, :3] = final_rgb.to(dtype=colors.dtype)
    result = copy.copy(voxel)
    result.voxel_colors = output_colors

    debug_colors = colors.clone()
    debug_colors[:, :3] = debug_rgb.to(dtype=colors.dtype)
    debug_voxel = copy.copy(voxel)
    debug_voxel.voxel_colors = debug_colors
    report_lines.append(
        f"reprojected={int(has_source.sum()):,}/{colors.shape[0]:,} "
        f"({float(has_source.float().mean()):.2%}), mean_alpha={float(blend_alpha.mean()):.4f}; PBR channels 3+ preserved"
    )
    report_lines.append(
        "preview=B/L/R/F through the same perspective cameras, FOV and handedness as reprojection"
    )
    return (
        result,
        make_voxel_camera_preview(result, float(fov)),
        make_voxel_camera_preview(debug_voxel, float(fov)),
        "\n".join(report_lines),
    )


def _projected_features_from_conditioning(conditioning) -> torch.Tensor:
    if not conditioning:
        raise ValueError("CONDITIONING is empty")
    for entry in conditioning:
        if isinstance(entry, (list, tuple)) and len(entry) >= 2 and isinstance(entry[1], dict):
            features = entry[1].get("trellis2_proj_feats")
            if isinstance(features, torch.Tensor):
                if features.ndim == 3 and features.shape[0] == 1:
                    features = features[0]
                if features.ndim != 2:
                    raise ValueError(
                        "Expected sparse trellis2_proj_feats [N,C], got "
                        f"{tuple(features.shape)}"
                    )
                return features
    raise ValueError(
        "CONDITIONING has no trellis2_proj_feats. Connect the output of "
        "Trellis2 Texture Stage, not the raw four-view conditioning."
    )


def _feature_neighbour_mads(
    features: torch.Tensor,
    coords: torch.Tensor,
    resolution: int,
) -> list[tuple[str, float | None, int]]:
    spatial = coords[:, 1:]
    extent = max(int(resolution), int(spatial.amax()) + 1)
    keys = (
        ((coords[:, 0] * extent + spatial[:, 0]) * extent + spatial[:, 1]) * extent
        + spatial[:, 2]
    )
    sorted_keys, sorted_indices = torch.sort(keys)
    results = []
    for axis, (axis_name, delta) in enumerate(
        zip("xyz", (extent * extent, extent, 1))
    ):
        sources = torch.nonzero(spatial[:, axis] < extent - 1, as_tuple=False).flatten()
        targets = keys[sources] + delta
        positions = torch.searchsorted(sorted_keys, targets)
        in_range = positions < sorted_keys.numel()
        matched = torch.zeros_like(in_range)
        matched[in_range] = sorted_keys[positions[in_range]] == targets[in_range]
        sources = sources[matched]
        if sources.numel() == 0:
            results.append((axis_name, None, 0))
            continue
        target_rows = sorted_indices[positions[matched]]
        mad = float((features[sources] - features[target_rows]).abs().mean())
        results.append((axis_name, mad, int(sources.numel())))
    return results


def _feature_rgb_projection(features: torch.Tensor) -> torch.Tensor:
    """Compress a feature field to three deterministic, approximately-PCA axes."""
    safe = torch.nan_to_num(features.float(), nan=0.0, posinf=0.0, neginf=0.0)
    centered = safe - safe.mean(dim=0, keepdim=True)
    channel_scale = safe.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-5)
    standardized = centered / channel_scale
    channels = standardized.shape[1]
    index = torch.arange(channels, dtype=torch.float32).unsqueeze(1)
    basis = torch.cat(
        [
            torch.sin(index * 0.017 + 0.3),
            torch.cos(index * 0.031 + 1.1),
            torch.sin(index * 0.047 + 2.3),
        ],
        dim=1,
    )
    basis = torch.linalg.qr(basis, mode="reduced").Q
    # Two power iterations are enough for a stable diagnostic colour field and
    # avoid the cost of a full SVD over 2048 channels.
    for _ in range(2):
        scores = standardized @ basis
        basis = torch.linalg.qr(standardized.T @ scores, mode="reduced").Q
    scores = standardized @ basis
    lows = torch.quantile(scores, 0.01, dim=0)
    highs = torch.quantile(scores, 0.99, dim=0)
    return ((scores - lows) / (highs - lows).clamp_min(1e-6)).clamp(0.0, 1.0)


def projected_feature_diagnostics(
    conditioning,
    shape_latent: dict,
) -> tuple[str, torch.Tensor]:
    native = _projected_features_from_conditioning(conditioning)
    _shape_features, coords = _flatten_sparse_latent(shape_latent)
    features = native.detach().float().cpu()
    if features.shape[0] != coords.shape[0]:
        raise ValueError(
            "Projected feature/token mismatch: "
            f"features={features.shape[0]}, coords={coords.shape[0]}"
        )
    if features.shape[1] < 2 or features.shape[1] % 2:
        raise ValueError(
            "Pixal3D texture projection must contain equal DINO and NAF halves; "
            f"got {features.shape[1]} channels"
        )

    resolution = int(
        shape_latent.get("coord_resolution", int(coords[:, 1:].amax()) + 1)
    )
    split = features.shape[1] // 2
    branches = (("DINO", features[:, :split]), ("NAF", features[:, split:]))
    report = [
        "Pixal3D projected feature diagnostics",
        f"features={tuple(native.shape)}, dtype={native.dtype}, tokens={features.shape[0]:,}, coord_resolution={resolution}",
    ]
    previews = []
    normalized_mads = {}
    for name, branch in branches:
        finite = torch.isfinite(branch)
        safe = torch.nan_to_num(branch, nan=0.0, posinf=0.0, neginf=0.0)
        channel_mean = safe.mean(dim=0)
        channel_std = safe.std(dim=0, unbiased=False)
        standardized = (safe - channel_mean) / channel_std.clamp_min(1e-5)
        raw_mads = _feature_neighbour_mads(safe, coords, resolution)
        norm_mads = _feature_neighbour_mads(standardized, coords, resolution)
        normalized_mads[name] = {axis: value for axis, value, _pairs in norm_mads}

        def format_mads(values):
            return ", ".join(
                f"{axis}={'n/a' if value is None else f'{value:.6f}'} (pairs={pairs:,})"
                for axis, value, pairs in values
            )

        report.extend(
            [
                f"{name}: finite={float(finite.float().mean()):.6f}, min={float(safe.amin()):.6f}, max={float(safe.amax()):.6f}, mean={float(safe.mean()):.6f}, std={float(safe.std(unbiased=False)):.6f}",
                f"{name}: channel_mean_range={float(channel_mean.amin()):.6f}..{float(channel_mean.amax()):.6f}, channel_std_range={float(channel_std.amin()):.6f}..{float(channel_std.amax()):.6f}",
                f"{name} neighbour MAD raw: {format_mads(raw_mads)}",
                f"{name} neighbour MAD standardized: {format_mads(norm_mads)}",
            ]
        )
        colors = _feature_rgb_projection(safe)

        class _FeatureVoxel:
            pass

        field = _FeatureVoxel()
        field.data = coords.to(torch.int32)
        field.voxel_colors = colors
        field.resolution = resolution
        previews.append(make_voxel_projection_preview(field))

    similarity = torch.nn.functional.cosine_similarity(
        features[:, :split], features[:, split:], dim=1, eps=1e-8
    )
    report.append(
        "DINO/NAF token cosine: "
        f"mean={float(similarity.mean()):.6f}, min={float(similarity.amin()):.6f}, max={float(similarity.amax()):.6f}"
    )
    for name in ("DINO", "NAF"):
        values = normalized_mads[name]
        other = [values.get("x"), values.get("z")]
        if values.get("y") is not None and all(value is not None for value in other):
            ratio = values["y"] / max(1e-8, 0.5 * (other[0] + other[1]))
            report.append(f"{name} standardized Y/(X,Z mean)={ratio:.6f}")
    report.append("preview batch: first 4=DINO, last 4=NAF; order Z- / X- / X+ / Z+")
    return "\n".join(report), torch.cat(previews, dim=0)


@dataclass(frozen=True)
class MeshTransform:
    center: torch.Tensor
    scale: float
    yaw: torch.Tensor


def normalize_mesh(mesh, yaw_degrees: float):
    vertices, faces = _single_mesh_tensors(mesh)
    v_min, v_max = vertices.amin(dim=0), vertices.amax(dim=0)
    center = (v_min + v_max) * 0.5
    max_extent = float((v_max - v_min).amax())
    if not math.isfinite(max_extent) or max_extent <= 0.0:
        raise ValueError("Mesh has zero or invalid extent")
    scale = 0.99999 / max_extent
    yaw = rotation_y(yaw_degrees, dtype=torch.float32)
    normalized = ((vertices - center) * scale) @ yaw.T

    out = copy.copy(mesh)
    out.vertices = normalized.unsqueeze(0)
    out.faces = faces.to(torch.int32).unsqueeze(0)
    if getattr(out, "normals", None) is not None:
        normals = out.normals[0].detach().float().cpu() @ yaw.T
        out.normals = torch.nn.functional.normalize(normals, dim=-1).unsqueeze(0)
    if getattr(out, "tangents", None) is not None:
        tangents = out.tangents[0].detach().float().cpu().clone()
        tangents[:, :3] = torch.nn.functional.normalize(tangents[:, :3] @ yaw.T, dim=-1)
        out.tangents = tangents.unsqueeze(0)
    return out, MeshTransform(center=center, scale=scale, yaw=yaw)


def restore_mesh(mesh, transform: MeshTransform):
    vertices, faces = _single_mesh_tensors(mesh)
    restored = (vertices @ transform.yaw) / transform.scale + transform.center
    out = copy.copy(mesh)
    out.vertices = restored.unsqueeze(0)
    out.faces = faces.to(torch.int32).unsqueeze(0)
    if getattr(out, "normals", None) is not None:
        normals = out.normals[0].detach().float().cpu() @ transform.yaw
        out.normals = torch.nn.functional.normalize(normals, dim=-1).unsqueeze(0)
    if getattr(out, "tangents", None) is not None:
        tangents = out.tangents[0].detach().float().cpu().clone()
        tangents[:, :3] = torch.nn.functional.normalize(tangents[:, :3] @ transform.yaw, dim=-1)
        out.tangents = tangents.unsqueeze(0)
    return out


class Trellis2ShapeEncoderLoader:
    @classmethod
    def INPUT_TYPES(cls):
        folder_paths = _ensure_encoder_folder()
        names = folder_paths.get_filename_list(ENCODER_FOLDER)
        return {"required": {"encoder_name": (names,)}}

    RETURN_TYPES = ("TRELLIS2_SHAPE_ENCODER",)
    RETURN_NAMES = ("shape_encoder",)
    FUNCTION = "load"
    CATEGORY = categoryFolder
    DESCRIPTION = (
        "Loads the official TRELLIS.2 external-mesh shape encoder used to turn a "
        "Hunyuan MESH into the shape latent required by Pixal3D's texture stage."
    )

    def load(self, encoder_name):
        import comfy.model_management
        import comfy.model_patcher
        import comfy.storage
        import comfy.utils


        folder_paths = _ensure_encoder_folder()
        path = folder_paths.get_full_path_or_raise(ENCODER_FOLDER, encoder_name)
        state_dict = comfy.utils.load_torch_file(path, safe_load=True)
        fast_disk = comfy.storage.state_dict_fast_disk(state_dict)
        model = ExternalMeshShapeEncoder()
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "Wrong TRELLIS.2 shape encoder checkpoint: "
                f"missing={list(missing)[:8]}, unexpected={list(unexpected)[:8]}"
            )
        model.eval()
        patcher = comfy.model_patcher.CoreModelPatcher(
            model,
            load_device=comfy.model_management.get_torch_device(),
            offload_device=comfy.model_management.unet_offload_device(),
            fast_disk=fast_disk,
        )
        return (patcher,)


class TrimeshToComfyMeshV040:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "copy_visuals": ("BOOLEAN", {"default": True}),
                "copy_normals": ("BOOLEAN", {"default": True}),
                "flip_uv_v": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "convert"
    CATEGORY = categoryFolder
    DESCRIPTION = (
        "Converts Hunyuan3DWrapper TRIMESH output to the expanded native MESH "
        "layout used by ComfyUI v0.40.0. Geometry is unchanged; available UVs, "
        "vertex colours, normals and PBR maps can also be copied."
    )

    def convert(self, trimesh, copy_visuals=True, copy_normals=True, flip_uv_v=False):
        from comfy_api.latest import Types


        payload = trimesh_to_payload(
            trimesh,
            copy_visuals=bool(copy_visuals),
            copy_normals=bool(copy_normals),
            flip_uv_v=bool(flip_uv_v),
        )
        mesh = Types.MESH(
            vertices=payload.vertices,
            faces=payload.faces,
            uvs=payload.uvs,
            vertex_colors=payload.vertex_colors,
            texture=payload.texture,
            metallic_roughness=payload.metallic_roughness,
            normals=payload.normals,
            normal_map=payload.normal_map,
            occlusion_in_mr=payload.occlusion_in_mr,
            material=payload.material,
            emissive=payload.emissive,
        )
        return (mesh,)


def _enable_dino_projection_fallback(conditionings) -> None:
    """Duplicate projected DINO features into the NAF half expected by Pixal3D."""
    seen = set()
    for conditioning in conditionings:
        if not conditioning:
            continue
        for entry in conditioning:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2 or not isinstance(entry[1], dict):
                continue
            proj_pack = entry[1].get("proj_feat_pack")
            if not isinstance(proj_pack, dict) or id(proj_pack) in seen:
                continue
            seen.add(id(proj_pack))
            stages = proj_pack.get("stages", {})
            for stage_name in ("shape_512", "shape_1024", "tex_1024"):
                stage = stages.get(stage_name)
                if isinstance(stage, dict) and isinstance(stage.get("feature_map"), torch.Tensor):
                    stage["feature_map_hr"] = stage["feature_map"]


def _native_four_view_conditioning(
    clip_vision_model,
    fov: float,
    prepared: dict[str, torch.Tensor],
    naf_mode: str,
):
    from comfy_extras.nodes_trellis2 import Pixal3DMultiViewConditioning

    if naf_mode not in NAF_MODES:
        raise ValueError(f"naf_mode must be one of {NAF_MODES}, got {naf_mode!r}")

    kwargs = {
        "front": prepared["front"],
        "left": prepared["left"],
        "back": prepared["back"],
        "right": prepared["right"],
    }
    if naf_mode == "comfy_v040_native":
        return Pixal3DMultiViewConditioning.execute(clip_vision_model, float(fov), **kwargs)

    if naf_mode == "dino_fallback" or getattr(clip_vision_model, "naf", None) is None:
        original_naf = getattr(clip_vision_model, "naf", None)
        clip_vision_model.naf = None
        try:
            result = Pixal3DMultiViewConditioning.execute(
                clip_vision_model, float(fov), **kwargs
            )
        finally:
            clip_vision_model.naf = original_naf
        _enable_dino_projection_fallback(result.result)
        logging.warning(
            "[Pixal3D adapter] NAF mode=dino_fallback: the 2048-channel projection "
            "duplicates DINO features and skips ComfyUI's NAF implementation"
        )
        return result

    # Patch only the single synchronous native-conditioning call, then restore
    # ComfyUI's module.  Other Pixal3D nodes keep their original behaviour.
    import comfy.image_encoders.naf as comfy_naf

    original_na2d = comfy_naf.na2d_pure
    comfy_naf.na2d_pure = stable_na2d_pure_v040
    try:
        result = Pixal3DMultiViewConditioning.execute(
            clip_vision_model, float(fov), **kwargs
        )
    finally:
        comfy_naf.na2d_pure = original_na2d
    logging.info(
        "[Pixal3D adapter] NAF mode=stable_fp32: used cell-wise fp32 QK/softmax "
        "for ComfyUI 0.40 conditioning"
    )
    return result


class Pixal3DFourViewConditioningHunyuan:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_vision_model": ("CLIP_VISION",),
                "front": ("IMAGE",),
                "left": ("IMAGE",),
                "right": ("IMAGE",),
                "back": ("IMAGE",),
                "fov": ("FLOAT", {"default": 20.0, "min": 1.0, "max": 170.0, "step": 0.01}),
                "mask_mode": (MASK_MODES, {"default": "ignore"}),
                "naf_mode": (NAF_MODES, {"default": "stable_fp32"}),
            },
            "optional": {
                "front_mask": ("MASK",),
                "left_mask": ("MASK",),
                "right_mask": ("MASK",),
                "back_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "IMAGE")
    RETURN_NAMES = ("positive", "negative", "prepared_views")
    FUNCTION = "condition"
    CATEGORY = categoryFolder
    DESCRIPTION = (
        "Builds ComfyUI's native Pixal3D multi-view conditioning from four semantic "
        "views. mask_mode is explicit because RemoveBackground uses 1=foreground "
        "while LoadImage alpha masks use 1=transparent. Use ignore after ImageCropToMask."
    )

    def condition(
        self,
        clip_vision_model,
        front,
        left,
        right,
        back,
        fov,
        mask_mode="ignore",
        naf_mode="stable_fp32",
        front_mask=None,
        left_mask=None,
        right_mask=None,
        back_mask=None,
    ):
        prepared = {
            "front": apply_comfy_transparency(front, front_mask, "front", mask_mode),
            "left": apply_comfy_transparency(left, left_mask, "left", mask_mode),
            "right": apply_comfy_transparency(right, right_mask, "right", mask_mode),
            "back": apply_comfy_transparency(back, back_mask, "back", mask_mode),
        }
        result = _native_four_view_conditioning(
            clip_vision_model,
            float(fov),
            prepared,
            naf_mode,
        )
        preview = make_prepared_views_preview(prepared)
        for index, name in enumerate(("front", "left", "right", "back")):
            image = preview[index]
            logging.info(
                "[Pixal3D adapter] prepared %s: min=%.4f max=%.4f mean=%.4f std=%.4f",
                name,
                float(image.amin()),
                float(image.amax()),
                float(image.mean()),
                float(image.std(unbiased=False)),
            )
        return (*result.result, preview)


class Pixal3DAutoAlignView:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shape_latent": ("LATENT",),
                "image": ("IMAGE",),
                "view": (VIEW_NAMES, {"default": "front"}),
                "fov": ("FLOAT", {"default": 20.0, "min": 1.0, "max": 170.0, "step": 0.01}),
                "target_size": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "mask_mode": (MASK_MODES, {"default": "ignore"}),
                "foreground_threshold": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 0.25, "step": 0.001}),
                "silhouette_scale_percent": ("FLOAT", {"default": 100.0, "min": 50.0, "max": 200.0, "step": 0.5}),
                "alignment_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "horizontal_offset_pixels": ("INT", {"default": 0, "min": -2048, "max": 2048, "step": 1}),
                "vertical_offset_pixels": ("INT", {"default": 0, "min": -2048, "max": 2048, "step": 1}),
                "x_deformation_percent": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 200.0, "step": 0.5}),
                "y_deformation_percent": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 200.0, "step": 0.5}),
                "overlay_point_size": ("INT", {"default": 5, "min": 1, "max": 65, "step": 2}),
                "overlay_point_density_percent": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "fit_mode": (("contain", "cover"), {"default": "contain"}),
                "auto_align_enabled": ("BOOLEAN", {"default": True}),
            },
            "optional": {"mask": ("MASK",)},
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("aligned_image", "overlay", "report")
    FUNCTION = "align"
    CATEGORY = categoryFolder
    DESCRIPTION = (
        "Aligns one semantic view to the Hunyuan shape projection. Size, automatic "
        "alignment strength, manual pixel offsets and X/Y deformation are independent "
        "for every view. Automatic fitting can be disabled for fully manual scale, "
        "offset and deformation control."
    )

    def align(
        self,
        shape_latent,
        image,
        view,
        fov,
        target_size,
        mask_mode,
        foreground_threshold,
        silhouette_scale_percent,
        alignment_strength,
        horizontal_offset_pixels,
        vertical_offset_pixels,
        x_deformation_percent,
        y_deformation_percent,
        overlay_point_size,
        overlay_point_density_percent,
        fit_mode,
        auto_align_enabled=True,
        mask=None,
    ):
        result = auto_align_single_view(
            shape_latent,
            image,
            mask,
            view,
            float(fov),
            int(target_size),
            mask_mode,
            float(foreground_threshold),
            float(silhouette_scale_percent),
            float(alignment_strength),
            int(horizontal_offset_pixels),
            int(vertical_offset_pixels),
            float(x_deformation_percent),
            float(y_deformation_percent),
            int(overlay_point_size),
            float(overlay_point_density_percent),
            fit_mode,
            bool(auto_align_enabled),
        )
        logging.info("%s", result[2].replace("\n", " | "))
        return result


class Pixal3DGeometryAlignSettings:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "view": (VIEW_NAMES, {"default": "front"}),
                "global_fit_mode": (GEOMETRY_ALIGN_FIT_MODES, {"default": "contain"}),
                "global_alignment_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "local_alignment_strength": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_local_shift_pixels": ("FLOAT", {"default": 64.0, "min": 0.0, "max": 512.0, "step": 1.0}),
                "boundary_lock_pixels": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 256.0, "step": 1.0}),
                "analysis_size": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 64}),
                "control_grid_size": ("INT", {"default": 32, "min": 8, "max": 96, "step": 4}),
                "optimization_steps": ("INT", {"default": 80, "min": 0, "max": 300, "step": 10}),
                "depth_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.05}),
                "edge_weight": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.05}),
                "silhouette_weight": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 10.0, "step": 0.05}),
                "rigidity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "confidence_threshold": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 0.95, "step": 0.01}),
                "source_depth_near_is_white": ("BOOLEAN", {"default": True}),
                "target_depth_near_is_white": ("BOOLEAN", {"default": True}),
                "background_threshold": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 0.5, "step": 0.001}),
                "output_mask_mode": (GEOMETRY_ALIGN_MASK_MODES, {"default": "target"}),
                "confidence_gate_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "source_depth_blur_pixels": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 32.0, "step": 0.25}),
                "fold_prevention_strength": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 100.0, "step": 0.5}),
            }
        }

    RETURN_TYPES = ("PIXAL3D_GEOMETRY_ALIGN_SETTINGS",)
    RETURN_NAMES = ("settings",)
    FUNCTION = "build"
    CATEGORY = categoryFolder
    DESCRIPTION = (
        "Collects camera-fit, regularized depth-flow, boundary locking and confidence "
        "controls for one Geometry-Aware View Align node."
    )

    def build(
        self,
        view,
        global_fit_mode,
        global_alignment_strength,
        local_alignment_strength,
        max_local_shift_pixels,
        boundary_lock_pixels,
        analysis_size,
        control_grid_size,
        optimization_steps,
        depth_weight,
        edge_weight,
        silhouette_weight,
        rigidity,
        confidence_threshold,
        source_depth_near_is_white,
        target_depth_near_is_white,
        background_threshold,
        output_mask_mode,
        confidence_gate_strength,
        source_depth_blur_pixels,
        fold_prevention_strength,
    ):
        settings = GeometryAlignSettings(
            view=str(view),
            global_fit_mode=str(global_fit_mode),
            global_alignment_strength=min(1.0, max(0.0, float(global_alignment_strength))),
            local_alignment_strength=min(1.0, max(0.0, float(local_alignment_strength))),
            max_local_shift_pixels=max(0.0, float(max_local_shift_pixels)),
            boundary_lock_pixels=max(0.0, float(boundary_lock_pixels)),
            analysis_size=max(256, int(analysis_size)),
            control_grid_size=max(8, int(control_grid_size)),
            optimization_steps=max(0, int(optimization_steps)),
            depth_weight=max(0.0, float(depth_weight)),
            edge_weight=max(0.0, float(edge_weight)),
            silhouette_weight=max(0.0, float(silhouette_weight)),
            rigidity=max(0.0, float(rigidity)),
            confidence_threshold=min(0.95, max(0.0, float(confidence_threshold))),
            source_depth_near_is_white=bool(source_depth_near_is_white),
            target_depth_near_is_white=bool(target_depth_near_is_white),
            background_threshold=min(1.0, max(0.0, float(background_threshold))),
            output_mask_mode=str(output_mask_mode),
            confidence_gate_strength=min(1.0, max(0.0, float(confidence_gate_strength))),
            source_depth_blur_pixels=max(0.0, float(source_depth_blur_pixels)),
            fold_prevention_strength=max(0.0, float(fold_prevention_strength)),
        )
        if settings.view not in VIEW_NAMES:
            raise ValueError(f"Unknown geometry align view: {settings.view!r}")
        if settings.global_fit_mode not in GEOMETRY_ALIGN_FIT_MODES:
            raise ValueError(f"Unknown geometry fit mode: {settings.global_fit_mode!r}")
        if settings.output_mask_mode not in GEOMETRY_ALIGN_MASK_MODES:
            raise ValueError(f"Unknown geometry output mask mode: {settings.output_mask_mode!r}")
        return (settings,)


class Pixal3DGeometryAwareViewAlign:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "source_depth": ("IMAGE",),
                "target_depth": ("IMAGE",),
                "settings": ("PIXAL3D_GEOMETRY_ALIGN_SETTINGS",),
            },
            "optional": {
                "source_mask": ("MASK",),
                "target_mask": ("MASK",),
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
        "MASK",
        "IMAGE",
        "MASK",
        "IMAGE",
        "IMAGE",
        "STRING",
    )
    RETURN_NAMES = (
        "aligned_image",
        "aligned_rgba",
        "aligned_mask",
        "warped_source_depth",
        "confidence_mask",
        "flow_preview",
        "depth_overlay",
        "report",
    )
    FUNCTION = "align"
    CATEGORY = categoryFolder
    DESCRIPTION = (
        "Fits one source RGB/depth view to the model's rendered depth using global "
        "silhouette placement plus a confidence-gated, contour-locked PyTorch flow "
        "grid. Use one node per Front/Left/Right/Back view."
    )

    def align(
        self,
        image,
        source_depth,
        target_depth,
        settings,
        source_mask=None,
        target_mask=None,
    ):
        result = geometry_aware_view_align(
            image,
            source_depth,
            target_depth,
            settings,
            source_mask,
            target_mask,
        )
        logging.info("%s", result[-1].replace("\n", " | "))
        return result


class Pixal3DHumanoidAlignSettings:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "view": (VIEW_NAMES, {"default": "front"}),
                "global_model": (HUMANOID_GLOBAL_MODELS, {"default": "similarity"}),
                "score_threshold": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "global_alignment_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "torso_strength": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.5, "step": 0.05}),
                "arm_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.5, "step": 0.05}),
                "leg_strength": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.5, "step": 0.05}),
                "head_strength": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.5, "step": 0.05}),
                "hand_strength": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.5, "step": 0.05}),
                "foot_strength": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.5, "step": 0.05}),
                "tps_regularization": ("FLOAT", {"default": 0.002, "min": 0.000001, "max": 0.1, "step": 0.0005}),
                "max_local_shift_pixels": ("FLOAT", {"default": 192.0, "min": 0.0, "max": 1024.0, "step": 1.0}),
                "outer_stability": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05}),
                "fold_protection": ("BOOLEAN", {"default": True}),
                "max_folded_area_percent": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 5.0, "step": 0.01}),
                "background_threshold": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 0.5, "step": 0.001}),
                "output_mask_mode": (GEOMETRY_ALIGN_MASK_MODES, {"default": "target"}),
                "source_person_index": ("INT", {"default": -1, "min": -1, "max": 15, "step": 1}),
                "target_person_index": ("INT", {"default": -1, "min": -1, "max": 15, "step": 1}),
            }
        }

    RETURN_TYPES = ("PIXAL3D_HUMANOID_ALIGN_SETTINGS",)
    RETURN_NAMES = ("settings",)
    FUNCTION = "build"
    CATEGORY = categoryFolder
    DESCRIPTION = (
        "Packages anatomy-aware alignment controls. Arms, legs, torso, head, hands "
        "and feet have independent strengths; -1 selects the largest detected person."
    )

    def build(
        self,
        view,
        global_model,
        score_threshold,
        global_alignment_strength,
        torso_strength,
        arm_strength,
        leg_strength,
        head_strength,
        hand_strength,
        foot_strength,
        tps_regularization,
        max_local_shift_pixels,
        outer_stability,
        fold_protection,
        max_folded_area_percent,
        background_threshold,
        output_mask_mode,
        source_person_index,
        target_person_index,
    ):
        settings = HumanoidAlignSettings(
            view=str(view),
            global_model=str(global_model),
            score_threshold=min(1.0, max(0.0, float(score_threshold))),
            global_alignment_strength=min(1.0, max(0.0, float(global_alignment_strength))),
            torso_strength=min(1.5, max(0.0, float(torso_strength))),
            arm_strength=min(1.5, max(0.0, float(arm_strength))),
            leg_strength=min(1.5, max(0.0, float(leg_strength))),
            head_strength=min(1.5, max(0.0, float(head_strength))),
            hand_strength=min(1.5, max(0.0, float(hand_strength))),
            foot_strength=min(1.5, max(0.0, float(foot_strength))),
            tps_regularization=max(1e-6, float(tps_regularization)),
            max_local_shift_pixels=max(0.0, float(max_local_shift_pixels)),
            outer_stability=min(1.0, max(0.0, float(outer_stability))),
            fold_protection=bool(fold_protection),
            max_folded_area_percent=min(5.0, max(0.0, float(max_folded_area_percent))),
            background_threshold=min(1.0, max(0.0, float(background_threshold))),
            output_mask_mode=str(output_mask_mode),
            source_person_index=max(-1, int(source_person_index)),
            target_person_index=max(-1, int(target_person_index)),
        )
        if settings.view not in VIEW_NAMES:
            raise ValueError(f"Unknown humanoid align view: {settings.view!r}")
        if settings.global_model not in HUMANOID_GLOBAL_MODELS:
            raise ValueError(f"Unknown humanoid global model: {settings.global_model!r}")
        if settings.output_mask_mode not in GEOMETRY_ALIGN_MASK_MODES:
            raise ValueError(f"Unknown humanoid output mask mode: {settings.output_mask_mode!r}")
        return (settings,)


class Pixal3DHumanoidPoseReference:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth_map": ("IMAGE",),
                "mesh_mask": ("MASK",),
                "relief_strength": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 12.0, "step": 0.1}),
                "edge_strength": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "background_level": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("pose_reference",)
    FUNCTION = "render"
    CATEGORY = categoryFolder
    DESCRIPTION = (
        "Builds a neutral shaded character projection from Shape View Maps depth and "
        "mesh mask. Feed it to a second SDPose Keypoint Extractor for target landmarks."
    )

    def render(self, depth_map, mesh_mask, relief_strength, edge_strength, background_level):
        return (
            make_humanoid_pose_reference(
                depth_map,
                mesh_mask,
                float(relief_strength),
                float(edge_strength),
                float(background_level),
            ),
        )


class Pixal3DSAM3DHumanoidPoseProjector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mhr_pose_data": ("MHR_POSE_DATA",),
                "target_mask": ("MASK",),
                "source_view": (VIEW_NAMES, {"default": "front"}),
                "target_view": (VIEW_NAMES, {"default": "left"}),
                "skeleton_height_percent": ("FLOAT", {"default": 84.0, "min": 50.0, "max": 120.0, "step": 0.5}),
                "horizontal_scale_percent": ("FLOAT", {"default": 100.0, "min": 25.0, "max": 200.0, "step": 1.0}),
                "horizontal_offset_pixels": ("FLOAT", {"default": 0.0, "min": -1024.0, "max": 1024.0, "step": 1.0}),
                "vertical_offset_pixels": ("FLOAT", {"default": 0.0, "min": -1024.0, "max": 1024.0, "step": 1.0}),
                "mirror_side_projection": ("BOOLEAN", {"default": False}),
                "include_hands": ("BOOLEAN", {"default": False}),
                "person_index": ("INT", {"default": -1, "min": -1, "max": 15, "step": 1}),
            }
        }

    RETURN_TYPES = ("POSE_KEYPOINT", "IMAGE", "STRING")
    RETURN_NAMES = ("target_pose", "pose_preview", "report")
    FUNCTION = "project"
    CATEGORY = categoryFolder
    DESCRIPTION = (
        "Projects the complete 3D skeleton predicted from a real source view by "
        "ComfyUI SAM3D Body into Front/Left/Right/Back and fits it to the target "
        "mesh mask. This avoids running a 2D pose detector on a synthetic side render."
    )

    def project(
        self,
        mhr_pose_data,
        target_mask,
        source_view,
        target_view,
        skeleton_height_percent,
        horizontal_scale_percent,
        horizontal_offset_pixels,
        vertical_offset_pixels,
        mirror_side_projection,
        include_hands,
        person_index,
    ):
        result = project_sam3d_humanoid_pose(
            mhr_pose_data,
            target_mask,
            source_view,
            target_view,
            float(skeleton_height_percent),
            float(horizontal_scale_percent),
            float(horizontal_offset_pixels),
            float(vertical_offset_pixels),
            bool(mirror_side_projection),
            bool(include_hands),
            int(person_index),
        )
        logging.info("%s", result[-1].replace("\n", " | "))
        return result


class Pixal3DHumanoidPoseViewAlign:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "source_pose": ("POSE_KEYPOINT",),
                "target_reference": ("IMAGE",),
                "target_pose": ("POSE_KEYPOINT",),
                "settings": ("PIXAL3D_HUMANOID_ALIGN_SETTINGS",),
            },
            "optional": {
                "source_mask": ("MASK",),
                "target_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = (
        "aligned_image",
        "aligned_rgba",
        "aligned_mask",
        "deformation_preview",
        "landmark_overlay",
        "report",
    )
    FUNCTION = "align"
    CATEGORY = categoryFolder
    DESCRIPTION = (
        "Character-specific view alignment driven by matching SDPose joints. It uses "
        "anatomical correspondences for arms and legs instead of asking depth alone "
        "to discover semantic parts. Use one instance per camera view."
    )

    def align(
        self,
        image,
        source_pose,
        target_reference,
        target_pose,
        settings,
        source_mask=None,
        target_mask=None,
    ):
        result = humanoid_pose_view_align(
            image,
            source_pose,
            target_reference,
            target_pose,
            settings,
            source_mask,
            target_mask,
        )
        logging.info("%s", result[-1].replace("\n", " | "))
        return result
class Pixal3DShapeViewMaps:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shape_latent": ("LATENT",),
                "mesh": ("MESH",),
                "view": (VIEW_NAMES, {"default": "front"}),
                "fov": ("FLOAT", {"default": 20.0, "min": 1.0, "max": 170.0, "step": 0.01}),
                "target_size": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "mask_expand_pixels": ("INT", {"default": 0, "min": -128, "max": 128, "step": 1}),
                "mask_feather_pixels": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 32.0, "step": 0.25}),
                "depth_source": (("mesh", "voxel"), {"default": "mesh"}),
                "depth_near_is_white": ("BOOLEAN", {"default": True}),
                "depth_gamma": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 4.0, "step": 0.05}),
                "canny_source": (("depth_map", "mesh_mask", "voxel_mask"), {"default": "depth_map"}),
                "canny_low_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.99, "step": 0.01}),
                "canny_high_threshold": ("FLOAT", {"default": 0.2, "min": 0.01, "max": 1.0, "step": 0.01}),
                "canny_preblur_pixels": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 16.0, "step": 0.25}),
                "canny_dilate_pixels": ("INT", {"default": 0, "min": 0, "max": 16, "step": 1}),
            }
        }

    RETURN_TYPES = ("MASK", "MASK", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("voxel_mask", "mesh_mask", "depth_map", "canny", "report")
    FUNCTION = "generate"
    CATEGORY = categoryFolder
    DESCRIPTION = (
        "Renders one selected Pixal3D camera view from the sparse occupancy and the "
        "normalized Hunyuan mesh. Outputs both masks, geometric z-buffer depth and a "
        "configurable Canny conditioning image without modifying Auto Align."
    )

    def generate(
        self,
        shape_latent,
        mesh,
        view,
        fov,
        target_size,
        mask_expand_pixels,
        mask_feather_pixels,
        depth_source,
        depth_near_is_white,
        depth_gamma,
        canny_source,
        canny_low_threshold,
        canny_high_threshold,
        canny_preblur_pixels,
        canny_dilate_pixels,
    ):
        result = make_shape_view_maps(
            shape_latent,
            mesh,
            view,
            float(fov),
            int(target_size),
            int(mask_expand_pixels),
            float(mask_feather_pixels),
            depth_source,
            bool(depth_near_is_white),
            float(depth_gamma),
            canny_source,
            float(canny_low_threshold),
            float(canny_high_threshold),
            float(canny_preblur_pixels),
            int(canny_dilate_pixels),
        )
        logging.info("%s", result[4].replace("\n", " | "))
        return result


class Pixal3DBatchFourViews:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "front": ("IMAGE",),
                "left": ("IMAGE",),
                "right": ("IMAGE",),
                "back": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("views_f_l_r_b",)
    FUNCTION = "batch"
    CATEGORY = categoryFolder
    DESCRIPTION = (
        "Combines four equally sized images into the exact semantic batch order "
        "Front, Left, Right, Back used by the reprojection node."
    )

    def batch(self, front, left, right, back):
        return (batch_four_views(front, left, right, back),)


class Pixal3DAutoAlignFourViews:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shape_latent": ("LATENT",),
                "front": ("IMAGE",),
                "left": ("IMAGE",),
                "right": ("IMAGE",),
                "back": ("IMAGE",),
                "fov": ("FLOAT", {"default": 20.0, "min": 1.0, "max": 170.0, "step": 0.01}),
                "target_size": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "mask_mode": (MASK_MODES, {"default": "ignore"}),
                "foreground_threshold": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 0.25, "step": 0.001}),
                "silhouette_scale_percent": ("FLOAT", {"default": 100.0, "min": 70.0, "max": 130.0, "step": 0.5}),
                "alignment_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "fit_mode": (("contain", "cover"), {"default": "contain"}),
            },
            "optional": {
                "front_mask": ("MASK",),
                "left_mask": ("MASK",),
                "right_mask": ("MASK",),
                "back_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = (
        "front",
        "left",
        "right",
        "back",
        "aligned_views",
        "overlays",
        "report",
    )
    FUNCTION = "align"
    CATEGORY = categoryFolder
    DESCRIPTION = (
        "Uniformly scales and translates four foregrounds so their bounds match the "
        "exact Pixal3D camera projection of the encoded Hunyuan shape. Outputs both "
        "individual views for conditioning and a high-resolution F/L/R/B batch for reprojection."
    )

    def align(
        self,
        shape_latent,
        front,
        left,
        right,
        back,
        fov,
        target_size,
        mask_mode,
        foreground_threshold,
        silhouette_scale_percent,
        alignment_strength,
        fit_mode,
        front_mask=None,
        left_mask=None,
        right_mask=None,
        back_mask=None,
    ):
        aligned, batch, overlays, report = auto_align_four_views(
            shape_latent,
            (front, left, right, back),
            (front_mask, left_mask, right_mask, back_mask),
            float(fov),
            int(target_size),
            mask_mode,
            float(foreground_threshold),
            float(silhouette_scale_percent),
            float(alignment_strength),
            fit_mode,
        )
        logging.info("%s", report.replace("\n", " | "))
        return (*aligned, batch, overlays, report)


class HunyuanMeshToPixal3DShapeLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shape_encoder": ("TRELLIS2_SHAPE_ENCODER",),
                "mesh": ("MESH",),
                "mesh_yaw_degrees": (
                    "INT",
                    {"default": 0, "min": -180, "max": 180, "step": 90},
                ),
                "voxel_resolution": (
                    ("1024", "1536"),
                    {"default": "1024"},
                ),
            }
        }

    RETURN_TYPES = ("LATENT", "SHAPE_SUBDIVIDES", "MESH", "PIXAL_MESH_TRANSFORM")
    RETURN_NAMES = ("shape_latent", "shape_subdivides", "normalized_mesh", "mesh_transform")
    FUNCTION = "encode"
    CATEGORY = categoryFolder
    DESCRIPTION = (
        "Normalizes and encodes one Hunyuan/ComfyUI MESH into a TRELLIS.2 shape "
        "latent. Use the latent with the native Pixal3D Multi-View Conditioning and "
        "Trellis2 Texture Stage nodes."
    )

    def encode(self, shape_encoder, mesh, mesh_yaw_degrees, voxel_resolution="1024"):

        voxel_resolution = int(voxel_resolution)
        normalized_mesh, transform = normalize_mesh(mesh, mesh_yaw_degrees)
        vertices, faces = _single_mesh_tensors(normalized_mesh)
        shape_latent, subdivides = encode_external_mesh(
            shape_encoder, vertices, faces, resolution=voxel_resolution
        )
        return shape_latent, subdivides, normalized_mesh, transform


class Pixal3DRebuildSparseVoxel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "voxel": ("VOXEL",),
                "mesh": ("MESH",),
                "target_resolution": (
                    ("1536", "2048"),
                    {"default": "2048"},
                ),
                "transfer_mode": (
                    ("edge_aware", "nearest"),
                    {"default": "edge_aware"},
                ),
                "max_search_radius_voxels": (
                    "INT",
                    {"default": 4, "min": 1, "max": 8, "step": 1},
                ),
                "rgb_edge_threshold": (
                    "FLOAT",
                    {"default": 0.20, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "pbr_edge_threshold": (
                    "FLOAT",
                    {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("VOXEL", "STRING")
    RETURN_NAMES = ("voxel", "report")
    FUNCTION = "rebuild"
    CATEGORY = categoryFolder
    DESCRIPTION = (
        "Re-voxelizes the normalized Hunyuan mesh at 1536 or 2048 and transfers "
        "every decoded texture/PBR channel from the lower-resolution sparse field. "
        "Edge-aware interpolation rejects neighbours across sharp colour/material "
        "changes. Place it before Image-Guided Voxel Reprojection."
    )

    @torch.no_grad()
    def rebuild(
        self,
        voxel,
        mesh,
        target_resolution,
        transfer_mode,
        max_search_radius_voxels,
        rgb_edge_threshold,
        pbr_edge_threshold,
    ):

        source_coords = voxel.data
        source_features = voxel.voxel_colors
        source_resolution = int(voxel.resolution)
        target_resolution = int(target_resolution)
        if not isinstance(source_coords, torch.Tensor) or not isinstance(
            source_features, torch.Tensor
        ):
            raise ValueError("voxel must contain tensor data and voxel_colors")
        if target_resolution < source_resolution:
            raise ValueError(
                "target_resolution must be greater than or equal to the decoded "
                f"voxel resolution ({source_resolution})"
            )

        vertices, faces = _single_mesh_tensors(mesh)
        target_spatial, dual_vertices, intersected = mesh_to_flexible_dual_grid(
            vertices, faces, target_resolution
        )
        target_spatial = target_spatial.to(torch.int32).cpu()
        del dual_vertices, intersected

        compute_device = source_features.device
        try:
            import comfy.model_management

            preferred_device = comfy.model_management.get_torch_device()
            if getattr(preferred_device, "type", "cpu") != "cpu":
                compute_device = preferred_device
        except (ImportError, AttributeError, RuntimeError):
            pass

        transferred, stats = transfer_sparse_voxel_features(
            source_coords.to(compute_device),
            source_features.to(compute_device),
            source_resolution,
            target_spatial.to(compute_device),
            target_resolution,
            transfer_mode,
            int(max_search_radius_voxels),
            float(rgb_edge_threshold),
            float(pbr_edge_threshold),
        )

        target_data = target_spatial.to(device=source_coords.device, dtype=source_coords.dtype)
        if source_coords.ndim == 2 and source_coords.shape[1] == 4:
            target_data = torch.cat(
                [torch.zeros_like(target_data[:, :1]), target_data], dim=1
            )
        transferred = transferred.to(
            device=source_features.device, dtype=source_features.dtype
        )
        rebuilt = copy.copy(voxel)
        rebuilt.data = target_data
        rebuilt.voxel_colors = transferred
        rebuilt.resolution = target_resolution

        target_count = max(1, int(target_spatial.shape[0]))
        matched = stats["primary"] + stats["fallback"]
        report = "\n".join(
            [
                "Pixal3D sparse voxel rebuild",
                f"resolution={source_resolution} -> {target_resolution}",
                f"voxels={int(source_features.shape[0]):,} -> {int(target_spatial.shape[0]):,}, channels={int(source_features.shape[1])}",
                f"transfer={transfer_mode}, compute_device={compute_device}",
                f"primary={stats['primary']:,}, fallback={stats['fallback']:,}, unmatched={stats['unmatched']:,}, matched={matched / target_count:.4%}",
                f"edge_rejected={stats['edge_rejected']:,}, rgb_threshold={float(rgb_edge_threshold):.3f}, pbr_threshold={float(pbr_edge_threshold):.3f}, search_radius={int(max_search_radius_voxels)}",
                "all decoded channels transferred; later RGB reprojection may replace channels 0..2 only",
            ]
        )
        logging.info("%s", report.replace("\n", " | "))
        return rebuilt, report


class RestoreHunyuanMeshTransform:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "mesh_transform": ("PIXAL_MESH_TRANSFORM",),
            }
        }

    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "restore"
    CATEGORY = categoryFolder
    DESCRIPTION = (
        "Restores the original Hunyuan mesh scale, position and yaw after UV unwrap, "
        "voxel texture baking and Apply Texture to Mesh."
    )

    def restore(self, mesh, mesh_transform):
        return (restore_mesh(mesh, mesh_transform),)


class PredictTrellis2ShapeSubdivides:
    """Predict decoder topology with the official shape VAE, without extracting a mesh."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shape_latent": ("LATENT",),
                "shape_vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("SHAPE_SUBDIVIDES",)
    RETURN_NAMES = ("shape_subdivides",)
    FUNCTION = "predict"
    CATEGORY = categoryFolder
    DESCRIPTION = (
        "Runs the native TRELLIS.2 shape decoder only far enough to predict its four "
        "subdivision maps. Use these maps for VAE Decode Texture; the decoded shape "
        "mesh is deliberately not extracted and the Hunyuan geometry remains unchanged."
    )

    @torch.no_grad()
    def predict(self, shape_latent, shape_vae):
        import comfy.latent_formats
        import comfy.model_management
        from comfy.ldm.trellis2.vae import SparseTensor, SparseUnetVaeDecoder

        samples = shape_latent["samples"]
        coords = shape_latent["coords"]
        counts = shape_latent.get("coord_counts")
        if samples.ndim != 4 or samples.shape[0] != 1:
            raise ValueError("External-mesh subdivision prediction supports one mesh per run")
        count = int(counts[0]) if counts is not None else int(coords.shape[0])
        if counts is not None and (counts.numel() != 1 or count != int(coords.shape[0])):
            raise ValueError("shape_latent coord_counts does not describe exactly one mesh")
        if count <= 0:
            raise ValueError("shape_latent has no sparse tokens")

        resolution = int(shape_latent.get("coord_resolution", 64)) * 16
        point_estimate = resolution * resolution * 5 // 4
        dtype_bytes = comfy.model_management.dtype_size(shape_vae.vae_dtype)
        memory_required = 2 * 1024**3 + point_estimate * (896 * dtype_bytes + 27 * 4)
        shape_vae.prepare_decode(samples.shape, memory_required=memory_required)
        bundle = shape_vae.first_stage_model
        if not hasattr(bundle, "shape_dec"):
            raise ValueError(
                "shape_vae must be trellis_2_shape_vae_bf16.safetensors, not the texture VAE"
            )

        device = comfy.model_management.get_torch_device()
        token_feats = samples.squeeze(-1).transpose(1, 2)[0, :count].to(device)
        raw_feats = comfy.latent_formats.Trellis2ShapeSLAT().process_out(token_feats)
        token_coords = coords[:count].to(device=device, dtype=torch.int32).clone()
        token_coords[:, 0] = 0
        slat = SparseTensor(feats=raw_feats, coords=token_coords)
        bundle.shape_dec.set_resolution(resolution)
        decoded, subdivisions = SparseUnetVaeDecoder.forward(
            bundle.shape_dec,
            slat.to(shape_vae.vae_dtype),
            return_subs=True,
        )
        del decoded

        output_device = comfy.model_management.intermediate_device()
        result = [
            SparseTensor(
                feats=sub.feats.to(output_device),
                coords=sub.coords.to(device=output_device, dtype=torch.int32),
                shape=sub.shape,
                scale=sub._scale,
            )
            for sub in subdivisions
        ]
        logging.info(
            "[Pixal3D adapter] shape-VAE subdivisions: latent=%d stages=%s resolution=%d",
            count,
            [int(sub.coords.shape[0]) for sub in result],
            resolution,
        )
        return (result,)


class Pixal3DReprojectionViewSettings:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "view": (VIEW_NAMES, {"default": "front"}),
                "enabled": ("BOOLEAN", {"default": True}),
                "weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                "coverage_degrees": ("FLOAT", {"default": 120.0, "min": 1.0, "max": 180.0, "step": 1.0}),
                "edge_softness_percent": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "background_threshold": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 0.25, "step": 0.001}),
                "blend_mode": (BLEND_MODES, {"default": "normal"}),
                "gaussian_blur_target": (GAUSSIAN_BLUR_TARGETS, {"default": "blend_mask"}),
                "gaussian_blur_sigma_pixels": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 32.0, "step": 0.25}),
                "flip_horizontal": ("BOOLEAN", {"default": False}),
                "edge_feather_pixels": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 128.0, "step": 0.5}),
                "crop_left_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 95.0, "step": 0.5}),
                "crop_right_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 95.0, "step": 0.5}),
                "crop_top_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 95.0, "step": 0.5}),
                "crop_bottom_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 95.0, "step": 0.5}),
                "coverage_mode": (COVERAGE_MODES, {"default": "angular_sector"}),
                "camera_feather_pixels": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 256.0, "step": 1.0}),
                "camera_feather_inset_pixels": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 256.0, "step": 1.0}),
            }
        }

    RETURN_TYPES = ("PIXAL3D_REPROJECTION_VIEW",)
    RETURN_NAMES = ("view_settings",)
    FUNCTION = "build"
    CATEGORY = categoryFolder
    DESCRIPTION = (
        "Collects every camera-specific reprojection control into one compact pipeline. "
        "Connect it to the matching Front, Left, Right or Back settings input. "
        "Independent L/R/T/B crops mask unwanted parts without rescaling the view. "
        "all_camera_visible covers the complete z-buffer-visible camera silhouette; "
        "camera feather inset and width control its transparent-to-opaque edge."
    )

    def build(
        self,
        view,
        enabled,
        weight,
        coverage_degrees,
        edge_softness_percent,
        background_threshold,
        blend_mode,
        gaussian_blur_target,
        gaussian_blur_sigma_pixels,
        flip_horizontal,
        edge_feather_pixels,
        crop_left_percent=0.0,
        crop_right_percent=0.0,
        crop_top_percent=0.0,
        crop_bottom_percent=0.0,
        coverage_mode="angular_sector",
        camera_feather_pixels=0.0,
        camera_feather_inset_pixels=0.0,
    ):
        crop_values = tuple(
            min(95.0, max(0.0, float(value)))
            for value in (
                crop_left_percent,
                crop_right_percent,
                crop_top_percent,
                crop_bottom_percent,
            )
        )
        if crop_values[0] + crop_values[1] >= 100.0:
            raise ValueError("crop_left_percent + crop_right_percent must be below 100")
        if crop_values[2] + crop_values[3] >= 100.0:
            raise ValueError("crop_top_percent + crop_bottom_percent must be below 100")
        settings = ReprojectionViewSettings(
            view=str(view),
            enabled=bool(enabled),
            weight=max(0.0, float(weight)),
            coverage_degrees=min(180.0, max(1.0, float(coverage_degrees))),
            edge_softness_percent=min(100.0, max(0.0, float(edge_softness_percent))),
            background_threshold=min(1.0, max(0.0, float(background_threshold))),
            blend_mode=str(blend_mode),
            gaussian_blur_target=str(gaussian_blur_target),
            gaussian_blur_sigma_pixels=max(0.0, float(gaussian_blur_sigma_pixels)),
            flip_horizontal=bool(flip_horizontal),
            edge_feather_pixels=max(0.0, float(edge_feather_pixels)),
            crop_left_percent=crop_values[0],
            crop_right_percent=crop_values[1],
            crop_top_percent=crop_values[2],
            crop_bottom_percent=crop_values[3],
            coverage_mode=str(coverage_mode),
            camera_feather_pixels=max(0.0, float(camera_feather_pixels)),
            camera_feather_inset_pixels=max(
                0.0, float(camera_feather_inset_pixels)
            ),
        )
        _validate_view_settings(settings, str(view))
        return (settings,)


class Pixal3DImageGuidedVoxelReprojection:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "voxel": ("VOXEL",),
                "prepared_views": ("IMAGE",),
                "front_settings": ("PIXAL3D_REPROJECTION_VIEW",),
                "left_settings": ("PIXAL3D_REPROJECTION_VIEW",),
                "right_settings": ("PIXAL3D_REPROJECTION_VIEW",),
                "back_settings": ("PIXAL3D_REPROJECTION_VIEW",),
                "fov": ("FLOAT", {"default": 20.0, "min": 1.0, "max": 170.0, "step": 0.01}),
                "layer_order_bottom_to_top": (VIEW_LAYER_ORDERS, {"default": DEFAULT_VIEW_LAYER_ORDER}),
                "reprojection_strength": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01}),
                "visibility_tolerance_voxels": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 16.0, "step": 0.25}),
                "generated_texture_strength_percent": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "uncovered_color": ("COLOR", {"default": "#808080"}),
                "camera_composite_mode": (
                    CAMERA_COMPOSITE_MODES,
                    {"default": "ordered_layers"},
                ),
            }
        }

    RETURN_TYPES = ("VOXEL", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("voxel", "projections", "camera_mix", "report")
    FUNCTION = "reproject"
    CATEGORY = categoryFolder
    DESCRIPTION = (
        "Projects high-resolution F/L/R/B RGB directly onto visible decoded voxels, "
        "while preserving Metallic, Roughness and all channels after RGB. Camera-specific "
        "controls arrive through four compact settings pipelines. Use weighted_average "
        "for normalized, order-independent seams or ordered_layers for Photoshop-style "
        "bottom-to-top compositing."
    )

    def reproject(
        self,
        voxel,
        prepared_views,
        front_settings,
        left_settings,
        right_settings,
        back_settings,
        fov,
        layer_order_bottom_to_top,
        reprojection_strength,
        visibility_tolerance_voxels,
        generated_texture_strength_percent=100.0,
        uncovered_color="#808080",
        camera_composite_mode="ordered_layers",
    ):
        settings = tuple(
            _validate_view_settings(value, expected)
            for value, expected in zip(
                (front_settings, left_settings, right_settings, back_settings),
                VIEW_NAMES,
            )
        )
        result = reproject_voxel_rgb(
            voxel,
            prepared_views,
            float(fov),
            float(reprojection_strength),
            tuple(item.weight if item.enabled else 0.0 for item in settings),
            tuple(item.coverage_degrees for item in settings),
            tuple(item.edge_softness_percent for item in settings),
            float(visibility_tolerance_voxels),
            tuple(item.background_threshold for item in settings),
            tuple(item.blend_mode for item in settings),
            tuple(item.gaussian_blur_target for item in settings),
            tuple(item.gaussian_blur_sigma_pixels for item in settings),
            float(generated_texture_strength_percent),
            uncovered_color,
            tuple(item.flip_horizontal for item in settings),
            tuple(item.edge_feather_pixels for item in settings),
            layer_order_bottom_to_top,
            view_crop_left_percent=tuple(item.crop_left_percent for item in settings),
            view_crop_right_percent=tuple(item.crop_right_percent for item in settings),
            view_crop_top_percent=tuple(item.crop_top_percent for item in settings),
            view_crop_bottom_percent=tuple(item.crop_bottom_percent for item in settings),
            view_coverage_modes=tuple(item.coverage_mode for item in settings),
            view_camera_feather_pixels=tuple(
                item.camera_feather_pixels for item in settings
            ),
            view_camera_feather_inset_pixels=tuple(
                item.camera_feather_inset_pixels for item in settings
            ),
            camera_composite_mode=str(camera_composite_mode),
        )
        logging.info("%s", result[3].replace("\n", " | "))
        return result


class Pixal3DImageGuidedVoxelReprojectionView:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "voxel": ("VOXEL",),
                "image": ("IMAGE",),
                "view": (VIEW_NAMES, {"default": "front"}),
                "fov": ("FLOAT", {"default": 20.0, "min": 1.0, "max": 170.0, "step": 0.01}),
                "blend_mode": (BLEND_MODES, {"default": "normal"}),
                "opacity": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01}),
                "coverage_degrees": ("FLOAT", {"default": 120.0, "min": 1.0, "max": 180.0, "step": 1.0}),
                "edge_softness_percent": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "visibility_tolerance_voxels": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 16.0, "step": 0.25}),
                "background_threshold": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 0.25, "step": 0.001}),
                "gaussian_blur_target": (GAUSSIAN_BLUR_TARGETS, {"default": "blend_mask"}),
                "gaussian_blur_sigma_pixels": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 32.0, "step": 0.25}),
            }
        }

    RETURN_TYPES = ("VOXEL", "STRING")
    RETURN_NAMES = ("voxel", "report")
    FUNCTION = "reproject"
    CATEGORY = categoryFolder
    DESCRIPTION = (
        "Projects one aligned camera image as one ordered layer onto decoded voxels. "
        "Chain Front/Left/Right/Back instances for independent Photoshop-style blend "
        "modes, opacity, angular coverage and Gaussian blur. PBR channels are preserved."
    )

    def reproject(
        self,
        voxel,
        image,
        view,
        fov,
        blend_mode,
        opacity,
        coverage_degrees,
        edge_softness_percent,
        visibility_tolerance_voxels,
        background_threshold,
        gaussian_blur_target,
        gaussian_blur_sigma_pixels,
    ):
        result = reproject_single_view_rgb(
            voxel,
            image,
            view,
            float(fov),
            blend_mode,
            float(opacity),
            float(coverage_degrees),
            float(edge_softness_percent),
            float(visibility_tolerance_voxels),
            float(background_threshold),
            gaussian_blur_target,
            float(gaussian_blur_sigma_pixels),
        )
        logging.info("%s", result[1].replace("\n", " | "))
        return result


class Pixal3DVoxelDiagnostics:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"voxel": ("VOXEL",)}}

    RETURN_TYPES = ("VOXEL", "STRING", "VOXEL", "IMAGE")
    RETURN_NAMES = ("voxel", "report", "coordinate_test", "projections")
    FUNCTION = "inspect"
    CATEGORY = categoryFolder
    DESCRIPTION = (
        "Passes the decoded voxel through, reports channel statistics, renders four RGB "
        "projections before UV baking, and makes an XYZ-gradient control voxel."
    )

    def inspect(self, voxel):
        report = voxel_diagnostic_text(voxel)
        logging.info("%s", report.replace("\n", " | "))
        coordinate_test = make_coordinate_test_voxel(voxel)
        projections = make_voxel_projection_preview(voxel)
        return voxel, report, coordinate_test, projections


class Pixal3DLatentDiagnostics:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "kind": (("shape", "texture"), {"default": "shape"}),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "report")
    FUNCTION = "inspect"
    CATEGORY = categoryFolder
    DESCRIPTION = (
        "Passes a sparse TRELLIS/Pixal3D latent through and reports its value range, "
        "coordinate range, and X/Y/Z neighbour discontinuity before VAE decoding."
    )

    def inspect(self, latent, kind):
        report = latent_diagnostic_text(latent, kind)
        logging.info("%s", report.replace("\n", " | "))
        return latent, report


class Pixal3DProjectionDiagnostics:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shape_latent": ("LATENT",),
                "prepared_views": ("IMAGE",),
                "fov": ("FLOAT", {"default": 20.0, "min": 1.0, "max": 170.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("overlays", "raw_rgb_projections", "report")
    FUNCTION = "inspect"
    CATEGORY = categoryFolder
    DESCRIPTION = (
        "Projects sparse shape tokens onto the prepared F/L/R/B images and directly "
        "samples their RGB. This tests camera/framing/axis alignment without DINO, "
        "diffusion sampling, texture VAE, UVs, or baking. Keep fov equal to conditioning."
    )

    def inspect(self, shape_latent, prepared_views, fov):
        overlays, raw_rgb_projections, report = make_projection_diagnostics(
            shape_latent, prepared_views, float(fov)
        )
        logging.info("%s", report.replace("\n", " | "))
        return overlays, raw_rgb_projections, report


class Pixal3DProjectedFeatureDiagnostics:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "shape_latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "STRING", "IMAGE")
    RETURN_NAMES = ("conditioning", "report", "feature_projections")
    FUNCTION = "inspect"
    CATEGORY = categoryFolder
    DESCRIPTION = (
        "Inspects the exact 2048-channel DINO+NAF sparse features emitted by "
        "Trellis2 Texture Stage, then passes conditioning through unchanged. "
        "The first four previews are DINO and the last four are NAF."
    )

    def inspect(self, conditioning, shape_latent):
        report, preview = projected_feature_diagnostics(conditioning, shape_latent)
        logging.info("%s", report.replace("\n", " | "))
        return conditioning, report, preview


NODE_CLASS_MAPPINGS_3D = {
    "CP MeshToTrimesh" : CP_MeshToTrimesh,
    "CP TrimeshWeldVertex": CP_TrimeshWeldVertex,
    "CP STLCheck": CP_STLCheck,
    "CP QuadRemesh": CP_QuadRemesh,
    "CP FixMeshIssues": CP_FixMeshIssues,
    "CP SaveTrimesh": CP_SaveTrimesh,
    "TrimeshToComfyMeshV040": TrimeshToComfyMeshV040,
    "Trellis2ShapeEncoderLoader": Trellis2ShapeEncoderLoader,
    "Pixal3DFourViewConditioningHunyuan": Pixal3DFourViewConditioningHunyuan,
    "Pixal3DAutoAlignView": Pixal3DAutoAlignView,
    "Pixal3DGeometryAlignSettings": Pixal3DGeometryAlignSettings,
    "Pixal3DGeometryAwareViewAlign": Pixal3DGeometryAwareViewAlign,
    "Pixal3DHumanoidAlignSettings": Pixal3DHumanoidAlignSettings,
    "Pixal3DHumanoidPoseReference": Pixal3DHumanoidPoseReference,
    "Pixal3DSAM3DHumanoidPoseProjector": Pixal3DSAM3DHumanoidPoseProjector,
    "Pixal3DHumanoidPoseViewAlign": Pixal3DHumanoidPoseViewAlign,
    "Pixal3DShapeViewMaps": Pixal3DShapeViewMaps,
    "Pixal3DBatchFourViews": Pixal3DBatchFourViews,
    "Pixal3DAutoAlignFourViews": Pixal3DAutoAlignFourViews,
    "HunyuanMeshToPixal3DShapeLatent": HunyuanMeshToPixal3DShapeLatent,
    "Pixal3DRebuildSparseVoxel": Pixal3DRebuildSparseVoxel,
    "RestoreHunyuanMeshTransform": RestoreHunyuanMeshTransform,
    "PredictTrellis2ShapeSubdivides": PredictTrellis2ShapeSubdivides,
    "Pixal3DReprojectionViewSettings": Pixal3DReprojectionViewSettings,
    "Pixal3DImageGuidedVoxelReprojection": Pixal3DImageGuidedVoxelReprojection,
    "Pixal3DImageGuidedVoxelReprojectionView": Pixal3DImageGuidedVoxelReprojectionView,
    "Pixal3DVoxelDiagnostics": Pixal3DVoxelDiagnostics,
    "Pixal3DLatentDiagnostics": Pixal3DLatentDiagnostics,
    "Pixal3DProjectionDiagnostics": Pixal3DProjectionDiagnostics,
    "Pixal3DProjectedFeatureDiagnostics": Pixal3DProjectedFeatureDiagnostics,
}

NODE_DISPLAY_NAME_MAPPINGS_3D = {
    "CP MeshToTrimesh": "Mesh to trimesh (custom)",
    "CP TrimeshWeldVertex": "Trimesh weld vertex",
    "CP STLCheck": "STL Check",
    "CP QuadRemesh": "Quad remesh",
    "CP FixMeshIssues": "Fix mesh issues",
    "CP SaveTrimesh": "Save trimesh custom",
    "TrimeshToComfyMeshV040": "TRIMESH → MESH (ComfyUI 0.40)",
    "Trellis2ShapeEncoderLoader": "Load TRELLIS.2 External Mesh Encoder",
    "Pixal3DFourViewConditioningHunyuan": "Pixal3D 4-View Conditioning (Hunyuan)",
    "Pixal3DAutoAlignView": "Pixal3D Auto Align View",
    "Pixal3DGeometryAlignSettings": "Pixal3D Geometry Align Settings",
    "Pixal3DGeometryAwareViewAlign": "Pixal3D Geometry-Aware View Align",
    "Pixal3DHumanoidAlignSettings": "Pixal3D Humanoid Align Settings",
    "Pixal3DHumanoidPoseReference": "Pixal3D Humanoid Pose Reference",
    "Pixal3DSAM3DHumanoidPoseProjector": "Pixal3D SAM3D Humanoid Pose Projector",
    "Pixal3DHumanoidPoseViewAlign": "Pixal3D Humanoid Pose View Align",
    "Pixal3DShapeViewMaps": "Pixal3D Shape View Maps (Mask / Depth / Canny)",
    "Pixal3DBatchFourViews": "Pixal3D Batch 4 Views (F/L/R/B)",
    "Pixal3DAutoAlignFourViews": "Pixal3D Auto Align 4 Views (Legacy)",
    "HunyuanMeshToPixal3DShapeLatent": "Hunyuan Mesh → Pixal3D Texture Latent",
    "Pixal3DRebuildSparseVoxel": "Pixal3D Rebuild Sparse Voxel (High Resolution)",
    "RestoreHunyuanMeshTransform": "Restore Hunyuan Mesh Transform",
    "PredictTrellis2ShapeSubdivides": "Predict TRELLIS.2 Shape Subdivides",
    "Pixal3DReprojectionViewSettings": "Pixal3D Reprojection View Settings",
    "Pixal3DImageGuidedVoxelReprojection": "Pixal3D Image-Guided Voxel Reprojection",
    "Pixal3DImageGuidedVoxelReprojectionView": "Pixal3D Image-Guided Reprojection View",
    "Pixal3DVoxelDiagnostics": "Pixal3D Voxel Diagnostics",
    "Pixal3DLatentDiagnostics": "Pixal3D Latent Diagnostics",
    "Pixal3DProjectionDiagnostics": "Pixal3D Projection Diagnostics",
    "Pixal3DProjectedFeatureDiagnostics": "Pixal3D Projected Feature Diagnostics",
}
