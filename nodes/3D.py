import torch
import pynanoinstantmeshes as PyNIM
import trimesh as trimesh_lib
from trimesh.grouping import merge_vertices, group_rows
from trimesh.constants import tol
import numpy as np
from scipy.spatial import KDTree

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
    CATEGORY = "Hunyuan3DWrapper"
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

NODE_CLASS_MAPPINGS_3D = {
    "CP MeshToTrimesh" : CP_MeshToTrimesh,
    "CP TrimeshWeldVertex": CP_TrimeshWeldVertex,
    "CP STLCheck": CP_STLCheck,
    "CP QuadRemesh": CP_QuadRemesh,
    "CP FixMeshIssues": CP_FixMeshIssues,
    "CP SaveTrimesh": CP_SaveTrimesh,
}

NODE_DISPLAY_NAME_MAPPINGS_3D = {
    "CP MeshToTrimesh": "Mesh to trimesh (custom)",
    "CP TrimeshWeldVertex": "Trimesh weld vertex",
    "CP STLCheck": "STL Check",
    "CP QuadRemesh": "Quad remesh",
    "CP FixMeshIssues": "Fix mesh issues",
    "CP SaveTrimesh": "Save trimesh custom",
}
