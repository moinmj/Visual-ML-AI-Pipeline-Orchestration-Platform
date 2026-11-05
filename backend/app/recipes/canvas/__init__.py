"""
Canvas Tools Recipe Package.

Provides 10 enterprise canvas utilities, visual documentation, layout organizers,
interactive chart visualizers, pipeline version diffing/snapshotting, secrets management,
and parameter sweep scheduling.
"""

from backend.app.recipes.canvas.canvas_note import CanvasNoteRecipe
from backend.app.recipes.canvas.canvas_group import CanvasGroupRecipe
from backend.app.recipes.canvas.canvas_comments import CanvasCommentsRecipe
from backend.app.recipes.canvas.chart_visualization import ChartVisualizationRecipe
from backend.app.recipes.canvas.version_snapshot import VersionSnapshotRecipe
from backend.app.recipes.canvas.template_macro_library import TemplateMacroLibraryRecipe
from backend.app.recipes.canvas.pipeline_diff_viewer import PipelineDiffViewerRecipe
from backend.app.recipes.canvas.readonly_pipeline_snapshot import ReadonlyPipelineSnapshotRecipe
from backend.app.recipes.canvas.secrets_manager_block import SecretsManagerBlockRecipe
from backend.app.recipes.canvas.parameter_sweep_runner import ParameterSweepRunnerRecipe

__all__ = [
    "CanvasNoteRecipe",
    "CanvasGroupRecipe",
    "CanvasCommentsRecipe",
    "ChartVisualizationRecipe",
    "VersionSnapshotRecipe",
    "TemplateMacroLibraryRecipe",
    "PipelineDiffViewerRecipe",
    "ReadonlyPipelineSnapshotRecipe",
    "SecretsManagerBlockRecipe",
    "ParameterSweepRunnerRecipe",
]
