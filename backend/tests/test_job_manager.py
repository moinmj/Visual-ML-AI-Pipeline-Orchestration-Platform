import pytest
import time
import pandas as pd
from backend.app.engine.dag.graph import WorkflowGraph, WorkflowNode, WorkflowEdge
from backend.app.engine.execution.job_manager import PipelineJobManager


def test_async_job_manager_execution():
    manager = PipelineJobManager()
    
    # Simple 2-node DAG: CSV Loader -> Feature Scaler
    nodes = [
        WorkflowNode(id="n1", recipe_id="csv_loader", config={}),
        WorkflowNode(id="n2", recipe_id="feature_scaler", config={"method": "standard"})
    ]
    edges = [
        WorkflowEdge(source="n1", target="n2")
    ]
    graph = WorkflowGraph(nodes=nodes, edges=edges)
    
    df = pd.DataFrame({"feat_a": [1.0, 2.0, 3.0, 4.0, 5.0], "feat_b": [10.0, 20.0, 30.0, 40.0, 50.0]})
    
    job_id = manager.submit_job(workflow=graph, initial_df=df)
    assert job_id.startswith("job_")
    
    # Wait for completion
    for _ in range(30):
        job = manager.get_job(job_id)
        if job and job["status"] in ["SUCCESS", "FAILED"]:
            break
        time.sleep(0.1)
        
    final_job = manager.get_job(job_id)
    assert final_job is not None
    assert final_job["status"] == "SUCCESS"
    assert final_job["duration_ms"] > 0
    assert final_job["result"] is not None
    assert "step_snapshots" in final_job["result"].model_dump()
