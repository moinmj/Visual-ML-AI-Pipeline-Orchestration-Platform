import asyncio
import traceback

def run_all():
    print("=== 1. Testing Data Profiler ===")
    from backend.tests.test_profiler import test_profiler_basic_stats, test_profiler_empty_dataframe
    test_profiler_basic_stats()
    print("  [PASS] test_profiler_basic_stats")
    test_profiler_empty_dataframe()
    print("  [PASS] test_profiler_empty_dataframe")

    print("\n=== 2. Testing DAG Graph & Validation ===")
    from backend.tests.test_dag import test_dag_cycle_detection, test_dag_valid_topological_sort
    test_dag_cycle_detection()
    print("  [PASS] test_dag_cycle_detection")
    test_dag_valid_topological_sort()
    print("  [PASS] test_dag_valid_topological_sort")

    print("\n=== 3. Testing Recipes & End-to-End Pipeline ===")
    from backend.tests.test_recipes import test_recipe_registry_loading, test_preprocessing_and_training_pipeline
    test_recipe_registry_loading()
    print("  [PASS] test_recipe_registry_loading")
    test_preprocessing_and_training_pipeline()
    print("  [PASS] test_preprocessing_and_training_pipeline")

    print("\n=== 4. Testing FastAPI Endpoints ===")
    from backend.tests.test_api import test_health_and_recipes_api, test_workflow_validation_api
    asyncio.run(test_health_and_recipes_api())
    print("  [PASS] test_health_and_recipes_api")
    asyncio.run(test_workflow_validation_api())
    print("  [PASS] test_workflow_validation_api")

    print("\n==========================================")
    print(" ALL 8 AUTOMATED TESTS PASSED SUCCESSFULLY! ")
    print("==========================================")

if __name__ == "__main__":
    run_all()
