"""
Unit and Integration Tests for Streaming & Event Ingestion Recipes.
Covers:
- Kafka Consumer (kafka_consumer)
- RabbitMQ Message Consumer (rabbitmq_consumer)
- MQTT / IoT Sensor Stream (mqtt_iot_stream)
- WebSocket Streaming Listener (websocket_listener)
- Event Bus Publisher (event_bus_publisher)
"""

import json
import pytest
import pandas as pd

from backend.app.recipes.base.registry import recipe_registry
from backend.app.recipes.ingestion.kafka_consumer import KafkaConsumerRecipe
from backend.app.recipes.ingestion.rabbitmq_consumer import RabbitMQConsumerRecipe
from backend.app.recipes.ingestion.mqtt_iot_stream import MQTTStreamRecipe
from backend.app.recipes.ingestion.websocket_listener import WebSocketListenerRecipe
from backend.app.recipes.ingestion.event_bus_publisher import EventBusPublisherRecipe
from backend.app.engine.dag.graph import WorkflowGraph, WorkflowNode, WorkflowEdge
from backend.app.engine.execution.executor import DAGExecutor


def test_streaming_recipes_registered_and_metadata():
    """Verify all 5 streaming recipes are registered, enabled, and mapped to Streaming & Change Data."""
    recipes = [
        ("kafka_consumer", "Kafka Consumer", "ingestion"),
        ("rabbitmq_consumer", "RabbitMQ Message Consumer", "ingestion"),
        ("mqtt_iot_stream", "MQTT / IoT Sensor Stream", "ingestion"),
        ("websocket_listener", "WebSocket Streaming Listener", "ingestion"),
        ("event_bus_publisher", "Event Bus Publisher (Kafka/RabbitMQ/SNS)", "export"),
    ]

    for rid, expected_name, expected_cat in recipes:
        assert recipe_registry.has(rid), f"Recipe '{rid}' must be registered and enabled."
        meta = recipe_registry.get_metadata(rid)
        assert meta.enabled is True
        assert meta.name == expected_name
        assert meta.group == "Ingestion & Orchestration"
        assert meta.subgroup == "Streaming & Change Data"
        assert meta.category == expected_cat
        assert "properties" in meta.parameters_schema


def test_streaming_recipe_aliases():
    """Verify streaming aliases resolve seamlessly to the primary recipe IDs."""
    aliases = [
        ("kafka", "kafka_consumer"),
        ("kafka_stream", "kafka_consumer"),
        ("rabbitmq", "rabbitmq_consumer"),
        ("rabbitmq_queue", "rabbitmq_consumer"),
        ("amqp", "rabbitmq_consumer"),
        ("mqtt", "mqtt_iot_stream"),
        ("mqtt_stream", "mqtt_iot_stream"),
        ("iot_stream", "mqtt_iot_stream"),
        ("websocket", "websocket_listener"),
        ("ws_stream", "websocket_listener"),
        ("event_bus", "event_bus_publisher"),
        ("event_publisher", "event_bus_publisher"),
    ]
    for alias, target_id in aliases:
        assert recipe_registry.has(alias)
        recipe = recipe_registry.get(alias)
        assert recipe.recipe_id == target_id


def test_kafka_consumer_validation():
    """Verify Kafka consumer parameter validation."""
    r = KafkaConsumerRecipe()

    # Missing bootstrap servers / topic / group_id
    errs = r.validate_config({"bootstrap_servers": "", "topic": "", "group_id": ""})
    assert any("Bootstrap servers" in e for e in errs)
    assert any("Kafka topic" in e for e in errs)
    assert any("Consumer group" in e for e in errs)

    # SASL requires username
    errs = r.validate_config({
        "bootstrap_servers": "localhost:9092",
        "topic": "events",
        "group_id": "grp1",
        "security_protocol": "SASL_PLAINTEXT",
        "sasl_username": ""
    })
    assert any("SASL Username is required" in e for e in errs)

    # Invalid max_messages
    errs = r.validate_config({
        "bootstrap_servers": "localhost:9092",
        "topic": "events",
        "group_id": "grp1",
        "max_messages": -5
    })
    assert any("greater than 0" in e for e in errs)


def test_kafka_consumer_execution_simulation():
    """Verify Kafka consumer simulation returns micro-batch records."""
    r = KafkaConsumerRecipe()
    cfg = {
        "bootstrap_servers": "localhost:9092",
        "topic": "telemetry-events",
        "group_id": "test-grp",
        "max_messages": 25,
        "flatten_json": True,
        "simulation_mode": True
    }
    res = r.execute({}, cfg)
    df = res["dataframe"]
    summary = res["output_summary"]
    metrics = res["metrics"]

    assert len(df) == 25
    assert "kafka_topic" in df.columns
    assert "kafka_partition" in df.columns
    assert "kafka_offset" in df.columns
    assert "event_type" in df.columns
    assert "transaction_sar" in df.columns
    assert summary["status"] == "SIMULATED"
    assert summary["messages_polled"] == 25
    assert metrics["delivery_status"] == "SIMULATED"


def test_rabbitmq_consumer_validation_and_execution():
    """Verify RabbitMQ consumer validation and simulation draining."""
    r = RabbitMQConsumerRecipe()

    # Missing queue name
    errs = r.validate_config({"queue_name": "", "connection_mode": "host_port", "host": "localhost"})
    assert any("Queue name is required" in e for e in errs)

    cfg = {
        "connection_mode": "host_port",
        "host": "localhost",
        "queue_name": "orders-incoming",
        "max_messages": 30,
        "simulation_mode": True
    }
    res = r.execute({}, cfg)
    df = res["dataframe"]
    summary = res["output_summary"]

    assert len(df) == 30
    assert "rabbitmq_queue" in df.columns
    assert "delivery_tag" in df.columns
    assert "action" in df.columns
    assert summary["queue_name"] == "orders-incoming"
    assert summary["messages_drained"] == 30
    assert summary["status"] == "SIMULATED"


def test_mqtt_iot_stream_validation_and_execution():
    """Verify MQTT stream validation and IoT telemetry simulation."""
    r = MQTTStreamRecipe()

    # Missing broker host / topic
    errs = r.validate_config({"broker_host": "", "topic": ""})
    assert any("MQTT broker host is required" in e for e in errs)
    assert any("MQTT topic is required" in e for e in errs)

    # Invalid port
    errs = r.validate_config({"broker_host": "b", "topic": "t", "port": 70000})
    assert any("Port must be between" in e for e in errs)

    cfg = {
        "broker_host": "broker.emqx.io",
        "topic": "factory/sensors/#",
        "qos": 1,
        "max_messages": 20,
        "simulation_mode": True
    }
    res = r.execute({}, cfg)
    df = res["dataframe"]
    summary = res["output_summary"]

    assert len(df) == 20
    assert "mqtt_topic" in df.columns
    assert "temperature_c" in df.columns
    assert "humidity_pct" in df.columns
    assert "pressure_hpa" in df.columns
    assert summary["status"] == "SIMULATED"
    assert summary["messages_ingested"] == 20


def test_websocket_listener_validation_and_execution():
    """Verify WebSocket listener URL validation and market trade simulation."""
    r = WebSocketListenerRecipe()

    # URL must start with ws:// or wss://
    errs = r.validate_config({"url": "http://stream.com"})
    assert any("must start with 'ws://' or 'wss://'" in e for e in errs)

    cfg = {
        "url": "wss://stream.binance.com:9443/ws/btcusdt@trade",
        "max_messages": 15,
        "timeout_seconds": 2,
        "simulation_mode": True
    }
    res = r.execute({}, cfg)
    df = res["dataframe"]
    summary = res["output_summary"]

    assert len(df) == 15
    assert "ws_endpoint" in df.columns
    assert "symbol" in df.columns
    assert "price" in df.columns
    assert "quantity" in df.columns
    assert summary["status"] == "SIMULATED"
    assert summary["messages_captured"] == 15


def test_event_bus_publisher_validation_and_execution():
    """Verify Event Bus Publisher validation and emission."""
    r = EventBusPublisherRecipe()

    # Missing destination
    errs = r.validate_config({"bus_type": "kafka", "destination": ""})
    assert any("Destination topic/queue/ARN is required" in e for e in errs)

    # Unsupported bus
    errs = r.validate_config({"bus_type": "invalid_bus", "destination": "topic"})
    assert any("Unsupported bus_type" in e for e in errs)

    input_df = pd.DataFrame([
        {"user_id": 101, "event": "LOGIN", "status": "OK"},
        {"user_id": 102, "event": "LOGOUT", "status": "OK"},
        {"user_id": 103, "event": "PURCHASE", "status": "PAID"},
    ])
    cfg = {
        "bus_type": "kafka",
        "destination": "user-audit-stream",
        "publish_mode": "row_by_row",
        "simulation_mode": True
    }
    res = r.execute({"input": input_df}, cfg)
    df_out = res["dataframe"]
    summary = res["output_summary"]

    assert len(df_out) == 3
    assert list(df_out.columns) == list(input_df.columns)  # Passthrough preserved
    assert summary["bus_type"] == "kafka"
    assert summary["messages_published"] == 3
    assert summary["status"] == "SIMULATED"


def test_to_code_for_all_streaming_recipes():
    """Verify code generation for all 5 streaming recipes."""
    recipes = [
        (KafkaConsumerRecipe(), {"bootstrap_servers": "localhost:9092", "topic": "events"}),
        (RabbitMQConsumerRecipe(), {"host": "localhost", "queue_name": "orders"}),
        (MQTTStreamRecipe(), {"broker_host": "broker.hivemq.com", "topic": "sensors/#"}),
        (WebSocketListenerRecipe(), {"url": "wss://feed.crypto.com/ws"}),
        (EventBusPublisherRecipe(), {"bus_type": "kafka", "destination": "out-topic"}),
    ]
    for r, cfg in recipes:
        code = r.to_code(cfg)
        assert isinstance(code, str)
        assert len(code) > 20
        assert "print" in code


def test_kafka_dag_workflow():
    """Verify Kafka Consumer in a multi-node pipeline with DAGExecutor."""
    nodes = [
        WorkflowNode(
            id="node_kafka",
            recipe_id="kafka_consumer",
            config={
                "bootstrap_servers": "localhost:9092",
                "topic": "telemetry-events",
                "group_id": "dag-test-grp",
                "max_messages": 30,
                "simulation_mode": True
            }
        ),
        WorkflowNode(
            id="node_scaler",
            recipe_id="feature_scaler",
            config={
                "method": "standard",
                "columns": ["transaction_sar", "latency_ms"]
            }
        )
    ]
    edges = [
        WorkflowEdge(id="e_k_s", source="node_kafka", target="node_scaler")
    ]
    graph = WorkflowGraph(id="wf_kafka_test", name="Kafka Stream DAG Test", nodes=nodes, edges=edges)

    exec_result = DAGExecutor.execute_workflow(
        execution_id="test_kafka_exec_100",
        workflow=graph,
        include_node_outputs=True
    )
    assert exec_result.status == "SUCCESS"
    assert len(exec_result.node_results) == 2
    res_k = next(r for r in exec_result.node_results if r.node_id == "node_kafka")
    assert res_k.status == "SUCCESS"
    summary_k = res_k.output_summary.get("output_summary") or res_k.output_summary
    assert summary_k.get("title") == "Kafka Consumer Summary"


def test_mqtt_to_event_bus_dag_workflow():
    """Verify MQTT Stream into Event Bus Publisher pipeline with DAGExecutor."""
    nodes = [
        WorkflowNode(
            id="node_mqtt",
            recipe_id="mqtt_iot_stream",
            config={
                "broker_host": "broker.emqx.io",
                "topic": "factory/telemetry/#",
                "max_messages": 20,
                "simulation_mode": True
            }
        ),
        WorkflowNode(
            id="node_imputer",
            recipe_id="missing_values",
            config={
                "strategy": "mean",
                "columns": ["temperature_c", "humidity_pct"]
            }
        ),
        WorkflowNode(
            id="node_pub",
            recipe_id="event_bus_publisher",
            config={
                "bus_type": "rabbitmq",
                "destination": "sanitized-iot-queue",
                "publish_mode": "row_by_row",
                "simulation_mode": True
            }
        )
    ]
    edges = [
        WorkflowEdge(id="e1", source="node_mqtt", target="node_imputer"),
        WorkflowEdge(id="e2", source="node_imputer", target="node_pub")
    ]
    graph = WorkflowGraph(id="wf_mqtt_bus_test", name="MQTT to Bus Test", nodes=nodes, edges=edges)

    exec_result = DAGExecutor.execute_workflow(
        execution_id="test_mqtt_bus_200",
        workflow=graph,
        include_node_outputs=True
    )
    assert exec_result.status == "SUCCESS"
    assert len(exec_result.node_results) == 3
    res_pub = next(r for r in exec_result.node_results if r.node_id == "node_pub")
    assert res_pub.status == "SUCCESS"
    summary_pub = res_pub.output_summary.get("output_summary") or res_pub.output_summary
    assert summary_pub.get("title") == "Event Bus Publisher Summary"
    assert summary_pub.get("messages_published") == 20
