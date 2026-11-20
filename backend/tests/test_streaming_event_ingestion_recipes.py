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
    assert summary["title"] == "RabbitMQ Queue Ingestion Metrics"
    assert summary["tag"] == "RABBITMQ"
    assert summary["queue_name"] == "orders-incoming"
    assert summary["messages_drained"] == 30
    assert summary["status"] == "SIMULATED"
    assert isinstance(summary["messages_drained"], int)
    assert isinstance(summary["columns_count"], int)
    assert isinstance(summary["duration_ms"], int)


def test_rabbitmq_consumer_live_connection_exception_handling():
    """Verify live connection error raises ValidationException with human-readable broker unreachable guidance."""
    from backend.app.core.exceptions import ValidationException
    r = RabbitMQConsumerRecipe()

    cfg = {
        "connection_mode": "host_port",
        "host": "127.0.0.1",
        "port": 5679,  # Unreachable port
        "queue_name": "orders-test",
        "simulation_mode": False
    }

    with pytest.raises(ValidationException) as exc_info:
        r.execute({}, cfg)

    err_msg = str(exc_info.value)
    assert "Failed connecting to RabbitMQ broker" in err_msg or "Connection refused" in err_msg
    assert "5679" in err_msg


def test_rabbitmq_consumer_flatten_json_toggle():
    """Verify that when flatten_json is False, nested RabbitMQ payload remains inside a single 'payload' column."""
    r = RabbitMQConsumerRecipe()

    # 1. flatten_json = False (Unchecked)
    cfg_unflat = {
        "queue_name": "tasks-queue",
        "max_messages": 10,
        "flatten_json": False,
        "simulation_mode": True
    }
    res_unflat = r.execute({}, cfg_unflat)
    df_unflat = res_unflat["dataframe"]
    assert "payload" in df_unflat.columns
    assert "action" not in df_unflat.columns
    assert "priority" not in df_unflat.columns

    # Verify JSON structure inside payload column
    parsed = json.loads(df_unflat["payload"].iloc[0])
    assert "action" in parsed
    assert "priority" in parsed

    # 2. flatten_json = True (Checked)
    cfg_flat = {
        "queue_name": "tasks-queue",
        "max_messages": 10,
        "flatten_json": True,
        "simulation_mode": True
    }
    res_flat = r.execute({}, cfg_flat)
    df_flat = res_flat["dataframe"]
    assert "action" in df_flat.columns
    assert "priority" in df_flat.columns


def test_rabbitmq_consumer_source_node_inputs_empty():
    """Verify RabbitMQ consumer has empty input_types and inputs (is a standalone DAG root)."""
    r = RabbitMQConsumerRecipe()
    assert r.input_types == []
    assert r.get_inputs() == []
    meta = r.to_metadata()
    assert meta.input_types == []
    assert meta.inputs == []


def test_source_node_error_diagnostics_no_handle_suggestion():
    """Verify diagnose_execution_error on source ingestion nodes never suggests checking incoming connection handles."""
    from backend.app.engine.execution.executor import diagnose_execution_error
    r = RabbitMQConsumerRecipe()

    diag = diagnose_execution_error(
        recipe=r,
        error=Exception("Connection refused to 127.0.0.1:5672"),
        inputs={},
        config={"queue_name": "test"},
        context={}
    )
    assert "Check incoming connection handles" not in diag["suggestion"]
    assert "Broker Unreachable" in diag["title"]
    assert "message broker" in diag["suggestion"]


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
    assert summary["title"] == "MQTT / IoT Sensor Stream Metrics"
    assert summary["tag"] == "MQTT"
    assert summary["status"] == "SIMULATED"
    assert summary["messages_ingested"] == 20
    assert isinstance(summary["messages_ingested"], int)
    assert isinstance(summary["columns_count"], int)
    assert isinstance(summary["duration_ms"], int)


def test_mqtt_iot_stream_flatten_json_toggle():
    """Verify that unchecking 'Flatten JSON Payloads' preserves the sensor payload as a single 'payload' raw JSON column."""
    r = MQTTStreamRecipe()

    # 1. flatten_json = False (Unchecked)
    cfg_unflat = {
        "broker_host": "broker.emqx.io",
        "topic": "factory/sensors/#",
        "max_messages": 10,
        "flatten_json": False,
        "simulation_mode": True
    }
    res_unflat = r.execute({}, cfg_unflat)
    df_unflat = res_unflat["dataframe"]
    assert "payload" in df_unflat.columns
    assert "temperature_c" not in df_unflat.columns
    assert "humidity_pct" not in df_unflat.columns
    assert "pressure_hpa" not in df_unflat.columns

    # Verify payload column contains valid JSON string
    first_payload = json.loads(df_unflat["payload"].iloc[0])
    assert "temperature_c" in first_payload
    assert "humidity_pct" in first_payload
    assert "device_id" in first_payload

    # 2. flatten_json = True (Checked)
    cfg_flat = {
        "broker_host": "broker.emqx.io",
        "topic": "factory/sensors/#",
        "max_messages": 10,
        "flatten_json": True,
        "simulation_mode": True
    }
    res_flat = r.execute({}, cfg_flat)
    df_flat = res_flat["dataframe"]
    assert "temperature_c" in df_flat.columns
    assert "humidity_pct" in df_flat.columns
    assert "pressure_hpa" in df_flat.columns


def test_mqtt_iot_stream_no_stray_delimiter_in_schema():
    """Verify MQTT schema does not contain stray delimiter or CSV properties."""
    r = MQTTStreamRecipe()
    schema = r.get_schema()
    props = schema["properties"]
    assert "delimiter" not in props
    assert "csv_delimiter" not in props
    assert "quote_char" not in props


def test_mqtt_iot_stream_live_connection_unreachable_exception():
    """Verify live connection to unreachable broker raises ValidationException with human-readable guidance."""
    from backend.app.core.exceptions import ValidationException
    r = MQTTStreamRecipe()

    cfg = {
        "broker_host": "127.0.0.1",
        "port": 1889,  # Non-existent port
        "topic": "sensors/test",
        "timeout_seconds": 1,
        "simulation_mode": False
    }

    with pytest.raises(ValidationException) as exc_info:
        r.execute({}, cfg)

    err = str(exc_info.value)
    assert "Failed connecting to MQTT broker" in err or "Connection refused" in err
    assert "1889" in err


def test_mqtt_iot_stream_live_message_conversion(monkeypatch):
    """Verify live incoming messages convert into real table records with delivery status 'LIVE'."""
    r = MQTTStreamRecipe()

    class MockMQTTMessage:
        def __init__(self, topic, payload, qos=0):
            self.topic = topic
            self.payload = payload.encode("utf-8") if isinstance(payload, str) else payload
            self.qos = qos

    class MockMQTTClient:
        def __init__(self, *args, **kwargs):
            self.on_connect = None
            self.on_message = None

        def username_pw_set(self, u, p):
            pass

        def connect(self, host, port, keepalive=30):
            pass

        def subscribe(self, topic, qos=0):
            pass

        def loop_start(self):
            # Simulate broker connection ACK and arrival of live messages
            if self.on_connect:
                self.on_connect(self, None, {}, 0)
            if self.on_message:
                self.on_message(self, None, MockMQTTMessage("factory/sensors/temp", '{"device": "SENSOR_01", "temp": 24.8, "status": "OK"}'))
                self.on_message(self, None, MockMQTTMessage("factory/sensors/vibe", '{"device": "SENSOR_02", "vibe": 0.04, "status": "OK"}'))

        def loop_stop(self):
            pass

        def disconnect(self):
            pass

    import paho.mqtt.client as paho_client
    monkeypatch.setattr(paho_client, "Client", MockMQTTClient)

    cfg = {
        "broker_host": "mock.broker.net",
        "port": 1883,
        "topic": "factory/sensors/#",
        "timeout_seconds": 1,
        "flatten_json": True,
        "simulation_mode": False
    }

    res = r.execute({}, cfg)
    assert res is not None
    assert "dataframe" in res
    df = res["dataframe"]
    summary = res["output_summary"]
    metrics = res["metrics"]

    assert len(df) == 2
    assert summary["status"] == "LIVE"
    assert metrics["delivery_status"] == "LIVE"
    assert "device" in df.columns
    assert df["device"].tolist() == ["SENSOR_01", "SENSOR_02"]
    assert "Ingested 2 live MQTT messages" in summary["message"]


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
    assert summary_k.get("title") == "Kafka Streaming Ingestion Metrics"
    assert summary_k.get("tag") == "KAFKA"


def test_kafka_consumer_resilient_offline_fallback():
    """Verify Kafka consumer gracefully switches to simulated benchmark records when broker is offline/unreachable."""
    r = KafkaConsumerRecipe()
    cfg = {
        "bootstrap_servers": "127.0.0.1:9099",  # Non-existent broker
        "topic": "telemetry-events",
        "group_id": "test-fallback-grp",
        "poll_timeout_ms": 500,
        "max_messages": 15,
        "flatten_json": True,
        "simulation_mode": False  # Live mode attempted, but must gracefully fall back without crashing
    }
    res = r.execute({}, cfg)
    assert res is not None
    assert "dataframe" in res
    df = res["dataframe"]
    assert len(df) == 15
    summary = res["output_summary"]
    metrics = res["metrics"]
    assert summary["status"] == "SIMULATED"
    assert summary["title"] == "Kafka Streaming Ingestion Metrics"
    assert summary["tag"] == "KAFKA"
    assert metrics["delivery_status"] == "SIMULATED"
    assert "messages_polled" in summary
    assert isinstance(summary["messages_polled"], int)
    assert summary["messages_polled"] == 15


def test_kafka_consumer_flatten_json_toggle():
    """Verify that when flatten_json is False, nested event payload remains serialized inside a single 'payload' column."""
    r = KafkaConsumerRecipe()
    
    # 1. flatten_json = False
    cfg_unflattened = {
        "topic": "telemetry-events",
        "max_messages": 10,
        "flatten_json": False,
        "simulation_mode": True
    }
    res_unflat = r.execute({}, cfg_unflattened)
    df_unflat = res_unflat["dataframe"]
    assert "payload" in df_unflat.columns
    assert "event_type" not in df_unflat.columns
    assert "transaction_sar" not in df_unflat.columns
    
    # Verify payload column contains valid JSON strings
    first_payload = json.loads(df_unflat["payload"].iloc[0])
    assert "event_type" in first_payload
    assert "transaction_sar" in first_payload

    # 2. flatten_json = True
    cfg_flattened = {
        "topic": "telemetry-events",
        "max_messages": 10,
        "flatten_json": True,
        "simulation_mode": True
    }
    res_flat = r.execute({}, cfg_flattened)
    df_flat = res_flat["dataframe"]
    assert "event_type" in df_flat.columns
    assert "transaction_sar" in df_flat.columns


def test_kafka_consumer_auto_offset_reset_simulation():
    """Verify simulation starts offsets at 0 for 'earliest' and at high number (500000) for 'latest'."""
    r = KafkaConsumerRecipe()

    # earliest -> starts at 0
    res_earliest = r.execute({}, {"auto_offset_reset": "earliest", "simulation_mode": True, "max_messages": 5})
    df_earliest = res_earliest["dataframe"]
    assert df_earliest["kafka_offset"].iloc[0] == 0
    assert df_earliest["kafka_offset"].iloc[4] == 4

    # latest -> starts at 500000
    res_latest = r.execute({}, {"auto_offset_reset": "latest", "simulation_mode": True, "max_messages": 5})
    df_latest = res_latest["dataframe"]
    assert df_latest["kafka_offset"].iloc[0] == 500000
    assert df_latest["kafka_offset"].iloc[4] == 500004


def test_kafka_consumer_source_node_inputs_empty():
    """Verify Kafka consumer has empty input_types and inputs (is a DAG root source)."""
    r = KafkaConsumerRecipe()
    assert r.input_types == []
    assert r.get_inputs() == []
    meta = r.to_metadata()
    assert meta.input_types == []
    assert meta.inputs == []


def test_kafka_consumer_sasl_schema_visible_when():
    """Verify SASL Mechanism, Username, and Password render for both SASL_PLAINTEXT and SASL_SSL."""
    r = KafkaConsumerRecipe()
    schema = r.get_schema()
    props = schema["properties"]
    
    for f in ["sasl_mechanism", "sasl_username", "sasl_password"]:
        vw = props[f]["visible_when"]
        assert vw["field"] == "security_protocol"
        assert "SASL_SSL" in vw.get("one_of", []) or vw.get("starts_with") == "SASL"


def test_kafka_consumer_metrics_whole_integer_formatting():
    """Verify metrics and summary counts are clean whole integers (no floating point decimals)."""
    r = KafkaConsumerRecipe()
    res = r.execute({}, {"simulation_mode": True, "max_messages": 10})
    summary = res["output_summary"]
    metrics = res["metrics"]

    assert summary["title"] == "Kafka Streaming Ingestion Metrics"
    assert summary["tag"] == "KAFKA"
    assert metrics["title"] == "Kafka Streaming Ingestion Metrics"
    assert metrics["tag"] == "KAFKA"

    for count_key in ["messages_polled", "columns_count", "duration_ms"]:
        assert isinstance(summary[count_key], int)
        assert isinstance(metrics[count_key], int)


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
