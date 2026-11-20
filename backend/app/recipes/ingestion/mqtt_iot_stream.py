"""
MQTT / IoT Sensor Stream Recipe (Ingestion & Orchestration -> Streaming & Change Data).

Subscribes to MQTT topics to ingest live IoT sensor telemetry readings into DataFrame micro-batches.
Supports broker host/port, topic wildcards (#, +), QoS levels (0, 1, 2), client credentials,
JSON sensor payload deserialization with auto-flattening, benchmark simulation fallback, and code generation.
"""

import os
import json
import time
import uuid
import threading
from typing import Dict, Any, List, Optional, Tuple

try:
    import pandas as pd
except ImportError:
    pd = None

from backend.app.recipes.base.recipe import BaseRecipe, RecipePort
from backend.app.core.exceptions import ValidationException
from backend.app.core.logging import logger
from backend.app.core.config import settings


class MQTTStreamRecipe(BaseRecipe):
    recipe_id = "mqtt_iot_stream"
    name = "MQTT / IoT Sensor Stream"
    version = "1.0.0"
    category = "ingestion"
    group = "Ingestion & Orchestration"
    subgroup = "Streaming & Change Data"
    description = "Subscribes to MQTT topics to ingest real-time IoT sensor readings into micro-batches of tabular rows."
    input_types: List[str] = []
    output_types = ["dataframe"]

    inputs: List[RecipePort] = []

    outputs = [
        RecipePort(
            id="output",
            label="MQTT Telemetry Dataset",
            type="dataframe",
            description="DataFrame containing collected MQTT IoT sensor messages"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "sections": [
                {
                    "id": "connection",
                    "title": "MQTT Broker & Client",
                    "description": "Configure broker hostname, port, client ID, and optional credentials",
                },
                {
                    "id": "subscription",
                    "title": "Topic Subscription & QoS",
                    "description": "Target topic with wildcard support and Quality of Service level",
                },
                {
                    "id": "runtime_limits",
                    "title": "Batching & Sampling",
                    "description": "Batch limits, sampling timeout window, JSON flattening, and simulation mode",
                },
            ],
            "properties": {
                "broker_host": {
                    "type": "string",
                    "title": "MQTT Broker Host",
                    "input_type": "text",
                    "section": "connection",
                    "default": "broker.emqx.io",
                    "description": "MQTT broker hostname or IP address (e.g. 'broker.hivemq.com', 'localhost').",
                },
                "port": {
                    "type": "integer",
                    "title": "Port",
                    "input_type": "number",
                    "section": "connection",
                    "default": 1883,
                    "description": "MQTT port (default 1883 for TCP, 8883 for TLS/SSL).",
                },
                "client_id": {
                    "type": "string",
                    "title": "Client ID",
                    "input_type": "text",
                    "section": "connection",
                    "default": "ml_pipeline_iot_subscriber",
                    "description": "Unique client identifier for the MQTT connection.",
                },
                "username": {
                    "type": "string",
                    "title": "Username (Optional)",
                    "input_type": "text",
                    "section": "connection",
                    "default": "",
                    "description": "Username for MQTT broker authentication.",
                },
                "password": {
                    "type": "string",
                    "title": "Password (Optional)",
                    "input_type": "password",
                    "section": "connection",
                    "default": "",
                    "description": "Password for MQTT broker authentication.",
                },
                "credential_id": {
                    "type": "string",
                    "title": "Connected MQTT Credential",
                    "input_type": "credential_dropdown",
                    "provider": "mqtt",
                    "section": "connection",
                    "description": "Select stored MQTT broker credentials from platform settings.",
                },
                "topic": {
                    "type": "string",
                    "title": "Topic / Wildcard Filter",
                    "input_type": "text",
                    "section": "subscription",
                    "default": "factory/sensors/telemetry/#",
                    "description": "MQTT topic to subscribe to. Wildcards like '#' (multi-level) and '+' (single-level) are supported.",
                },
                "qos": {
                    "type": "integer",
                    "title": "QoS Level",
                    "input_type": "dropdown",
                    "section": "subscription",
                    "enum": [0, 1, 2],
                    "default": 0,
                    "description": "Quality of Service level: 0 (at most once), 1 (at least once), 2 (exactly once).",
                },
                "max_messages": {
                    "type": "integer",
                    "title": "Max Messages per Batch",
                    "input_type": "number",
                    "section": "runtime_limits",
                    "default": 100,
                    "description": "Maximum number of messages to collect before completing the micro-batch (1 to 20,000).",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "title": "Sampling Window (Seconds)",
                    "input_type": "number",
                    "section": "runtime_limits",
                    "default": 3,
                    "description": "Time window in seconds to wait for incoming MQTT messages.",
                },
                "flatten_json": {
                    "type": "boolean",
                    "title": "Flatten JSON Payloads",
                    "input_type": "checkbox",
                    "section": "runtime_limits",
                    "default": True,
                    "description": "Automatically unpack nested JSON object fields into top-level columns.",
                },
                "simulation_mode": {
                    "type": "boolean",
                    "title": "Simulation / Demo Mode",
                    "input_type": "checkbox",
                    "section": "runtime_limits",
                    "default": False,
                    "description": "When enabled or when MQTT broker is offline, generates realistic industrial IoT telemetry records.",
                },
            },
            "required": ["broker_host", "topic"],
        }

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        if not config.get("simulation_mode"):
            if not str(config.get("broker_host", "")).strip():
                errors.append("MQTT broker host is required.")
            if not str(config.get("topic", "")).strip():
                errors.append("MQTT topic is required.")

        port = config.get("port")
        if port is not None:
            try:
                p_val = int(port)
                if p_val <= 0 or p_val > 65535:
                    errors.append("Port must be between 1 and 65535.")
            except (ValueError, TypeError):
                errors.append("Port must be an integer.")

        max_msgs = config.get("max_messages")
        if max_msgs is not None:
            try:
                m_val = int(max_msgs)
                if m_val <= 0:
                    errors.append("Max messages must be greater than 0.")
            except (ValueError, TypeError):
                errors.append("Max messages must be an integer.")

        return errors

    def _generate_benchmark_data(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        limit = max(1, min(int(config.get("max_messages", 100)), 5000))
        topic = str(config.get("topic", "factory/sensors/telemetry/#"))
        qos = int(config.get("qos", 0))
        flatten_json = bool(config.get("flatten_json", True))

        sensors = ["TEMP_HUMIDITY", "VIBRATION_AXIS", "PRESSURE_BAR", "FLOW_METER", "CURRENT_AMPS"]
        models = ["SENS-X100", "BOSCH-BME680", "TI-TMP117", "OMRON-D6F", "STM-LSM6D"]

        records: List[Dict[str, Any]] = []
        for i in range(limit):
            dev_id = f"EDGE_NODE_{10 + (i % 20):03d}"
            nested_payload = {
                "device_id": dev_id,
                "sensor_model": models[i % len(models)],
                "sensor_type": sensors[i % len(sensors)],
                "temperature_c": round(21.5 + ((i * 1.35) % 45.0), 2),
                "humidity_pct": round(45.0 + ((i * 2.1) % 50.0), 1),
                "pressure_hpa": round(1013.25 + ((i * 0.45) % 25.0), 2),
                "vibration_rms": round(0.02 + ((i * 0.05) % 1.5), 3),
                "battery_pct": int(max(10, 100 - (i % 90))),
                "payload_bytes": int(96 + ((i * 16) % 256)),
            }
            base_record: Dict[str, Any] = {
                "mqtt_topic": topic.replace("#", f"node_{dev_id}/data").replace("+", "sensor1"),
                "mqtt_qos": int(qos),
                "timestamp": f"2026-03-{((i % 28) + 1):02d}T16:{((i % 60)):02d}:00Z"
            }
            if flatten_json:
                base_record.update(nested_payload)
            else:
                base_record["payload"] = json.dumps(nested_payload, ensure_ascii=False)
            records.append(base_record)
        return records

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        start_time = time.time()
        max_messages = max(1, min(int(config.get("max_messages", 100)), 20000))
        timeout_sec = max(1, min(int(config.get("timeout_seconds", 3)), 30))
        flatten_json = bool(config.get("flatten_json", True))
        simulation_mode = bool(config.get("simulation_mode", False))

        broker_host = str(config.get("broker_host", "broker.emqx.io")).strip()
        port = int(config.get("port", 1883))
        topic = str(config.get("topic", "factory/sensors/telemetry/#")).strip()
        client_id_prefix = str(config.get("client_id", "ml_pipeline_iot_subscriber")).strip()
        qos = int(config.get("qos", 0))

        records: List[Dict[str, Any]] = []
        delivery_status = "SIMULATED"
        status_msg = ""

        # Live Execution Attempt with Paho MQTT
        if not simulation_mode and broker_host and topic:
            try:
                import paho.mqtt.client as mqtt

                unique_client_id = f"{client_id_prefix}_{uuid.uuid4().hex[:6]}" if client_id_prefix else f"subscriber_{uuid.uuid4().hex[:8]}"
                collected_msgs: List[Dict[str, Any]] = []
                connected_event = threading.Event()
                connection_error: List[str] = []

                def on_connect(client, userdata, flags, rc, *args):
                    reason_code = getattr(rc, "value", rc)
                    if reason_code == 0:
                        connected_event.set()
                        client.subscribe(topic, qos=qos)
                    else:
                        connection_error.append(f"Connection rejected by broker with reason code {reason_code}")
                        connected_event.set()

                def on_message(client, userdata, msg):
                    try:
                        raw_payload = msg.payload.decode("utf-8", "ignore")
                        try:
                            parsed = json.loads(raw_payload)
                        except Exception:
                            parsed = {"raw_payload": raw_payload}
                    except Exception:
                        parsed = {"raw_payload": str(msg.payload)}

                    row: Dict[str, Any] = {
                        "mqtt_topic": msg.topic,
                        "mqtt_qos": int(msg.qos),
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    }
                    if flatten_json:
                        if isinstance(parsed, dict):
                            row.update(parsed)
                        else:
                            row["payload"] = parsed
                    else:
                        row["payload"] = json.dumps(parsed, ensure_ascii=False) if isinstance(parsed, (dict, list)) else str(parsed)

                    collected_msgs.append(row)

                try:
                    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=unique_client_id)
                except (AttributeError, TypeError):
                    client = mqtt.Client(client_id=unique_client_id)

                client.on_connect = on_connect
                client.on_message = on_message

                u = str(config.get("username", "")).strip()
                p = str(config.get("password", "")).strip()
                if u or p:
                    client.username_pw_set(u, p)

                # Connect with keepalive
                client.connect(broker_host, port, keepalive=30)
                client.loop_start()

                # Wait for connection handshake (CONNACK)
                conn_timeout = min(float(timeout_sec), 5.0)
                connected_ok = connected_event.wait(timeout=conn_timeout)

                if not connected_ok:
                    client.loop_stop()
                    try:
                        client.disconnect()
                    except Exception:
                        pass
                    raise ValidationException(
                        f"Failed connecting to MQTT broker at '{broker_host}:{port}' within {conn_timeout:.1f}s. "
                        f"Verify broker hostname, port {port}, and network accessibility, or enable Simulation Mode."
                    )

                if connection_error:
                    client.loop_stop()
                    try:
                        client.disconnect()
                    except Exception:
                        pass
                    raise ValidationException(
                        f"MQTT broker '{broker_host}:{port}' rejected connection: {connection_error[0]}. "
                        f"Check username, password, or client permissions."
                    )

                # Listen for incoming published messages
                t_end = time.time() + timeout_sec
                while time.time() < t_end and len(collected_msgs) < max_messages:
                    time.sleep(0.05)

                client.loop_stop()
                try:
                    client.disconnect()
                except Exception:
                    pass

                delivery_status = "LIVE"
                if collected_msgs:
                    records = collected_msgs
                    status_msg = f"Ingested {len(records)} live MQTT messages from broker '{broker_host}' on topic '{topic}'."
                else:
                    records = []
                    status_msg = f"Connected to live broker '{broker_host}', but no messages were published to '{topic}' during the {timeout_sec}s sampling window."

            except ImportError:
                raise ValidationException("paho-mqtt driver not installed. Please install 'paho-mqtt' to connect to MQTT brokers.")
            except ValidationException:
                raise
            except Exception as e:
                logger.warning(f"MQTT connection failed: {e}")
                err_str = str(e) or type(e).__name__
                if "refused" in err_str.lower() or "connectionrefused" in err_str.lower():
                    raise ValidationException(
                        f"Failed connecting to MQTT broker at '{broker_host}:{port}' (Connection refused). "
                        f"Verify the broker is running and port {port} is accessible, or enable Simulation Mode."
                    )
                raise ValidationException(f"MQTT connection to '{broker_host}:{port}' failed: {err_str}")

        # Simulation Mode (when simulation_mode is True)
        if simulation_mode:
            records = self._generate_benchmark_data(config)
            delivery_status = "SIMULATED"
            status_msg = f"Loaded {len(records)} benchmark IoT sensor records for topic '{topic}'."

        if pd is not None and records:
            if flatten_json:
                df_out = pd.json_normalize(records)
            else:
                df_out = pd.DataFrame(records)
        elif pd is not None:
            df_out = pd.DataFrame()
        else:
            df_out = records

        duration_ms = int((time.time() - start_time) * 1000)
        col_count = int(len(df_out.columns) if hasattr(df_out, "columns") else (len(records[0].keys()) if records else 0))

        summary = {
            "title": "MQTT / IoT Sensor Stream Metrics",
            "tag": "MQTT",
            "message": status_msg,
            "status": delivery_status,
            "broker": broker_host,
            "topic": topic,
            "messages_ingested": int(len(records)),
            "columns_count": int(col_count),
            "duration_ms": int(duration_ms)
        }

        metrics = {
            "title": "MQTT / IoT Sensor Stream Metrics",
            "tag": "MQTT",
            "messages_ingested": int(len(records)),
            "columns_count": int(col_count),
            "duration_ms": int(duration_ms),
            "delivery_status": delivery_status
        }

        return {
            "dataframe": df_out,
            "output": df_out,
            "output_summary": summary,
            "metrics": metrics
        }

    def to_code(self, config: Dict[str, Any]) -> str:
        broker = config.get("broker_host", "broker.emqx.io")
        port = config.get("port", 1883)
        topic = config.get("topic", "factory/sensors/#")
        max_msgs = config.get("max_messages", 100)

        return (
            f"# Generated MQTT IoT Stream Ingestion Script\n"
            f"import paho.mqtt.client as mqtt\n"
            f"import json\n"
            f"import pandas as pd\n\n"
            f"records = []\n"
            f"def on_msg(client, userdata, msg):\n"
            f"    records.append(json.loads(msg.payload.decode('utf-8')))\n"
            f"client = mqtt.Client()\n"
            f"client.on_message = on_msg\n"
            f"client.connect('{broker}', {port}, 60)\n"
            f"client.subscribe('{topic}')\n"
            f"# Ingest micro-batch\n"
            f"df = pd.json_normalize(records)\n"
            f"print(f'Ingested {{len(df)}} sensor records.')\n"
        )
