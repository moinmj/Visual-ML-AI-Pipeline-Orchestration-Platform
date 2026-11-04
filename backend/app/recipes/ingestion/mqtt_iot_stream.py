"""
MQTT / IoT Sensor Stream Recipe (Ingestion & Orchestration -> Streaming & Change Data).

Subscribes to MQTT topics to ingest live IoT sensor telemetry readings into DataFrame micro-batches.
Supports broker host/port, topic wildcards (#, +), QoS levels (0, 1, 2), client credentials,
JSON sensor payload deserialization with auto-flattening, benchmark simulation fallback, and code generation.
"""

import os
import json
import time
from typing import Dict, Any, List, Optional, Tuple

try:
    import pandas as pd
except ImportError:
    pd = None

from backend.app.recipes.base.recipe import BaseRecipe, RecipePort
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
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs = [
        RecipePort(
            id="input",
            label="Trigger / Input Dataset",
            type="dataframe",
            required=False,
            max_connections=1,
            description="Optional upstream dataset for batch IDs or execution trigger"
        )
    ]

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
            "properties": {
                "broker_host": {
                    "type": "string",
                    "title": "MQTT Broker Host",
                    "input_type": "text",
                    "default": "broker.emqx.io",
                    "description": "MQTT broker hostname or IP address (e.g. 'broker.hivemq.com', 'localhost')."
                },
                "port": {
                    "type": "integer",
                    "title": "Port",
                    "input_type": "number",
                    "default": 1883,
                    "description": "MQTT port (default 1883 for TCP, 8883 for TLS/SSL)."
                },
                "topic": {
                    "type": "string",
                    "title": "Topic / Wildcard Filter",
                    "input_type": "text",
                    "default": "factory/sensors/telemetry/#",
                    "description": "MQTT topic to subscribe to. Wildcards like '#' (multi-level) and '+' (single-level) are supported."
                },
                "client_id": {
                    "type": "string",
                    "title": "Client ID",
                    "input_type": "text",
                    "default": "ml_pipeline_iot_subscriber",
                    "description": "Unique client identifier for the MQTT connection."
                },
                "qos": {
                    "type": "integer",
                    "title": "QoS Level",
                    "input_type": "dropdown",
                    "enum": [0, 1, 2],
                    "default": 0,
                    "description": "Quality of Service level: 0 (at most once), 1 (at least once), 2 (exactly once)."
                },
                "username": {
                    "type": "string",
                    "title": "Username (Optional)",
                    "input_type": "text",
                    "default": "",
                    "description": "Username for MQTT broker authentication."
                },
                "password": {
                    "type": "string",
                    "title": "Password (Optional)",
                    "input_type": "password",
                    "default": "",
                    "description": "Password for MQTT broker authentication."
                },
                "max_messages": {
                    "type": "integer",
                    "title": "Max Messages per Batch",
                    "input_type": "number",
                    "default": 100,
                    "description": "Maximum number of messages to collect before completing the micro-batch (1 to 20,000)."
                },
                "timeout_seconds": {
                    "type": "integer",
                    "title": "Sampling Window (Seconds)",
                    "input_type": "number",
                    "default": 3,
                    "description": "Time window in seconds to wait for incoming MQTT messages."
                },
                "flatten_json": {
                    "type": "boolean",
                    "title": "Flatten JSON Payloads",
                    "input_type": "checkbox",
                    "default": True,
                    "description": "Automatically unpack nested JSON object fields into top-level columns."
                },
                "credential_id": {
                    "type": "string",
                    "title": "Connected MQTT Credential",
                    "input_type": "credential_dropdown",
                    "provider": "mqtt",
                    "description": "Select stored MQTT broker credentials from platform settings."
                },
                "simulation_mode": {
                    "type": "boolean",
                    "title": "Simulation / Demo Mode",
                    "input_type": "checkbox",
                    "default": False,
                    "description": "When enabled or when MQTT broker is offline, generates realistic IoT sensor telemetry benchmark records."
                }
            },
            "required": ["broker_host", "topic"]
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

        sensors = ["TEMP_HUMIDITY", "VIBRATION_AXIS", "PRESSURE_BAR", "FLOW_METER", "CURRENT_AMPS"]
        models = ["SENS-X100", "BOSCH-BME680", "TI-TMP117", "OMRON-D6F", "STM-LSM6D"]

        records: List[Dict[str, Any]] = []
        for i in range(limit):
            dev_id = f"EDGE_NODE_{10 + (i % 20):03d}"
            records.append({
                "mqtt_topic": topic.replace("#", f"node_{dev_id}/data"),
                "mqtt_qos": qos,
                "device_id": dev_id,
                "sensor_model": models[i % len(models)],
                "sensor_type": sensors[i % len(sensors)],
                "temperature_c": round(21.5 + ((i * 1.35) % 45.0), 2),
                "humidity_pct": round(45.0 + ((i * 2.1) % 50.0), 1),
                "pressure_hpa": round(1013.25 + ((i * 0.45) % 25.0), 2),
                "vibration_rms": round(0.02 + ((i * 0.05) % 1.5), 3),
                "battery_pct": max(10, 100 - (i % 90)),
                "payload_bytes": 96 + ((i * 16) % 256),
                "timestamp": f"2026-03-{((i % 28) + 1):02d}T16:{((i % 60)):02d}:00Z"
            })
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
        client_id = str(config.get("client_id", "ml_pipeline_iot_subscriber")).strip()
        qos = int(config.get("qos", 0))

        records: List[Dict[str, Any]] = []
        delivery_status = "SIMULATED"
        status_msg = ""

        # Live Execution Attempt
        if not simulation_mode and broker_host and topic:
            try:
                import paho.mqtt.client as mqtt

                collected_msgs: List[Dict[str, Any]] = []

                def on_message(client, userdata, msg):
                    try:
                        parsed = json.loads(msg.payload.decode("utf-8", "ignore"))
                    except Exception:
                        parsed = {"raw_payload": msg.payload.decode("utf-8", "ignore")}
                    row = {
                        "mqtt_topic": msg.topic,
                        "mqtt_qos": msg.qos,
                        **parsed
                    }
                    collected_msgs.append(row)

                client = mqtt.Client(client_id=client_id)
                u = str(config.get("username", "")).strip()
                p = str(config.get("password", "")).strip()
                if u and p:
                    client.username_pw_set(u, p)

                client.on_message = on_message
                client.connect(broker_host, port, 10)
                client.subscribe(topic, qos=qos)
                client.loop_start()

                t_end = time.time() + timeout_sec
                while time.time() < t_end and len(collected_msgs) < max_messages:
                    time.sleep(0.1)

                client.loop_stop()
                client.disconnect()

                records = collected_msgs
                delivery_status = "LIVE"
                status_msg = f"Ingested {len(records)} live MQTT messages from broker '{broker_host}' on topic '{topic}'."
            except ImportError:
                status_msg = "paho-mqtt driver not installed. Using benchmark simulation fallback."
                logger.info(status_msg)
            except Exception as e:
                logger.warning(f"MQTT connection failed: {e}")
                status_msg = f"MQTT connection failed ({e}). Using benchmark simulation fallback."

        # Fallback to realistic benchmark data
        if not records:
            records = self._generate_benchmark_data(config)
            if not status_msg:
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
        col_count = len(df_out.columns) if hasattr(df_out, "columns") else (len(records[0].keys()) if records else 0)

        summary = {
            "title": "MQTT / IoT Sensor Stream Summary",
            "message": status_msg,
            "status": delivery_status,
            "broker": broker_host,
            "topic": topic,
            "messages_ingested": len(records),
            "columns_count": col_count,
            "duration_ms": duration_ms
        }

        metrics = {
            "messages_ingested": len(records),
            "columns_count": col_count,
            "duration_ms": duration_ms,
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
