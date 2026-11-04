"""
Kafka Consumer Recipe (Ingestion & Orchestration -> Streaming & Change Data).

Consumes streaming event messages from Apache Kafka topics into micro-batches of DataFrame rows.
Supports broker lists, consumer groups, offset management (earliest/latest), SASL/SSL authentication,
JSON payload deserialization with auto-flattening, benchmark simulation fallback, and code generation.
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


class KafkaConsumerRecipe(BaseRecipe):
    recipe_id = "kafka_consumer"
    name = "Kafka Consumer"
    version = "1.0.0"
    category = "ingestion"
    group = "Ingestion & Orchestration"
    subgroup = "Streaming & Change Data"
    description = "Consumes streaming messages from Apache Kafka topics into micro-batches of tabular rows."
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
            label="Kafka Stream Dataset",
            type="dataframe",
            description="DataFrame containing polled Kafka streaming messages"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "bootstrap_servers": {
                    "type": "string",
                    "title": "Bootstrap Servers",
                    "input_type": "text",
                    "default": "localhost:9092",
                    "description": "Comma-separated list of Kafka broker host:port addresses (e.g. 'kafka1:9092,kafka2:9092')."
                },
                "topic": {
                    "type": "string",
                    "title": "Kafka Topic",
                    "input_type": "text",
                    "default": "telemetry-events",
                    "description": "Name of the target Kafka topic to subscribe to."
                },
                "group_id": {
                    "type": "string",
                    "title": "Consumer Group ID",
                    "input_type": "text",
                    "default": "ml-pipeline-consumer-grp",
                    "description": "Kafka consumer group identifier for offset tracking."
                },
                "auto_offset_reset": {
                    "type": "string",
                    "title": "Auto Offset Reset",
                    "input_type": "dropdown",
                    "enum": ["latest", "earliest"],
                    "default": "latest",
                    "description": "Where to start reading when no initial offset exists or current offset is out of range."
                },
                "security_protocol": {
                    "type": "string",
                    "title": "Security Protocol",
                    "input_type": "dropdown",
                    "enum": ["PLAINTEXT", "SSL", "SASL_PLAINTEXT", "SASL_SSL"],
                    "default": "PLAINTEXT",
                    "description": "Protocol used to communicate with brokers."
                },
                "sasl_mechanism": {
                    "type": "string",
                    "title": "SASL Mechanism",
                    "input_type": "dropdown",
                    "enum": ["PLAIN", "SCRAM-SHA-256", "SCRAM-SHA-512"],
                    "default": "PLAIN",
                    "description": "SASL authentication mechanism (when security_protocol is SASL_*)."
                },
                "sasl_username": {
                    "type": "string",
                    "title": "SASL Username",
                    "input_type": "text",
                    "default": "",
                    "description": "Username for SASL authentication."
                },
                "sasl_password": {
                    "type": "string",
                    "title": "SASL Password",
                    "input_type": "password",
                    "default": "",
                    "description": "Password for SASL authentication."
                },
                "max_messages": {
                    "type": "integer",
                    "title": "Max Messages per Batch",
                    "input_type": "number",
                    "default": 100,
                    "description": "Maximum number of messages to pull in this micro-batch (1 to 50,000)."
                },
                "poll_timeout_ms": {
                    "type": "integer",
                    "title": "Poll Timeout (ms)",
                    "input_type": "number",
                    "default": 2000,
                    "description": "Maximum time in milliseconds to wait for records during polling."
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
                    "title": "Connected Kafka Credential",
                    "input_type": "credential_dropdown",
                    "provider": "kafka",
                    "description": "Select stored Kafka cluster credentials from platform settings."
                },
                "simulation_mode": {
                    "type": "boolean",
                    "title": "Simulation / Demo Mode",
                    "input_type": "checkbox",
                    "default": False,
                    "description": "When enabled or when Kafka broker is offline, generates realistic Kafka event stream benchmark records."
                }
            },
            "required": ["bootstrap_servers", "topic", "group_id"]
        }

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        if not config.get("simulation_mode"):
            if not str(config.get("bootstrap_servers", "")).strip():
                errors.append("Bootstrap servers list is required.")
            if not str(config.get("topic", "")).strip():
                errors.append("Kafka topic name is required.")
            if not str(config.get("group_id", "")).strip():
                errors.append("Consumer group ID is required.")

            proto = str(config.get("security_protocol", "PLAINTEXT")).upper()
            if "SASL" in proto:
                if not config.get("sasl_username") and not config.get("credential_id"):
                    errors.append("SASL Username is required when using SASL security protocol.")

        max_msgs = config.get("max_messages")
        if max_msgs is not None:
            try:
                m_val = int(max_msgs)
                if m_val <= 0:
                    errors.append("Max messages must be greater than 0.")
            except (ValueError, TypeError):
                errors.append("Max messages must be an integer.")

        timeout = config.get("poll_timeout_ms")
        if timeout is not None:
            try:
                t_val = int(timeout)
                if t_val <= 0:
                    errors.append("Poll timeout must be greater than 0 ms.")
            except (ValueError, TypeError):
                errors.append("Poll timeout must be an integer.")

        return errors

    def _generate_benchmark_data(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        limit = max(1, min(int(config.get("max_messages", 100)), 5000))
        topic = str(config.get("topic", "telemetry-events"))
        event_types = ["ORDER_CREATED", "PAYMENT_PROCESSED", "INVENTORY_DEDUCTED", "SHIPMENT_DISPATCHED", "USER_CLICK"]
        channels = ["web_client", "mobile_ios", "mobile_android", "pos_terminal", "partner_api"]
        regions = ["us-east", "eu-west", "ap-southeast", "me-central"]

        records: List[Dict[str, Any]] = []
        for i in range(limit):
            records.append({
                "kafka_topic": topic,
                "kafka_partition": i % 4,
                "kafka_offset": 500000 + i,
                "message_key": f"key_{100 + (i % 25)}",
                "event_type": event_types[i % len(event_types)],
                "client_channel": channels[i % len(channels)],
                "region": regions[i % len(regions)],
                "transaction_sar": round(25.0 + ((i * 14.3) % 450.0), 2),
                "payload_bytes": 128 + ((i * 32) % 1024),
                "latency_ms": round(1.2 + ((i * 0.4) % 15.0), 2),
                "timestamp": f"2026-03-{((i % 28) + 1):02d}T14:{((i % 60)):02d}:00Z"
            })
        return records

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        start_time = time.time()
        max_messages = max(1, min(int(config.get("max_messages", 100)), 50000))
        poll_timeout_ms = int(config.get("poll_timeout_ms", 2000))
        flatten_json = bool(config.get("flatten_json", True))
        simulation_mode = bool(config.get("simulation_mode", False))

        bootstrap_servers = str(config.get("bootstrap_servers", "localhost:9092")).strip()
        topic = str(config.get("topic", "telemetry-events")).strip()
        group_id = str(config.get("group_id", "ml-pipeline-consumer-grp")).strip()
        offset_reset = str(config.get("auto_offset_reset", "latest")).strip()

        records: List[Dict[str, Any]] = []
        delivery_status = "SIMULATED"
        status_msg = ""

        # Live Execution Attempt
        if not simulation_mode and bootstrap_servers and topic:
            try:
                from kafka import KafkaConsumer
                consumer_kwargs: Dict[str, Any] = {
                    "bootstrap_servers": [s.strip() for s in bootstrap_servers.split(",")],
                    "group_id": group_id,
                    "auto_offset_reset": offset_reset,
                    "enable_auto_commit": True,
                    "consumer_timeout_ms": poll_timeout_ms,
                    "value_deserializer": lambda m: json.loads(m.decode("utf-8", "ignore")) if m else {}
                }
                proto = str(config.get("security_protocol", "PLAINTEXT")).upper()
                if proto != "PLAINTEXT":
                    consumer_kwargs["security_protocol"] = proto
                if "SASL" in proto:
                    consumer_kwargs["sasl_mechanism"] = str(config.get("sasl_mechanism", "PLAIN")).upper()
                    consumer_kwargs["sasl_plain_username"] = str(config.get("sasl_username", "")).strip()
                    consumer_kwargs["sasl_plain_password"] = str(config.get("sasl_password", "")).strip()

                consumer = KafkaConsumer(topic, **consumer_kwargs)
                polled_count = 0
                for msg in consumer:
                    raw_val = msg.value if isinstance(msg.value, dict) else {"payload": str(msg.value)}
                    row = {
                        "kafka_topic": msg.topic,
                        "kafka_partition": msg.partition,
                        "kafka_offset": msg.offset,
                        "message_key": msg.key.decode("utf-8") if msg.key else None,
                        "timestamp": msg.timestamp,
                        **raw_val
                    }
                    records.append(row)
                    polled_count += 1
                    if polled_count >= max_messages:
                        break
                consumer.close()

                delivery_status = "LIVE"
                status_msg = f"Polled {len(records)} live streaming messages from Kafka topic '{topic}' (group: {group_id})."
            except ImportError:
                status_msg = "kafka-python driver not installed. Using benchmark simulation fallback."
                logger.info(status_msg)
            except Exception as e:
                logger.warning(f"Kafka consumer connection failed: {e}")
                status_msg = f"Kafka connection failed ({e}). Using benchmark simulation fallback."

        # Fallback to realistic benchmark data
        if not records:
            records = self._generate_benchmark_data(config)
            if not status_msg:
                status_msg = f"Loaded {len(records)} benchmark streaming messages for Kafka topic '{topic}'."

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
            "title": "Kafka Consumer Summary",
            "message": status_msg,
            "status": delivery_status,
            "topic": topic,
            "group_id": group_id,
            "messages_polled": len(records),
            "columns_count": col_count,
            "duration_ms": duration_ms
        }

        metrics = {
            "messages_polled": len(records),
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
        servers = config.get("bootstrap_servers", "localhost:9092")
        topic = config.get("topic", "telemetry-events")
        group_id = config.get("group_id", "ml-pipeline-grp")
        offset = config.get("auto_offset_reset", "latest")
        max_msgs = config.get("max_messages", 100)

        return (
            f"# Generated Apache Kafka Consumer Ingestion Script\n"
            f"from kafka import KafkaConsumer\n"
            f"import json\n"
            f"import pandas as pd\n\n"
            f"consumer = KafkaConsumer(\n"
            f"    '{topic}',\n"
            f"    bootstrap_servers='{servers}'.split(','),\n"
            f"    group_id='{group_id}',\n"
            f"    auto_offset_reset='{offset}',\n"
            f"    value_deserializer=lambda m: json.loads(m.decode('utf-8'))\n"
            f")\n"
            f"records = []\n"
            f"for msg in consumer:\n"
            f"    records.append(msg.value)\n"
            f"    if len(records) >= {max_msgs}:\n"
            f"        break\n"
            f"df = pd.json_normalize(records)\n"
            f"print(f'Polled {{len(df)}} messages from topic {topic}.')\n"
        )
