"""
RabbitMQ Consumer Recipe (Ingestion & Orchestration -> Streaming & Change Data).

Consumes message queues from RabbitMQ message brokers into micro-batches of DataFrame rows.
Supports host/port/vhost, AMQP connection URIs, queue draining, auto-acknowledgment options,
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


class RabbitMQConsumerRecipe(BaseRecipe):
    recipe_id = "rabbitmq_consumer"
    name = "RabbitMQ Message Consumer"
    version = "1.0.0"
    category = "ingestion"
    group = "Ingestion & Orchestration"
    subgroup = "Streaming & Change Data"
    description = "Drains message queues from RabbitMQ brokers into micro-batches of tabular rows."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    inputs: List[RecipePort] = []

    outputs = [
        RecipePort(
            id="output",
            label="RabbitMQ Stream Dataset",
            type="dataframe",
            description="DataFrame containing drained RabbitMQ queue messages"
        )
    ]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "sections": [
                {
                    "id": "connection",
                    "title": "Broker Connection",
                    "description": "Configure connection mode, broker parameters, or AMQP URI",
                },
                {
                    "id": "queue",
                    "title": "Queue & Acknowledgment",
                    "description": "Target queue name and message acknowledgment settings",
                },
                {
                    "id": "runtime_limits",
                    "title": "Batching & Parsing",
                    "description": "Batch message limit, JSON flattening, and simulation mode",
                },
            ],
            "properties": {
                "connection_mode": {
                    "type": "string",
                    "title": "Connection Mode",
                    "input_type": "dropdown",
                    "section": "connection",
                    "enum": ["host_port", "amqp_uri", "credential"],
                    "default": "host_port",
                    "description": "Choose between host/port parameters, a full AMQP URI, or stored credentials.",
                },
                "host": {
                    "type": "string",
                    "title": "Broker Host",
                    "input_type": "text",
                    "section": "connection",
                    "default": "localhost",
                    "description": "RabbitMQ server hostname or IP address.",
                    "visible_when": {
                        "field": "connection_mode",
                        "equals": "host_port",
                    },
                },
                "port": {
                    "type": "integer",
                    "title": "Port",
                    "input_type": "number",
                    "section": "connection",
                    "default": 5672,
                    "description": "RabbitMQ server port (default 5672 for AMQP, 5671 for TLS).",
                    "visible_when": {
                        "field": "connection_mode",
                        "equals": "host_port",
                    },
                },
                "virtual_host": {
                    "type": "string",
                    "title": "Virtual Host",
                    "input_type": "text",
                    "section": "connection",
                    "default": "/",
                    "description": "Target virtual host on the RabbitMQ broker (default '/').",
                    "visible_when": {
                        "field": "connection_mode",
                        "equals": "host_port",
                    },
                },
                "username": {
                    "type": "string",
                    "title": "Username",
                    "input_type": "text",
                    "section": "connection",
                    "default": "guest",
                    "description": "RabbitMQ user login name.",
                    "visible_when": {
                        "field": "connection_mode",
                        "equals": "host_port",
                    },
                },
                "password": {
                    "type": "string",
                    "title": "Password",
                    "input_type": "password",
                    "section": "connection",
                    "default": "guest",
                    "description": "RabbitMQ user password.",
                    "visible_when": {
                        "field": "connection_mode",
                        "equals": "host_port",
                    },
                },
                "amqp_uri": {
                    "type": "string",
                    "title": "AMQP Connection URI",
                    "input_type": "text",
                    "section": "connection",
                    "default": "",
                    "description": "Full AMQP connection string e.g. amqp://guest:guest@localhost:5672/%2F",
                    "visible_when": {
                        "field": "connection_mode",
                        "equals": "amqp_uri",
                    },
                },
                "credential_id": {
                    "type": "string",
                    "title": "Connected RabbitMQ Credential",
                    "input_type": "credential_dropdown",
                    "provider": "rabbitmq",
                    "section": "connection",
                    "description": "Select stored RabbitMQ credentials from platform settings.",
                    "visible_when": {
                        "field": "connection_mode",
                        "equals": "credential",
                    },
                },
                "queue_name": {
                    "type": "string",
                    "title": "Queue Name",
                    "input_type": "text",
                    "section": "queue",
                    "default": "orders-incoming",
                    "description": "Target RabbitMQ queue name to consume from.",
                },
                "auto_ack": {
                    "type": "boolean",
                    "title": "Auto Acknowledge (Acknowledge on Read)",
                    "input_type": "checkbox",
                    "section": "queue",
                    "default": True,
                    "description": "Automatically acknowledge messages upon pulling them from the queue.",
                },
                "max_messages": {
                    "type": "integer",
                    "title": "Max Messages per Batch",
                    "input_type": "number",
                    "section": "runtime_limits",
                    "default": 100,
                    "description": "Maximum number of messages to drain in this batch (1 to 20,000).",
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
                    "description": "When enabled or when RabbitMQ broker is offline, generates realistic order event stream benchmark records.",
                },
            },
            "required": ["queue_name"],
        }

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        mode = str(config.get("connection_mode", "host_port")).strip().lower()

        if not config.get("simulation_mode"):
            if not str(config.get("queue_name", "")).strip():
                errors.append("Queue name is required.")

            if mode == "host_port":
                if not str(config.get("host", "")).strip():
                    errors.append("Broker host is required in 'host_port' mode.")
            elif mode == "amqp_uri":
                uri = str(config.get("amqp_uri", "")).strip()
                if not uri and not config.get("host"):
                    errors.append("AMQP URI is required in 'amqp_uri' mode.")
            elif mode == "credential":
                if not config.get("credential_id"):
                    errors.append("Credential ID is required in 'credential' mode.")

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
        queue = str(config.get("queue_name", "orders-incoming"))
        actions = ["DISPATCH_TASK", "NOTIFY_CUSTOMER", "SYNC_ACCOUNT", "INDEX_DOCUMENT", "AUDIT_EVENT"]
        priorities = [0, 1, 2, 5, 9]

        records: List[Dict[str, Any]] = []
        for i in range(limit):
            records.append({
                "rabbitmq_queue": queue,
                "delivery_tag": 1000 + i,
                "message_id": f"msg_amqp_{500000 + i}",
                "action": actions[i % len(actions)],
                "priority": priorities[i % len(priorities)],
                "redelivered": False if i % 8 != 0 else True,
                "exchange": "amq.direct",
                "routing_key": queue,
                "payload_bytes": 140 + ((i * 24) % 512),
                "timestamp": f"2026-03-{((i % 28) + 1):02d}T15:{((i % 60)):02d}:00Z"
            })
        return records

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        start_time = time.time()
        max_messages = max(1, min(int(config.get("max_messages", 100)), 20000))
        auto_ack = bool(config.get("auto_ack", True))
        flatten_json = bool(config.get("flatten_json", True))
        simulation_mode = bool(config.get("simulation_mode", False))

        queue_name = str(config.get("queue_name", "orders-incoming")).strip()
        host = str(config.get("host", "localhost")).strip()
        port = int(config.get("port", 5672))
        vhost = str(config.get("virtual_host", "/")).strip()
        username = str(config.get("username", "guest")).strip()
        password = str(config.get("password", "guest")).strip()

        records: List[Dict[str, Any]] = []
        delivery_status = "SIMULATED"
        status_msg = ""

        # Live Execution Attempt with pika
        if not simulation_mode and queue_name and host:
            try:
                import pika
                credentials = pika.PlainCredentials(username, password)
                parameters = pika.ConnectionParameters(
                    host=host,
                    port=port,
                    virtual_host=vhost,
                    credentials=credentials,
                    connection_attempts=1,
                    retry_delay=1
                )
                connection = pika.BlockingConnection(parameters)
                channel = connection.channel()

                pulled_count = 0
                for _ in range(max_messages):
                    method_frame, header_frame, body = channel.basic_get(queue=queue_name, auto_ack=auto_ack)
                    if method_frame:
                        try:
                            parsed_body = json.loads(body.decode("utf-8", "ignore"))
                        except Exception:
                            parsed_body = {"body_text": body.decode("utf-8", "ignore")}

                        row = {
                            "rabbitmq_queue": queue_name,
                            "delivery_tag": method_frame.delivery_tag,
                            "redelivered": method_frame.redelivered,
                            "exchange": method_frame.exchange,
                            "routing_key": method_frame.routing_key,
                            **parsed_body
                        }
                        records.append(row)
                        pulled_count += 1
                    else:
                        break  # Queue empty

                connection.close()
                delivery_status = "LIVE"
                status_msg = f"Drained {len(records)} live messages from RabbitMQ queue '{queue_name}'."
            except ImportError:
                raise ValidationException("pika driver not installed. Please install 'pika' to connect to RabbitMQ.")
            except Exception as e:
                logger.warning(f"RabbitMQ connection failed: {e}")
                raise ValidationException(f"RabbitMQ connection failed: {str(e)}")

        # Simulation Mode (ONLY when simulation_mode is True)
        if simulation_mode:
            records = self._generate_benchmark_data(config)
            status_msg = f"Loaded {len(records)} benchmark queue messages for RabbitMQ queue '{queue_name}'."

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
            "title": "RabbitMQ Consumer Summary",
            "message": status_msg,
            "status": delivery_status,
            "queue_name": queue_name,
            "messages_drained": len(records),
            "columns_count": col_count,
            "duration_ms": duration_ms
        }

        metrics = {
            "messages_drained": len(records),
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
        host = config.get("host", "localhost")
        port = config.get("port", 5672)
        vhost = config.get("virtual_host", "/")
        queue = config.get("queue_name", "orders-incoming")
        max_msgs = config.get("max_messages", 100)

        return (
            f"# Generated RabbitMQ Queue Consumer Script\n"
            f"import pika\n"
            f"import json\n"
            f"import pandas as pd\n\n"
            f"params = pika.ConnectionParameters(host='{host}', port={port}, virtual_host='{vhost}')\n"
            f"connection = pika.BlockingConnection(params)\n"
            f"channel = connection.channel()\n"
            f"records = []\n"
            f"for _ in range({max_msgs}):\n"
            f"    method, header, body = channel.basic_get(queue='{queue}', auto_ack=True)\n"
            f"    if method:\n"
            f"        records.append(json.loads(body))\n"
            f"    else:\n"
            f"        break\n"
            f"connection.close()\n"
            f"df = pd.json_normalize(records)\n"
            f"print(f'Drained {{len(df)}} messages from queue {queue}.')\n"
        )
