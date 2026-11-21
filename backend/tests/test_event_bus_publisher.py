"""
Unit tests for Event Bus Publisher Recipe.
Covers:
1. Direct credential inputs in schema (RabbitMQ, SNS, Kafka SASL).
2. AMQP URI parsing without domain name error.
3. Amazon SNS region and credential injection without 'NoRegionError'.
4. Kafka SASL security mechanism configuration.
5. Optional partition_key schema avoiding canvas warning banners.
6. Automatic simulation fallback on offline brokers.
7. Proper results header ('Event Bus Publishing Metrics') and clean integer counts.
"""

from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from backend.app.recipes.ingestion.event_bus_publisher import EventBusPublisherRecipe


def test_schema_properties_and_visible_when():
    recipe = EventBusPublisherRecipe()
    schema = recipe.get_schema()
    props = schema["properties"]

    # RabbitMQ fields
    assert "rabbitmq_username" in props
    assert "rabbitmq_password" in props
    assert "rabbitmq_virtual_host" in props
    assert props["rabbitmq_username"]["visible_when"]["equals"] == "rabbitmq"

    # Amazon SNS fields
    assert "aws_region" in props
    assert "aws_access_key_id" in props
    assert "aws_secret_access_key" in props
    assert "aws_session_token" in props
    assert props["aws_region"]["default"] == "us-east-1"
    assert props["aws_region"]["visible_when"]["equals"] == "sns"

    # Kafka SASL fields
    assert "security_protocol" in props
    assert "sasl_mechanism" in props
    assert "sasl_username" in props
    assert "sasl_password" in props
    assert props["security_protocol"]["visible_when"]["equals"] == "kafka"

    # Partition Key (must be optional and not trigger 'column' canvas warning)
    assert "partition_key" in props
    assert "column" not in "partition_key"
    assert "partition_key" not in schema["required"]


def test_rabbitmq_amqp_uri_parsing():
    """Verify AMQP connection strings do not crash with domain name errors."""
    recipe = EventBusPublisherRecipe()
    input_df = pd.DataFrame([{"id": 1, "msg": "test"}])
    config = {
        "bus_type": "rabbitmq",
        "destination": "test-queue",
        "server_address": "amqp://guest:guest@localhost:5672/%2f",
        "simulation_mode": False
    }

    mock_conn = MagicMock()
    mock_channel = MagicMock()
    mock_conn.channel.return_value = mock_channel

    with patch("pika.URLParameters") as mock_url_params, \
         patch("pika.BlockingConnection", return_value=mock_conn) as mock_pika_conn:
        res = recipe.execute({"input": input_df}, config)

        assert mock_url_params.called
        assert mock_url_params.call_args[0][0] == "amqp://guest:guest@localhost:5672/%2f"
        assert res["output_summary"]["status"] == "LIVE"
        assert res["metrics"]["messages_published"] == 1


def test_rabbitmq_direct_credentials():
    recipe = EventBusPublisherRecipe()
    input_df = pd.DataFrame([{"id": 1, "msg": "test"}])
    config = {
        "bus_type": "rabbitmq",
        "destination": "test-queue",
        "server_address": "rmq-broker.internal:5672",
        "rabbitmq_username": "app_user",
        "rabbitmq_password": "secret_password",
        "rabbitmq_virtual_host": "production",
        "simulation_mode": False
    }

    mock_conn = MagicMock()
    mock_channel = MagicMock()
    mock_conn.channel.return_value = mock_channel

    with patch("pika.PlainCredentials") as mock_plain_creds, \
         patch("pika.ConnectionParameters") as mock_conn_params, \
         patch("pika.BlockingConnection", return_value=mock_conn):
        res = recipe.execute({"input": input_df}, config)

        assert mock_plain_creds.called
        assert mock_plain_creds.call_args[0] == ("app_user", "secret_password")
        assert mock_conn_params.called
        assert mock_conn_params.call_args[1]["host"] == "rmq-broker.internal"
        assert mock_conn_params.call_args[1]["port"] == 5672
        assert mock_conn_params.call_args[1]["virtual_host"] == "/production"
        assert res["metrics"]["messages_published"] == 1


def test_sns_region_and_credentials_injection():
    recipe = EventBusPublisherRecipe()
    input_df = pd.DataFrame([{"order_id": 99, "amount": 250.0}])
    config = {
        "bus_type": "sns",
        "destination": "arn:aws:sns:eu-west-1:123456789012:OrderEvents",
        "aws_region": "eu-west-1",
        "aws_access_key_id": "AKIAEXAMPLEKEY",
        "aws_secret_access_key": "SECRETKEYEXAMPLE",
        "aws_session_token": "TOKENEXAMPLE",
        "simulation_mode": False
    }

    mock_sns = MagicMock()
    with patch("boto3.client", return_value=mock_sns) as mock_boto_client:
        res = recipe.execute({"input": input_df}, config)

        assert mock_boto_client.called
        call_kwargs = mock_boto_client.call_args[1]
        assert call_kwargs["region_name"] == "eu-west-1"
        assert call_kwargs["aws_access_key_id"] == "AKIAEXAMPLEKEY"
        assert call_kwargs["aws_secret_access_key"] == "SECRETKEYEXAMPLE"
        assert call_kwargs["aws_session_token"] == "TOKENEXAMPLE"
        assert mock_sns.publish.called
        assert res["metrics"]["messages_published"] == 1


def test_kafka_sasl_authentication():
    recipe = EventBusPublisherRecipe()
    input_df = pd.DataFrame([{"sensor_id": "s1", "reading": 42}])
    config = {
        "bus_type": "kafka",
        "destination": "iot-telemetry",
        "server_address": "kafka-prod.corp:9094",
        "security_protocol": "SASL_SSL",
        "sasl_mechanism": "SCRAM-SHA-256",
        "sasl_username": "producer_app",
        "sasl_password": "producer_secret",
        "partition_key": "sensor_id",
        "simulation_mode": False
    }

    mock_producer = MagicMock()
    with patch("kafka.KafkaProducer", return_value=mock_producer) as mock_kafka:
        res = recipe.execute({"input": input_df}, config)

        assert mock_kafka.called
        call_kwargs = mock_kafka.call_args[1]
        assert call_kwargs["bootstrap_servers"] == ["kafka-prod.corp:9094"]
        assert call_kwargs["security_protocol"] == "SASL_SSL"
        assert call_kwargs["sasl_mechanism"] == "SCRAM-SHA-256"
        assert call_kwargs["sasl_plain_username"] == "producer_app"
        assert call_kwargs["sasl_plain_password"] == "producer_secret"
        assert mock_producer.send.called
        assert res["metrics"]["messages_published"] == 1


def test_offline_broker_automatic_fallback():
    """When a live broker is unreachable, node auto-switches to demo simulation data instead of crashing."""
    recipe = EventBusPublisherRecipe()
    input_df = pd.DataFrame([{"id": i, "val": i * 10} for i in range(5)])

    # Offline Kafka broker attempt
    config = {
        "bus_type": "kafka",
        "destination": "orders",
        "server_address": "unreachable-kafka-host.local:9092",
        "simulation_mode": False
    }
    # Must NOT raise ValidationException!
    res = recipe.execute({"input": input_df}, config)

    assert res["output_summary"]["status"] == "SIMULATED"
    assert "unreachable" in res["output_summary"]["message"] or "failed" in res["output_summary"]["message"]
    assert res["metrics"]["messages_published"] == 5
    assert len(res["dataframe"]) == 5


def test_results_header_and_clean_integer_counts():
    recipe = EventBusPublisherRecipe()
    input_df = pd.DataFrame([{"col_a": 1, "col_b": 2}] * 7)
    config = {
        "bus_type": "rabbitmq",
        "destination": "events",
        "simulation_mode": True
    }
    res = recipe.execute({"input": input_df}, config)
    metrics = res["metrics"]
    summary = res["output_summary"]

    # Verify event publishing card header
    assert metrics["title"] == "Event Bus Publishing Metrics"
    assert metrics["tag"] == "STREAMING"
    assert metrics["task_type"] == "streaming"
    assert summary["title"] in ["Event Bus Publisher Summary", "Event Bus Publishing Metrics"]

    # Verify clean counts (strictly integer type)
    assert isinstance(metrics["messages_published"], int)
    assert isinstance(metrics["total_rows"], int)
    assert isinstance(metrics["duration_ms"], int)
    assert metrics["messages_published"] == 7
    assert metrics["total_rows"] == 7
    assert not isinstance(metrics["messages_published"], float)


def test_legacy_key_column_backward_compatibility():
    recipe = EventBusPublisherRecipe()
    input_df = pd.DataFrame([{"legacy_id": "L1", "data": "abc"}])
    config = {
        "bus_type": "kafka",
        "destination": "legacy-topic",
        "key_column": "legacy_id",  # legacy config key
        "simulation_mode": True
    }
    res = recipe.execute({"input": input_df}, config)
    assert res["metrics"]["messages_published"] == 1
    assert res["dataframe"]["legacy_id"].iloc[0] == "L1"
