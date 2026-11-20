"""
Test ensuring all pipeline recipes accept the 'context' keyword argument when executed
by DAGExecutor (workflow engine) or 'Test Step' in the frontend canvas.
"""

import pytest
from backend.app.recipes.ingestion.ftp_sftp_watcher import FTPSFTPWatcherRecipe
from backend.app.recipes.ingestion.file_system_watcher import FileSystemWatcherRecipe
from backend.app.recipes.ingestion.web_scraper import WebScraperRecipe
from backend.app.recipes.ingestion.rss_atom_aggregator import RSSAtomAggregatorRecipe
from backend.app.recipes.ingestion.archive_unpacker import ArchiveUnpackerRecipe
from backend.app.recipes.ingestion.email_attachment_extractor import EmailAttachmentExtractorRecipe
from backend.app.recipes.flow_control.batch_scheduler import BatchSchedulerRecipe
from backend.app.recipes.flow_control.dead_letter_queue import DeadLetterQueueRecipe
from backend.app.recipes.flow_control.retry_error_handler import RetryErrorHandlerRecipe
from backend.app.recipes.flow_control.sub_pipeline_call import SubPipelineCallRecipe


def test_ingestion_recipes_accept_context_parameter():
    test_context = {
        "pipeline_id": "wf_test_123",
        "tenant_id": "tenant_default",
        "model": None,
        "scaler": None,
    }

    # 1. FTP / SFTP Watcher
    ftp_rec = FTPSFTPWatcherRecipe()
    res = ftp_rec.execute(
        inputs={},
        config={"protocol": "ftp", "host": "127.0.0.1", "port": 2121, "simulation_fallback": True},
        context=test_context,
    )
    assert "dataframe" in res
    assert "metrics" in res

    # 2. File System Watcher
    fs_rec = FileSystemWatcherRecipe()
    res = fs_rec.execute(
        inputs={},
        config={"directory_path": "./data", "simulation_mode": True},
        context=test_context,
    )
    assert "dataframe" in res

    # 3. Web Scraper
    ws_rec = WebScraperRecipe()
    res = ws_rec.execute(
        inputs={},
        config={"url": "https://example.com", "simulation_mode": True},
        context=test_context,
    )
    assert "dataframe" in res

    # 4. RSS Aggregator
    rss_rec = RSSAtomAggregatorRecipe()
    res = rss_rec.execute(
        inputs={},
        config={"feed_urls": "https://feeds.bbci.co.uk/news/rss.xml", "simulation_mode": True},
        context=test_context,
    )
    assert "dataframe" in res

    # 5. Archive Unpacker
    arch_rec = ArchiveUnpackerRecipe()
    res = arch_rec.execute(
        inputs={},
        config={"archive_path": "daily.zip", "simulation_mode": True},
        context=test_context,
    )
    assert "dataframe" in res

    # 6. Email Attachment Extractor
    email_rec = EmailAttachmentExtractorRecipe()
    res = email_rec.execute(
        inputs={},
        config={"imap_server": "imap.example.com", "username": "u", "password": "p", "simulation_mode": True},
        context=test_context,
    )
    assert "dataframe" in res


def test_flow_control_recipes_accept_context_parameter():
    test_context = {
        "pipeline_id": "wf_test_123",
        "tenant_id": "tenant_default",
    }

    # 7. Batch Scheduler
    batch_rec = BatchSchedulerRecipe()
    res = batch_rec.execute(
        inputs={},
        config={"batch_strategy": "partition_column", "simulation_fallback": True},
        context=test_context,
    )
    assert "dataframe" in res

    # 8. Dead Letter Queue
    dlq_rec = DeadLetterQueueRecipe()
    res = dlq_rec.execute(
        inputs={},
        config={"dlq_mode": "trap_and_isolate", "simulation_mode": True},
        context=test_context,
    )
    assert "dataframe" in res
