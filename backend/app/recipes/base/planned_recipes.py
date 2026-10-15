"""
Planned ("coming soon") recipes.

These are listed by GET /api/v1/recipes/ in exactly the same JSON shape as the
real recipes, but with "enabled": false. They have no implementation yet:
the registry refuses to run, test or validate them ("not available yet"), and
the AI recommender never suggests them.

To ship one: implement the recipe class with the SAME recipe_id, register it in
recipes/__init__.py and delete its line below. The registered class always
wins, so it automatically becomes "enabled": true.

Each entry: (recipe_id, name, port_kind, description). port_kind picks the
input/output ports shown on the canvas (see _PORTS).
"""
from typing import Dict, List, Tuple

from backend.app.recipes.base.recipe import RecipeMetadata, RecipePort


def _port(pid: str, label: str, ptype: str, required: bool = True, description: str = None) -> RecipePort:
    return RecipePort(id=pid, label=label, type=ptype, required=required, max_connections=1, description=description)


_DF_IN = _port("input", "Input Table", "dataframe")
_DF_OUT = _port("output", "Output Table", "dataframe", description="Primary dataset")

# port_kind -> (inputs, outputs)
_PORTS: Dict[str, Tuple[List[RecipePort], List[RecipePort]]] = {
    # reads data from somewhere: no input, one table out
    "source": ([], [_DF_OUT]),
    # table in -> table out
    "transform": ([_DF_IN], [_DF_OUT]),
    # sends / writes data out, passes the table through
    "sink": (
        [_port("input", "Input Data", "dataframe", required=False, description="Upstream dataset or metrics")],
        [_port("output", "Passthrough Output", "dataframe", description="Passthrough dataset for downstream nodes")],
    ),
    # supervised trainer: train_data -> model
    "trainer": (
        [_port("input", "Input (train_data)", "train_data")],
        [_port("output", "Output (model)", "model", description="Primary output")],
    ),
    # unsupervised: table -> table (with labels/components) + fitted model
    "unsupervised": (
        [_DF_IN],
        [_port("output_1", "Output Table", "dataframe", description="Primary dataset"),
         _port("output_2", "Output (model)", "model", description="Fitted model")],
    ),
    # model -> improved model (calibration etc.)
    "model_transform": (
        [_port("input_1", "Input (model)", "model"), _port("input_2", "Input (train_data)", "train_data")],
        [_port("output", "Output (model)", "model", description="Primary output")],
    ),
    # model + test data -> metrics + report
    "evaluate": (
        [_port("input_1", "Input (model)", "model"), _port("input_2", "Input (test_data)", "test_data")],
        [_port("output_1", "Output (metrics)", "metrics", description="Primary output"),
         _port("output_2", "Output (report)", "report", description="Primary output")],
    ),
    # model + metrics -> report (documentation / governance)
    "model_report": (
        [_port("input_1", "Input (model)", "model"), _port("input_2", "Input (metrics)", "metrics", required=False)],
        [_port("output", "Output (report)", "report", description="Primary output")],
    ),
    # table -> metrics (monitoring / tests)
    "monitor": (
        [_DF_IN],
        [_port("output", "Output (metrics)", "metrics", description="Primary output")],
    ),
    # table -> report
    "report": (
        [_DF_IN],
        [_port("output", "Output (report)", "report", description="Primary output")],
    ),
    # model -> deployment record
    "serve": (
        [_port("input", "Input (model)", "model")],
        [_port("output", "Output (governance_record)", "governance_record", description="Primary output")],
    ),
    # canvas-only helper, no data ports
    "canvas": ([], []),
}


# (group, subgroup, category, [(recipe_id, name, port_kind, description), ...])
_PLANNED: List[Tuple[str, str, str, List[Tuple[str, str, str, str]]]] = [
    # ------------------------------------------------------------------ 1. Ingestion & Orchestration (28)
    ("Ingestion & Orchestration", "Load", "ingestion", [
        ("manual_data_entry", "Manual Data Entry", "source", "Type or paste rows directly into a small editable table and use it as pipeline input."),
    ]),
    ("Ingestion & Orchestration", "Output", "ingestion", [
        ("notifications", "Notifications", "sink", "Sends a generic pipeline notification to a configured channel (email, chat or webhook)."),
    ]),
    ("Ingestion & Orchestration", "Database & Warehouse Connectors", "ingestion", [
        ("postgresql_connector", "PostgreSQL Connector", "source", "Reads tables or query results from a PostgreSQL database."),
        ("mysql_connector", "MySQL Connector", "source", "Reads tables or query results from a MySQL / MariaDB database."),
        ("mongodb_connector", "MongoDB Connector", "source", "Reads documents from a MongoDB collection and flattens them into a table."),
        ("snowflake_connector", "Snowflake Connector", "source", "Reads tables or query results from a Snowflake data warehouse."),
        ("bigquery_connector", "BigQuery Connector", "source", "Reads tables or query results from Google BigQuery."),
        ("redshift_connector", "Redshift Connector", "source", "Reads tables or query results from Amazon Redshift."),
    ]),
    ("Ingestion & Orchestration", "Cloud Storage Connectors", "ingestion", [
        ("s3_connector", "S3 Connector", "source", "Loads CSV, Parquet or JSON files from an Amazon S3 (or S3-compatible) bucket."),
        ("gcs_connector", "GCS Connector", "source", "Loads CSV, Parquet or JSON files from a Google Cloud Storage bucket."),
        ("azure_blob_connector", "Azure Blob Connector", "source", "Loads CSV, Parquet or JSON files from Azure Blob Storage."),
    ]),
    ("Ingestion & Orchestration", "Streaming & Change Data", "ingestion", [
        ("kafka_consumer", "Kafka Consumer", "source", "Consumes messages from Kafka topics into micro-batches of rows."),
        ("mqtt_iot_stream", "MQTT / IoT Sensor Stream", "source", "Subscribes to MQTT topics to ingest IoT sensor readings."),
        ("websocket_listener", "WebSocket Listener", "source", "Listens to a WebSocket feed and collects incoming messages as rows."),
        ("cdc_ingestion", "Change Data Capture (CDC) Ingestion", "source", "Captures inserts, updates and deletes from a database change log."),
        ("cdc_replicator", "Database CDC Replicator (real-time sync)", "source", "Continuously replicates database changes into the platform in near real time."),
        ("event_bus_publisher", "Event Bus Publisher (Kafka/RabbitMQ/SNS)", "sink", "Publishes rows or pipeline events to Kafka, RabbitMQ or Amazon SNS."),
    ]),
    ("Ingestion & Orchestration", "File & Web Ingestion", "ingestion", [
        ("ftp_sftp_watcher", "FTP/SFTP File Watcher", "source", "Watches an FTP/SFTP folder and loads new files as they arrive."),
        ("file_system_watcher", "File System Watcher", "source", "Watches a folder on a file system and loads new or changed files."),
        ("email_attachment_extractor", "Email Attachment Extractor", "source", "Reads a mailbox and extracts CSV/Excel attachments as datasets."),
        ("web_scraper", "Web Scraper / HTML Table Extractor", "source", "Extracts HTML tables or page content from a web page into a table."),
        ("rss_atom_aggregator", "RSS/Atom Aggregator (with dedup)", "source", "Aggregates items from multiple RSS/Atom feeds and removes duplicates."),
        ("archive_unpacker", "Zip/Archive Unpacker", "source", "Unpacks ZIP/TAR archives and loads the contained data files."),
    ]),
    ("Ingestion & Orchestration", "Control Flow & Reliability", "flow_control", [
        ("sub_pipeline_call", "Sub-Pipeline Call", "transform", "Runs another saved pipeline as a single step and returns its output."),
        ("retry_error_handler", "Retry / Error-Handling Wrapper", "transform", "Retries a failing step with backoff and routes errors to a fallback path."),
        ("batch_scheduler", "Batch Scheduler with Dependency Graph", "transform", "Schedules batches of jobs that run in order of their dependencies."),
        ("dead_letter_queue", "Dead-Letter Queue Handler", "transform", "Collects rows or events that failed processing for review and replay."),
    ]),

    # ------------------------------------------------------------------ 2. Data Preparation & Transformation (30)
    ("Data Preparation & Transformation", "Combine/Structure", "preprocessing", [
        ("select_columns", "Select (Rename / Reorder Columns)", "transform", "Renames and reorders columns in one step."),
    ]),
    ("Data Preparation & Transformation", "Cleanup & Quality", "preprocessing", [
        ("fuzzy_deduplication", "Fuzzy-Match Deduplication", "transform", "Finds and merges near-duplicate rows using fuzzy string matching."),
        ("semantic_deduplication", "Semantic Deduplication (embedding similarity)", "transform", "Finds duplicates with the same meaning using text-embedding similarity."),
        ("data_profiling_report", "Data Profiling Report", "report", "Produces a profile of every column: types, missing values, distributions and quality issues."),
        ("schema_validation", "Schema / Contract Validation", "transform", "Checks the dataset against an expected schema or data contract and flags violations."),
    ]),
    ("Data Preparation & Transformation", "Custom Logic", "preprocessing", [
        ("guardrails_validation", "Guardrails (validation checks)", "transform", "Applies rule-based data checks (ranges, nulls, uniqueness) and stops or flags on failure."),
        ("formula_calculator", "Formula Calculator", "transform", "Creates or updates columns using spreadsheet-style formulas."),
        ("replace_value", "Replace Value", "transform", "Replaces specific values in selected columns."),
        ("regex_replace_extract", "Regex Replace / Extract", "transform", "Replaces or extracts text in columns using regular expressions."),
        ("parse_date", "Parse Date", "transform", "Parses text into proper dates using a given or auto-detected format."),
    ]),
    ("Data Preparation & Transformation", "Domain-Specific Cleaning", "preprocessing", [
        ("geo_distance_calculator", "Geocode / Reverse Geocode / Distance Calculator", "transform", "Converts addresses to coordinates and back, and calculates distances."),
        ("timezone_converter", "Timezone Converter", "transform", "Converts timestamp columns between time zones."),
        ("business_day_calculator", "Business-Day Calculator / Holiday Calendar", "transform", "Calculates business days and flags holidays using regional calendars."),
        ("currency_unit_normalizer", "Currency / Unit Normalizer", "transform", "Converts amounts to a single currency or unit of measure."),
        ("address_parser", "Address Parser / Standardizer", "transform", "Splits addresses into components and standardizes their format."),
        ("phone_number_validator", "Phone Number Validator / Formatter", "transform", "Validates phone numbers and formats them consistently (E.164)."),
        ("email_domain_validator", "Email / Domain Validator", "transform", "Validates email addresses and domains and flags invalid entries."),
        ("json_xml_flattener", "JSON / XML Flattener & Nester", "transform", "Flattens nested JSON/XML into columns, or nests columns back into JSON."),
        ("hierarchy_flattener", "Hierarchical / Tree Flattener", "transform", "Flattens parent-child hierarchies into level columns."),
        ("crosstab_generator", "Cross-Tab / Contingency Table Generator", "transform", "Builds cross-tabulation (contingency) tables between two or more columns."),
        ("date_range_expander", "Date Range Expander", "transform", "Expands start/end date ranges into one row per day, week or month."),
        ("differential_privacy_anonymizer", "Data Anonymization (Differential Privacy)", "transform", "Adds calibrated noise to protect individual privacy while keeping aggregate accuracy."),
        ("compliance_tokenization", "Compliance Tokenization (PCI/PII masking)", "transform", "Replaces card numbers and personal data with reversible or irreversible tokens."),
        ("sampling_operator", "Sampling Operator (random/stratified/systematic)", "transform", "Takes a random, stratified or systematic sample of rows."),
    ]),
    ("Data Preparation & Transformation", "NLP/Text", "nlp", [
        ("sentiment_analysis", "Sentiment Analysis", "transform", "Scores text columns as positive, negative or neutral."),
        ("text_translation", "Translation", "transform", "Translates text columns into another language."),
        ("text_summarization", "Summarization", "transform", "Generates short summaries of long text columns."),
        ("document_parsing", "Document Parsing", "transform", "Extracts text and tables from PDF, Word and other documents."),
    ]),

    # ------------------------------------------------------------------ 3. Modeling
    ("Modeling", "Deep Learning / AutoML", "training", [
        ("neural_network_trainer", "Neural Network Trainer", "trainer", "Trains a feed-forward neural network for classification or regression."),
        ("automl_search", "AutoML / Hyperparameter Search", "trainer", "Automatically tries multiple algorithms and settings and picks the best model."),
        ("hyperparameter_tuner", "Hyperparameter Tuner (standalone node)", "trainer", "Searches the best hyperparameters for a chosen algorithm (grid, random or Bayesian)."),
        ("ensemble_stacker", "Ensemble Stacker", "trainer", "Combines several models into a stacked ensemble for better accuracy."),
        ("model_calibration", "Model Calibration (Platt / Isotonic)", "model_transform", "Calibrates predicted probabilities so they match real-world frequencies."),
    ]),
    ("Modeling", "Specialized Models", "training", [
        ("survival_model", "Survival / Time-to-Event Model (Cox)", "trainer", "Models time until an event (e.g. churn, failure) with Cox proportional hazards."),
        ("recommendation_engine", "Recommendation Engine (Collaborative / Content-based)", "unsupervised", "Recommends items to users from interaction history or item attributes."),
        ("next_best_offer", "Market Basket / Next-Best-Offer Recommender", "unsupervised", "Suggests the next product or offer for each customer."),
        ("bayesian_ab_estimator", "Bayesian A/B Test Estimator", "monitor", "Estimates the probability that variant B beats A using Bayesian statistics."),
        ("graph_network_analysis", "Graph / Network Analysis", "transform", "Computes network metrics (centrality, communities) from edge lists."),
        ("model_explainability", "Model Explainability (SHAP / LIME)", "evaluate", "Explains which features drive each prediction using SHAP or LIME."),
    ]),
    ("Modeling", "Computer Vision & Audio", "vision_audio", [
        #("object_detection", "Object Detection", "transform", "Detects and locates objects in images with bounding boxes."),
        #("image_embedding_generator", "Image Embedding Generator", "transform", "Converts images into numeric embedding vectors for search or ML."),
        ("speech_to_text", "Speech-to-Text", "transform", "Transcribes audio recordings into text."),
        ("text_to_speech", "Text-to-Speech", "transform", "Converts text into spoken audio files."),
        #("video_object_tracking", "Video Frame Extractor + Object Tracking", "transform", "Extracts frames from video and tracks objects across them."),
    ]),

    # ------------------------------------------------------------------ 4. Evaluation & Governance (10)
    ("Evaluation & Governance", "Fairness & Compliance", "governance", [
        ("bias_fairness_audit", "Bias & Fairness Audit", "evaluate", "Measures model bias across groups using standard fairness metrics."),
        ("fairness_slicing_report", "Fairness Slicing Report (by protected group)", "evaluate", "Breaks down model performance by protected attributes (e.g. gender, age band)."),
        ("audit_log_exporter", "Audit Log / Compliance Trail Exporter", "sink", "Exports a compliance audit trail of pipeline runs, data and model versions."),
        ("model_card_generator", "Model Card Generator", "model_report", "Generates a model card documenting purpose, data, metrics and limitations."),
    ]),

    # ------------------------------------------------------------------ 5. Canvas Tools (10)
    ("Canvas Tools", "Documentation & Layout", "canvas", [
        ("canvas_note", "Note (canvas annotation)", "canvas", "Adds a text note to the canvas to document the pipeline."),
        ("canvas_group", "Group (visual grouping)", "canvas", "Visually groups related nodes on the canvas."),
        ("canvas_comments", "Comment / Annotation Threads", "canvas", "Adds discussion threads to nodes for team collaboration."),
    ]),
    ("Canvas Tools", "Visualization", "canvas", [
        ("chart_visualization", "Chart / Visualization Operator", "report", "Creates charts (bar, line, scatter, histogram) from the incoming data."),
    ]),
    ("Canvas Tools", "Versioning & Reuse", "canvas", [
        ("version_snapshot", "Version Snapshot / Checkpoint & Rollback", "canvas", "Saves a version checkpoint of the pipeline and restores it later."),
        ("template_macro_library", "Reusable Template / Macro Block Library", "canvas", "Saves groups of nodes as reusable blocks and inserts them into other pipelines."),
        ("pipeline_diff_viewer", "Pipeline Diff Viewer", "canvas", "Shows the differences between two versions of a pipeline."),
        ("readonly_pipeline_snapshot", "Shareable Read-Only Pipeline Snapshot", "canvas", "Shares a read-only view of the pipeline via a link."),
    ]),
    ("Canvas Tools", "Configuration", "canvas", [
        ("secrets_manager_block", "Environment Variable / Secrets Manager Block", "canvas", "Stores environment variables and secrets used by nodes, without exposing them."),
        ("parameter_sweep_runner", "Parameter Sweep Runner", "canvas", "Runs the pipeline multiple times over a range of parameter values."),
    ]),

    # ------------------------------------------------------------------ 6. Integrations & Live APIs (35)
    ("Apps & Integrations", "Market & Reference Data", "integrations", [
        ("rss_news_reader", "RSS/News Feed Reader", "source", "Reads items from a single RSS or news feed."),
        ("open_data_api", "Public / Open-Data (Government) API Connector", "source", "Fetches datasets from public and government open-data portals."),
    ]),
    ("Apps & Integrations", "Social & Comms", "integrations", [
        ("twitter_x_connector", "Twitter/X Connector", "source", "Fetches posts, mentions or metrics from Twitter/X."),
        ("reddit_connector", "Reddit Connector", "source", "Fetches posts and comments from subreddits."),
        ("linkedin_connector", "LinkedIn Connector", "source", "Fetches company page posts and engagement metrics from LinkedIn."),
        ("ms_teams_notifier", "Microsoft Teams Notification Sender", "sink", "Posts pipeline alerts and summaries to a Microsoft Teams channel."),
        ("sms_sender", "SMS Sender", "sink", "Sends SMS alerts to one or more phone numbers."),
        ("push_notification_sender", "Push Notification Sender", "sink", "Sends mobile or web push notifications."),
        ("twilio_connector", "Twilio Voice/SMS Connector", "sink", "Sends SMS or places voice calls through Twilio."),
    ]),
    ("Apps & Integrations", "Maps & Location", "integrations", [
        ("google_maps_connector", "Google Maps Connector", "transform", "Enriches rows with geocodes, places, distances and travel times from Google Maps."),
        ("mapbox_connector", "Mapbox Connector", "transform", "Enriches rows with geocodes and routing data from Mapbox."),
    ]),
    ("Apps & Integrations", "CRM / Support / Business", "integrations", [
        ("salesforce_connector", "Salesforce CRM Connector", "source", "Reads Salesforce objects such as accounts, leads and opportunities."),
        ("hubspot_connector", "HubSpot CRM Connector", "source", "Reads HubSpot contacts, companies and deals."),
        ("zendesk_connector", "Zendesk Connector", "source", "Reads Zendesk tickets, users and satisfaction ratings."),
        ("freshdesk_connector", "Freshdesk Connector", "source", "Reads Freshdesk tickets and contacts."),
        ("intercom_connector", "Intercom Connector", "source", "Reads Intercom conversations, users and companies."),
        ("stripe_connector", "Stripe Payments Connector", "source", "Reads Stripe payments, customers and subscriptions."),
        ("shopify_connector", "Shopify E-commerce Connector", "source", "Reads Shopify orders, products and customers."),
        ("google_ads_connector", "Google Ads Connector", "source", "Reads Google Ads campaign performance data."),
        ("meta_ads_connector", "Meta Ads Connector", "source", "Reads Facebook/Instagram (Meta) Ads campaign performance data."),
    ]),
    ("Apps & Integrations", "Project & Dev Tools", "integrations", [
        ("jira_connector", "Jira Connector", "source", "Reads Jira issues, sprints and worklogs."),
        ("linear_connector", "Linear Connector", "source", "Reads Linear issues, projects and cycles."),
        ("asana_connector", "Asana Connector", "source", "Reads Asana tasks and projects."),
        ("github_connector", "GitHub Connector", "source", "Reads GitHub issues, pull requests and repository activity."),
        ("gitlab_connector", "GitLab Connector", "source", "Reads GitLab issues, merge requests and pipeline data."),
    ]),
    ("Apps & Integrations", "Productivity & Files", "integrations", [
        ("google_calendar_connector", "Google Calendar Connector", "source", "Reads or creates Google Calendar events."),
        ("outlook_calendar_connector", "Outlook Calendar Connector", "source", "Reads or creates Outlook calendar events."),
        ("onedrive_sync", "OneDrive Sync", "source", "Reads and writes files in Microsoft OneDrive."),
        ("dropbox_sync", "Dropbox Sync", "source", "Reads and writes files in Dropbox."),
        ("airtable_connector", "Airtable Connector", "source", "Reads and writes Airtable base records."),
        ("notion_connector", "Notion Connector", "source", "Reads and writes Notion databases and pages."),
    ]),
    ("Apps & Integrations", "Generic", "integrations", [
        ("rest_graphql_caller", "Generic REST/GraphQL API Caller", "source", "Calls any REST or GraphQL API and turns the response into a table."),
    ]),

    # ------------------------------------------------------------------ 7. Agents & LLM Orchestration (18)
    ("Agents & LLM Orchestration", "Core Agents", "agents", [
        ("llm_agent", "LLM Agent (ReAct / tool-calling)", "transform", "Runs an LLM agent that reasons step by step and calls tools to complete a task."),
        ("multi_agent_orchestrator", "Multi-Agent Orchestrator (supervisor + workers)", "transform", "Coordinates a supervisor agent and multiple worker agents on a task."),
        ("agent_handoff_router", "Agent-to-Agent Handoff Router", "transform", "Routes a conversation or task between specialised agents."),
        ("web_browsing_agent", "Web-Browsing Agent", "transform", "An agent that searches and reads web pages to answer questions."),
        ("document_qa_agent", "Document Q&A Agent", "transform", "Answers questions over uploaded documents."),
        ("multimodal_agent", "Multi-Modal Agent (image + text)", "transform", "An agent that understands both images and text."),
        ("code_execution_sandbox", "Tool-Use Sandbox (code execution agent)", "transform", "An agent that writes and runs code in an isolated sandbox."),
        ("chatbot_endpoint", "Chatbot Endpoint", "transform", "Exposes a pipeline or agent as a chatbot endpoint."),
    ]),
    ("Agents & LLM Orchestration", "RAG & Retrieval", "agents", [
        ("rag_pipeline", "RAG Retrieval Pipeline", "transform", "Retrieves relevant context from a knowledge base and answers with an LLM."),
        ("vector_db_indexer", "Vector DB Indexer / Query", "transform", "Indexes embeddings into a vector database and queries them."),
        ("embedding_generator", "Embedding Generator", "transform", "Converts text into embedding vectors."),
    ]),
    ("Agents & LLM Orchestration", "Prompt & Memory Management", "agents", [
        ("prompt_template_manager", "Prompt Template Manager", "transform", "Manages reusable, versioned prompt templates with variables."),
        ("conversation_memory", "Conversational Memory / State Store", "transform", "Stores conversation history and state for agents."),
        ("structured_output_enforcer", "Structured-Output Enforcer", "transform", "Forces LLM output into a defined JSON schema and validates it."),
    ]),
    ("Agents & LLM Orchestration", "Governance & Ops", "agents", [
        ("agent_guardrail", "Agent Guardrail / Moderation Layer", "transform", "Filters unsafe inputs and outputs and enforces usage policies."),
        ("cost_aware_model_router", "Cost-Aware Model Router", "transform", "Routes each request to the cheapest model that can handle it."),
        ("fine_tuning_trigger", "Fine-Tuning Trigger", "sink", "Starts a fine-tuning job using the incoming training examples."),
        ("llm_evaluation", "LLM Evaluation (hallucination / groundedness)", "monitor", "Scores LLM answers for hallucination, groundedness and relevance."),
    ]),

    # ------------------------------------------------------------------ 8. Reporting & Delivery (4)
    ("Reporting & Delivery", "Reports & Alerts", "reporting", [
        ("scheduled_report_generator", "Scheduled Report Generator (PDF/Excel snapshot)", "report", "Generates PDF or Excel report snapshots on a schedule."),
        ("dashboard_publisher", "Dashboard Embed / Publish Operator", "sink", "Publishes results to a dashboard or produces an embeddable view."),
        ("threshold_alert_monitor", "Alert / Threshold Monitor", "monitor", "Raises alerts when a metric crosses a configured threshold."),
        ("bi_tool_export", "BI Tool Export (Tableau / Power BI / Looker)", "sink", "Exports results in a format ready for Tableau, Power BI or Looker."),
    ]),

    # ------------------------------------------------------------------ 9. Security & Access (4)
    ("Security & Access", "Access & Audit", "security", [
        ("row_level_access_control", "Row-Level Access Control Operator", "transform", "Restricts which rows each user or role can see."),
        ("audit_trail_logger", "Audit Trail Logger", "sink", "Logs who ran what, when, and on which data."),
        ("credential_rotation_manager", "API Key / Credential Rotation Manager", "canvas", "Rotates API keys and credentials used by pipeline nodes on a schedule."),
        ("encryption_toggle", "Encryption At-Rest / In-Transit Toggle", "canvas", "Controls encryption of pipeline data at rest and in transit."),
    ]),
]


def _build() -> Dict[str, RecipeMetadata]:
    catalog: Dict[str, RecipeMetadata] = {}
    for group, subgroup, category, items in _PLANNED:
        for recipe_id, name, kind, description in items:
            if recipe_id in catalog:
                raise ValueError(f"Duplicate planned recipe_id '{recipe_id}'")
            inputs, outputs = _PORTS[kind]
            catalog[recipe_id] = RecipeMetadata(
                recipe_id=recipe_id,
                name=name,
                version="0.0.0",
                category=category,
                group=group,
                subgroup=subgroup,
                description=description,
                input_types=[p.type for p in inputs],
                output_types=[p.type for p in outputs],
                inputs=[p.model_copy() for p in inputs],
                outputs=[p.model_copy() for p in outputs],
                parameters_schema={"type": "object", "properties": {}},
                enabled=False,
            )
    return catalog


PLANNED_RECIPES: Dict[str, RecipeMetadata] = _build()