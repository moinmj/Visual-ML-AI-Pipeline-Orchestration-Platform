import pytest
import pandas as pd
from backend.app.recipes import recipe_registry
from backend.app.recipes.integrations.twitter_connector import TwitterConnectorRecipe
from backend.app.recipes.integrations.reddit_connector import RedditConnectorRecipe
from backend.app.recipes.integrations.linkedin_connector import LinkedInConnectorRecipe


def test_registry_registration_and_aliases():
    """Verify all 3 connectors and their aliases are properly registered in the recipe registry."""
    # Twitter
    assert recipe_registry.has("twitter_connector")
    assert recipe_registry.has("twitter")
    assert recipe_registry.has("x_connector")
    twitter_rec = recipe_registry.get("twitter_connector")
    assert isinstance(twitter_rec, TwitterConnectorRecipe)

    # Reddit
    assert recipe_registry.has("reddit_connector")
    assert recipe_registry.has("reddit")
    reddit_rec = recipe_registry.get("reddit_connector")
    assert isinstance(reddit_rec, RedditConnectorRecipe)

    # LinkedIn
    assert recipe_registry.has("linkedin_connector")
    assert recipe_registry.has("linkedin")
    linkedin_rec = recipe_registry.get("linkedin_connector")
    assert isinstance(linkedin_rec, LinkedInConnectorRecipe)

    # Metadata & Categories
    tw_meta = recipe_registry.get_metadata("twitter_connector")
    assert tw_meta.group == "Apps & Integrations"
    assert tw_meta.subgroup == "Social & Comms"

    rd_meta = recipe_registry.get_metadata("reddit_connector")
    assert rd_meta.group == "Apps & Integrations"
    assert rd_meta.subgroup == "Social & Comms"

    li_meta = recipe_registry.get_metadata("linkedin_connector")
    assert li_meta.group == "Apps & Integrations"
    assert li_meta.subgroup == "Social & Comms"


def test_twitter_connector_schema_and_execution():
    """Verify TwitterConnectorRecipe schema and execution modes."""
    recipe = TwitterConnectorRecipe()
    schema = recipe.get_schema()
    assert "operation" in schema["properties"]
    assert "query" in schema["properties"]
    assert "bearer_token" in schema["properties"]

    # Verify every property has an input_type
    for prop_name, prop_data in schema["properties"].items():
        assert "input_type" in prop_data, f"Twitter schema missing input_type for {prop_name}"

    # Search Recent Tweets execution
    res = recipe.execute(
        inputs={},
        config={"operation": "Search Recent Tweets", "query": "#MachineLearning", "limit": 15}
    )
    assert "output" in res and "dataframe" in res
    df = res["dataframe"]
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 15
    assert "tweet_id" in df.columns
    assert "text" in df.columns
    assert "like_count" in df.columns
    assert "retweet_count" in df.columns

    # User Timeline Tweets execution
    res_tl = recipe.execute(
        inputs={},
        config={"operation": "User Timeline Tweets", "username": "OpenAI", "limit": 10}
    )
    df_tl = res_tl["dataframe"]
    assert len(df_tl) == 10
    assert (df_tl["author_username"] == "OpenAI").all()

    # Upstream dataset dynamic query enrichment
    upstream_df = pd.DataFrame([{"query": "#DeepLearning"}])
    res_up = recipe.execute(
        inputs={"input": upstream_df},
        config={"operation": "Search Recent Tweets", "limit": 8}
    )
    df_up = res_up["dataframe"]
    assert len(df_up) == 8
    assert "#DeepLearning" in df_up["text"].iloc[0]


def test_reddit_connector_schema_and_execution():
    """Verify RedditConnectorRecipe schema and execution modes."""
    recipe = RedditConnectorRecipe()
    schema = recipe.get_schema()
    assert "operation" in schema["properties"]
    assert "subreddit" in schema["properties"]
    assert "client_id" in schema["properties"]
    assert "client_secret" in schema["properties"]

    # Verify input_type on all fields
    for prop_name, prop_data in schema["properties"].items():
        assert "input_type" in prop_data, f"Reddit schema missing input_type for {prop_name}"

    # Subreddit Hot Posts
    res = recipe.execute(
        inputs={},
        config={"operation": "Subreddit Hot Posts", "subreddit": "MachineLearning", "limit": 12}
    )
    assert "output" in res and "dataframe" in res
    df = res["dataframe"]
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 12
    assert "post_id" in df.columns
    assert "title" in df.columns
    assert "score" in df.columns
    assert "num_comments" in df.columns
    assert (df["subreddit"] == "MachineLearning").all()

    # Subreddit Top Posts with dynamic upstream subreddit
    upstream_df = pd.DataFrame([{"subreddit": "datascience"}])
    res_up = recipe.execute(
        inputs={"input": upstream_df},
        config={"operation": "Subreddit Top Posts", "limit": 6}
    )
    df_up = res_up["dataframe"]
    assert len(df_up) == 6
    assert (df_up["subreddit"] == "datascience").all()


def test_linkedin_connector_schema_and_execution():
    """Verify LinkedInConnectorRecipe schema and execution modes."""
    recipe = LinkedInConnectorRecipe()
    schema = recipe.get_schema()
    assert "operation" in schema["properties"]
    assert "organization_urn" in schema["properties"]
    assert "access_token" in schema["properties"]

    # Verify input_type on all fields
    for prop_name, prop_data in schema["properties"].items():
        assert "input_type" in prop_data, f"LinkedIn schema missing input_type for {prop_name}"

    # Company Page Posts & Updates
    res_posts = recipe.execute(
        inputs={},
        config={"operation": "Company Page Posts & Updates", "organization_urn": "1035", "limit": 10}
    )
    assert "output" in res_posts and "dataframe" in res_posts
    df_posts = res_posts["dataframe"]
    assert isinstance(df_posts, pd.DataFrame)
    assert len(df_posts) == 10
    assert "post_urn" in df_posts.columns
    assert "author_urn" in df_posts.columns
    assert df_posts["author_urn"].iloc[0] == "urn:li:organization:1035"
    assert "like_count" in df_posts.columns
    assert "content_format" in df_posts.columns
    assert "comment_count" in df_posts.columns

    # Company Engagement Metrics & Statistics
    res_stats = recipe.execute(
        inputs={},
        config={"operation": "Company Engagement Metrics & Statistics", "organization_urn": "urn:li:organization:9999", "limit": 7}
    )
    df_stats = res_stats["dataframe"]
    assert len(df_stats) == 7
    assert "impression_count" in df_stats.columns
    assert "click_count" in df_stats.columns
    assert "engagement_rate" in df_stats.columns
