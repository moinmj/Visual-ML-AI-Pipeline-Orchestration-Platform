#!/bin/bash

set -euo pipefail

EVENT="${1:-}"

if [[ -z "$EVENT" ]]; then
    echo "Usage: ./notify.sh started|success|failed"
    exit 1
fi

case "$EVENT" in
    started)
        THEME_COLOR="0078D7"
        SUMMARY="ML Pipeline Deployment Started"
        ACTIVITY_TITLE="🚀 ML Pipeline Deployment Started"
        ;;
    success)
        THEME_COLOR="00C851"
        SUMMARY="ML Pipeline Deployment Successful"
        ACTIVITY_TITLE="✅ ML Pipeline Deployment Successful"
        ;;
    failed)
        THEME_COLOR="FF4444"
        SUMMARY="ML Pipeline Deployment Failed"
        ACTIVITY_TITLE="❌ ML Pipeline Deployment Failed"
        ;;
    *)
        echo "❌ Invalid event: $EVENT"
        exit 1
        ;;
esac

if [[ -z "${TEAMS_WEBHOOK:-}" ]]; then
    echo "❌ TEAMS_WEBHOOK is not set."
    exit 1
fi

APPLICATION="ML-Pipeline-Platform"
ENVIRONMENT="Development"

PIPELINE_NAME="${BUILD_DEFINITIONNAME:-N/A}"
BUILD_NUMBER="${BUILD_BUILDNUMBER:-N/A}"
BRANCH="${BUILD_SOURCEBRANCHNAME:-N/A}"
COMMIT_AUTHOR="${BUILD_SOURCEVERSIONAUTHOR:-N/A}"
COMMIT_ID="${BUILD_SOURCEVERSION:-N/A}"
COMMIT_MSG="${BUILD_SOURCEVERSIONMESSAGE:-N/A}"

COLLECTION_URI="${SYSTEM_COLLECTIONURI:-}"
TEAM_PROJECT="${SYSTEM_TEAMPROJECT:-}"
BUILD_ID="${BUILD_BUILDID:-}"

BUILD_URL="${COLLECTION_URI}${TEAM_PROJECT}/_build/results?buildId=${BUILD_ID}"

if ! command -v jq >/dev/null 2>&1; then
    echo "❌ jq is not installed."
    exit 1
fi

PAYLOAD=$(jq -n \
    --arg theme "$THEME_COLOR" \
    --arg summary "$SUMMARY" \
    --arg title "$ACTIVITY_TITLE" \
    --arg application "$APPLICATION" \
    --arg pipeline "$PIPELINE_NAME" \
    --arg build "$BUILD_NUMBER" \
    --arg branch "$BRANCH" \
    --arg author "$COMMIT_AUTHOR" \
    --arg commit "$COMMIT_MSG" \
    --arg commit_id "$COMMIT_ID" \
    --arg environment "$ENVIRONMENT" \
    --arg build_url "$BUILD_URL" \
'
{
    "@type": "MessageCard",
    "@context": "https://schema.org/extensions",
    "themeColor": $theme,
    "summary": $summary,
    "sections": [
        {
            "activityTitle": $title,
            "facts": [
                {"name": "Application", "value": $application},
                {"name": "Pipeline", "value": $pipeline},
                {"name": "Build", "value": $build},
                {"name": "Branch", "value": $branch},
                {"name": "Commit Author", "value": $author},
                {"name": "Commit", "value": $commit},
                {"name": "Commit ID", "value": $commit_id},
                {"name": "Environment", "value": $environment}
            ]
        }
    ],
    "potentialAction": [
        {
            "@type": "OpenUri",
            "name": "View Build",
            "targets": [
                {
                    "os": "default",
                    "uri": $build_url
                }
            ]
        }
    ]
}
')

echo "📢 Sending ${EVENT} notification..."

HTTP_STATUS=$(curl \
    --silent \
    --show-error \
    --output /tmp/teams_response.txt \
    --write-out "%{http_code}" \
    --request POST \
    --header "Content-Type: application/json" \
    --data "$PAYLOAD" \
    "$TEAMS_WEBHOOK")

if [[ "$HTTP_STATUS" =~ ^2[0-9][0-9]$ ]]; then
    echo "✅ ${EVENT} notification sent successfully."
else
    echo "❌ Failed to send ${EVENT} notification."
    echo "HTTP Status: ${HTTP_STATUS}"

    [[ -s /tmp/teams_response.txt ]] && cat /tmp/teams_response.txt

    exit 1
fi

rm -f /tmp/teams_response.txt

echo "✅ Notification completed."
