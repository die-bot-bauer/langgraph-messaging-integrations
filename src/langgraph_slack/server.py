 langgraph_slack/server.py

import asyncio
import logging
import re
import json
import uuid
from typing import Awaitable, Callable, TypedDict
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from langgraph_sdk import get_client
from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler
from slack_bolt.async_app import AsyncApp
import httpx

from langgraph_slack import config

# --- Basic Setup ---
LOGGER = logging.getLogger(__name__)
LANGGRAPH_CLIENT = get_client(url=config.LANGGRAPH_URL)
GRAPH_CONFIG = (
    json.loads(config.CONFIG) if isinstance(config.CONFIG, str) else config.CONFIG
)
USER_NAME_CACHE: dict[str, str] = {}
TASK_QUEUE: asyncio.Queue = asyncio.Queue()

# --- Type Definitions ---
class SlackMessageData(TypedDict):
    user: str
    type: str
    subtype: str | None
    ts: str
    thread_ts: str | None
    client_msg_id: str
    text: str
    team: str
    parent_user_id: str
    blocks: list[dict]
    channel: str
    event_ts: str
    channel_type: str

# --- Core Application Logic ---

async def worker():
    LOGGER.info("Background worker started.")
    while True:
        try:
            task = await TASK_QUEUE.get()
            if not task:
                LOGGER.info("Worker received sentinel, exiting.")
                break
            LOGGER.info(f"Worker got a new task: {task}")
            await _process_task(task)
        except Exception as exc:
            LOGGER.exception(f"Error in worker: {exc}")
        finally:
            TASK_QUEUE.task_done()

async def _process_task(task: dict):
    event = task["event"]
    LOGGER.info(f"Processing task for event: {event['ts']}")

    if not (await _is_mention(event) or _is_dm(event)):
        LOGGER.info("Skipping non-mention/non-DM message")
        return

    text_with_names = await _build_contextual_message(event)
    thread_id = _get_thread_id(event.get("thread_ts") or event["ts"], event["channel"])
    
    LOGGER.info(f"Context sent to LangGraph for thread {thread_id}:\n---\n{text_with_names}\n---")
    
    final_state = None
    try:
        async for chunk in LANGGRAPH_CLIENT.runs.stream(
            thread_id=thread_id,
            assistant_id=config.ASSISTANT_ID,
            input={"messages": [{"role": "user", "content": text_with_names}]},
            config=GRAPH_CONFIG,
            stream_mode="values",
        ):
            final_state = chunk
        LOGGER.info("LangGraph stream finished.")
    except Exception as e:
        LOGGER.exception(f"Error calling or streaming from LangGraph: {e}")
        return

    if not final_state:
        LOGGER.error("LangGraph run finished but did not return a final state.")
        return

    messages = final_state.get("messages", [])
    if not messages:
        LOGGER.error("Final state from LangGraph does not contain 'messages'.")
        return
    
    response_message = messages[-1]
    text_to_send = _clean_markdown(_get_text(response_message["content"]))

    webhook_url = config.SLACK_WEBHOOK_URL
    if not webhook_url:
        LOGGER.error("SLACK_WEBHOOK_URL is not configured. Cannot send reply.")
        return

    thread_ts = event.get("thread_ts") or event["ts"]
    payload = {"text": text_to_send, "thread_ts": thread_ts}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(webhook_url, json=payload)
            response.raise_for_status()
        LOGGER.info(f"Successfully sent final reply to Slack thread {thread_ts}")
    except Exception as e:
        LOGGER.exception(f"The final reply to Slack failed: {e}")

# --- Slack Event Handlers ---

async def handle_message(event: SlackMessageData, say: Callable, ack: Callable):
    LOGGER.info("Enqueuing handle_message task...")
    TASK_QUEUE.put_nowait({"type": "slack_message", "event": event})
    await ack()

async def just_ack(ack: Callable[..., Awaitable], event):
    LOGGER.info(f"Acknowledging {event.get('type')} event")
    await ack()

# --- FastAPI and Slack Bolt App Setup ---

APP_HANDLER = AsyncSlackRequestHandler(AsyncApp(logger=LOGGER))
MENTION_REGEX = re.compile(r"<@([A-Z0-9]+)>")
USER_ID_PATTERN = None

APP_HANDLER.app.event("message")(ack=just_ack, lazy=[handle_message])
APP_HANDLER.app.event("app_mention")(ack=just_ack, lazy=[handle_message])

@asynccontextmanager
async def lifespan(app: FastAPI):
    LOGGER.info("App is starting up. Creating background worker...")
    loop = asyncio.get_running_loop()
    loop.create_task(worker())
    yield
    LOGGER.info("App is shutting down. Stopping background worker...")
    TASK_QUEUE.put_nowait(None)

APP = FastAPI(lifespan=lifespan)

@APP.post("/events/slack")
async def slack_endpoint(req: Request):
    return await APP_HANDLER.handle(req)

# --- Helper Functions ---

def _get_text(content: str | list[dict]):
    if isinstance(content, str):
        return content
    return "".join([block["text"] for block in content if block["type"] == "text"])

def _clean_markdown(text: str) -> str:
    text = re.sub(r"^```[^\n]*\n", "```\n", text, flags=re.MULTILINE)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"<\2|\1>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"*\1*", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"_\1_", text)
    text = re.sub(r"_([^_]+)_", r"_\1_", text)
    text = re.sub(r"^\s*[-*]\s", "• ", text, flags=re.MULTILINE)
    return text

async def _is_mention(event: SlackMessageData):
    global USER_ID_PATTERN
    if not config.BOT_USER_ID or config.BOT_USER_ID == "fake-user-id":
        auth_info = await APP_HANDLER.app.client.auth_test()
        config.BOT_USER_ID = auth_info["user_id"]
        LOGGER.info(f"Bot user ID fetched: {config.BOT_USER_ID}")
    
    if USER_ID_PATTERN is None:
        USER_ID_PATTERN = re.compile(rf"<@{config.BOT_USER_ID}>")

    return bool(re.search(USER_ID_PATTERN, event.get("text", "")))

def _get_thread_id(thread_ts: str, channel: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"SLACK:{thread_ts}-{channel}"))

def _is_dm(event: SlackMessageData):
    return event.get("channel_type") == "im"

async def _fetch_thread_history(channel_id: str, thread_ts: str) -> list[SlackMessageData]:
    LOGGER.info(f"Fetching thread history for channel={channel_id}, thread_ts={thread_ts}")
    all_messages = []
    cursor = None
    try:
        while True:
            response = await APP_HANDLER.app.client.conversations_replies(
                channel=channel_id, ts=thread_ts, inclusive=True, limit=200, cursor=cursor
            )
            all_messages.extend(response["messages"])
            if not response.get("has_more"):
                break
            cursor = response["response_metadata"]["next_cursor"]
    except Exception as exc:
        LOGGER.exception(f"Error fetching thread messages: {exc}")
    return all_messages

async def _fetch_user_names(user_ids: set[str]) -> dict[str, str]:
    uncached_ids = [uid for uid in user_ids if uid not in USER_NAME_CACHE]
    if uncached_ids:
        tasks = [APP_HANDLER.app.client.users_info(user=uid) for uid in uncached_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for uid, result in zip(uncached_ids, results):
            if isinstance(result, Exception):
                LOGGER.warning(f"Failed to fetch user info for {uid}: {result}")
                continue
            user_obj = result.get("user", {})
            profile = user_obj.get("profile", {})
            display_name = profile.get("display_name") or profile.get("real_name") or uid
            USER_NAME_CACHE[uid] = display_name
    return {uid: USER_NAME_CACHE[uid] for uid in user_ids if uid in USER_NAME_CACHE}

async def _build_contextual_message(event: SlackMessageData) -> str:
    thread_ts = event.get("thread_ts") or event["ts"]
    channel_id = event["channel"]
    history = await _fetch_thread_history(channel_id, thread_ts)
    
    included = []
    for msg in reversed(history):
        if msg.get("bot_id") == config.BOT_USER_ID:
            break
        included.append(msg)
    included.reverse() 

    all_user_ids = {msg.get("user", "unknown") for msg in included}
    for msg in included:
        all_user_ids.update(MENTION_REGEX.findall(msg.get("text", "")))
    
    all_user_ids.add(event.get("user", "unknown"))
    all_user_ids.update(MENTION_REGEX.findall(event.get("text", "")))

    user_names = await _fetch_user_names(all_user_ids)

    def format_message(msg: SlackMessageData) -> str:
        text = msg.get("text", "")
        user_id = msg.get("user", "unknown")
        def repl(match: re.Match) -> str:
            uid = match.group(1)
            return f"@{user_names.get(uid, uid)}"
        replaced_text = MENTION_REGEX.sub(repl, text)
        speaker_name = user_names.get(user_id, user_id)
        return f'User "{speaker_name}": {replaced_text}'

    contextual_messages = [format_message(msg) for msg in included]
    
    # *** THIS IS THE FIX ***
    # Format and add the new incoming message to the context
    new_message = format_message(event)
    contextual_messages.append(new_message)

    return "\n\n".join(contextual_messages)


# --- Main Execution ---

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("langgraph_slack.server:APP", host="0.0.0.0", port=8080, reload=True)