import requests
import subprocess
import sounddevice as sd
import wave
import os
import numpy as np
import time
import psutil
import pyautogui
import sqlite3
import threading
import re
import keyboard
import sys
import webbrowser
import pystray
from PIL import Image, ImageDraw
import tkinter as tk
from tkinter import ttk, messagebox

from datetime import datetime, timedelta
from pathlib import Path
from faster_whisper import WhisperModel
from ddgs import DDGS
from flask import Flask, request, jsonify


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\gabeh\OneDrive\Desktop\Orion"

DATABASE_PATH = os.path.join(
    BASE_DIR,
    "atlas_memory.db"
)

PIPER_EXE = os.path.join(
    BASE_DIR,
    "piper",
    "piper.exe"
)

PIPER_VOICE = os.path.join(
    BASE_DIR,
    "piper",
    "voices",
    "en_US-ryan-medium.onnx"
)

CLASSES_DIR = os.path.join(BASE_DIR, "Classes")
os.makedirs(CLASSES_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

AI_MODEL = "llama3.2:3b"

# ============================================================
# APP VERSION / UPDATES
# ============================================================

ATLAS_VERSION = "0.1.5"

# Later, point this to a small JSON file you host online.
# Example JSON:
# {
#   "version": "0.2.0",
#   "download_url": "https://example.com/AtlasSetup.exe",
#   "notes": "Added Exam Mode and dashboard improvements."
# }
UPDATE_MANIFEST_URL = ""

WHISPER_MODEL = "small.en"

SAMPLE_RATE = 16000
BLOCK_SECONDS = 0.25

START_THRESHOLD = 0.04
STOP_THRESHOLD = 0.008

SILENCE_SECONDS = 1.4
MAX_RECORD_SECONDS = 30

CONVERSATION_TIMEOUT = 45

WAKE_PHRASES = [
    "hey atlas",
    "okay atlas",
    "ok atlas",
    "atlas"
]

END_CONVERSATION_PHRASES = [
    "stop listening",
    "end conversation",
    "that's all",
    "that is all",
    "stand by",
    "go to standby",
    "go to sleep"
]


# ============================================================
# PERSONALITY
# ============================================================

SYSTEM_PROMPT = """
You are Atlas, my personal AI assistant.

Your name is Atlas.

Be calm, intelligent, conversational, helpful,
and concise.

You speak responses aloud, so normally answer
in one or two short sentences unless more detail
is specifically requested.

Do not use markdown when speaking.

Do not constantly introduce yourself.

Do not constantly mention that you are an AI.

You have access to:
- persistent memory
- notes
- a to-do list
- a shopping list
- timers
- persistent reminders
- live web search
- approved Windows computer controls
- application launching
- class recording, transcription, detailed class notes, and study guides
- recall and interactive quizzes from saved classes

You can remember the current conversation and
understand natural follow-up questions.

Respond naturally like a personal assistant.
"""


conversation = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT
    }
]


# ============================================================
# LOCKS
# ============================================================

speech_lock = threading.Lock()
timer_lock = threading.Lock()
reminder_lock = threading.Lock()
microphone_lock = threading.Lock()

# Used by the phone API so commands return text without
# making the PC speak every phone response.
atlas_request_context = threading.local()

atlas_muted = False
atlas_shutdown_event = threading.Event()
tray_icon = None
atlas_desktop_root = None
atlas_desktop_thread = None
atlas_manual_talk_event = threading.Event()


# ============================================================
# FIND MICROPHONE
# ============================================================

def find_microphone():
    devices = sd.query_devices()

    for i, device in enumerate(devices):
        name = device["name"]

        if (
            "Microphone Array" in name
            and device["max_input_channels"] > 0
        ):
            print(
                f"Microphone found: {name} "
                f"(Device {i})"
            )
            return i

    default_input = sd.default.device[0]

    if (
        default_input is not None
        and default_input >= 0
    ):
        print(
            f"Using default microphone "
            f"(Device {default_input})"
        )
        return default_input

    raise RuntimeError(
        "Atlas could not find a microphone."
    )


MIC_DEVICE = find_microphone()


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def initialize_database():
    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            note TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS todos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item TEXT NOT NULL,
            completed INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS shopping (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS reminders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reminder TEXT NOT NULL,
            due_at TEXT NOT NULL,
            completed INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        )
        """
    )

    connection.commit()
    connection.close()


initialize_database()


# ============================================================
# GENERIC DATABASE HELPERS
# ============================================================

def database_insert(
    table,
    column,
    value
):
    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    cursor.execute(
        f"""
        INSERT INTO {table}
        ({column}, created_at)
        VALUES (?, ?)
        """,
        (
            value,
            datetime.now().isoformat()
        )
    )

    connection.commit()
    connection.close()


def database_get_all(
    table,
    column
):
    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    cursor.execute(
        f"""
        SELECT id, {column}, created_at
        FROM {table}
        ORDER BY id ASC
        """
    )

    rows = cursor.fetchall()

    connection.close()

    return rows


def delete_matching_item(
    table,
    column,
    query
):
    query = (
        query.lower()
        .strip(" ,.!?")
    )

    rows = database_get_all(
        table,
        column
    )

    exact_matches = []
    partial_matches = []

    for row in rows:
        item_id = row[0]
        text = (
            row[1]
            .lower()
        )

        if text == query:
            exact_matches.append(
                item_id
            )

        elif query in text:
            partial_matches.append(
                item_id
            )

    matches = (
        exact_matches
        if exact_matches
        else partial_matches
    )

    if not matches:
        return 0

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    for item_id in matches:
        cursor.execute(
            f"""
            DELETE FROM {table}
            WHERE id = ?
            """,
            (
                item_id,
            )
        )

    connection.commit()
    connection.close()

    return len(matches)


def clear_table(
    table
):
    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    cursor.execute(
        f"""
        SELECT COUNT(*)
        FROM {table}
        """
    )

    count = cursor.fetchone()[0]

    cursor.execute(
        f"""
        DELETE FROM {table}
        """
    )

    connection.commit()
    connection.close()

    return count


# ============================================================
# PERMANENT MEMORY
# ============================================================

def save_memory(
    memory_text
):
    memory_text = (
        memory_text.strip()
    )

    if not memory_text:
        return False

    database_insert(
        "memories",
        "memory",
        memory_text
    )

    return True


def get_all_memories():
    return database_get_all(
        "memories",
        "memory"
    )


def search_memories(
    query
):
    query = (
        query.lower()
        .strip()
    )

    memories = (
        get_all_memories()
    )

    if not query:
        return memories

    ignored_words = {
        "what",
        "do",
        "you",
        "remember",
        "about",
        "the",
        "a",
        "an",
        "my",
        "i",
        "did",
        "tell"
    }

    query_words = {
        word
        for word in query.split()
        if word not in ignored_words
    }

    matches = []

    for memory in memories:
        text = (
            memory[1]
            .lower()
        )

        score = sum(
            1
            for word in query_words
            if word in text
        )

        if score > 0:
            matches.append(
                (
                    score,
                    memory
                )
            )

    matches.sort(
        key=lambda item:
        item[0],
        reverse=True
    )

    return [
        item[1]
        for item in matches[:10]
    ]


def forget_memory(
    query
):
    return delete_matching_item(
        "memories",
        "memory",
        query
    )


# ============================================================
# NOTES
# ============================================================

def add_note(
    note
):
    note = (
        note.strip(" ,.!?")
    )

    if not note:
        return False

    database_insert(
        "notes",
        "note",
        note
    )

    return True


def get_notes():
    return database_get_all(
        "notes",
        "note"
    )


def delete_note(
    query
):
    return delete_matching_item(
        "notes",
        "note",
        query
    )


# ============================================================
# TO-DO LIST
# ============================================================

def add_todo(
    item
):
    item = (
        item.strip(" ,.!?")
    )

    if not item:
        return False

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT id
        FROM todos
        WHERE LOWER(item) = LOWER(?)
        AND completed = 0
        """,
        (
            item,
        )
    )

    if cursor.fetchone():
        connection.close()
        return False

    cursor.execute(
        """
        INSERT INTO todos
        (
            item,
            completed,
            created_at
        )
        VALUES (?, 0, ?)
        """,
        (
            item,
            datetime.now().isoformat()
        )
    )

    connection.commit()
    connection.close()

    return True


def get_todos(
    include_completed=False
):
    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    if include_completed:
        cursor.execute(
            """
            SELECT
            id,
            item,
            completed,
            created_at
            FROM todos
            ORDER BY id ASC
            """
        )

    else:
        cursor.execute(
            """
            SELECT
            id,
            item,
            completed,
            created_at
            FROM todos
            WHERE completed = 0
            ORDER BY id ASC
            """
        )

    todos = cursor.fetchall()

    connection.close()

    return todos


def complete_todo(
    query
):
    query = (
        query.lower()
        .strip(" ,.!?")
    )

    todos = get_todos()

    matches = []

    for todo in todos:
        if (
            todo[1].lower()
            == query
        ):
            matches = [
                todo[0]
            ]
            break

    if not matches:
        for todo in todos:
            if (
                query
                in todo[1].lower()
            ):
                matches.append(
                    todo[0]
                )

    if not matches:
        return 0

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    for todo_id in matches:
        cursor.execute(
            """
            UPDATE todos
            SET completed = 1
            WHERE id = ?
            """,
            (
                todo_id,
            )
        )

    connection.commit()
    connection.close()

    return len(matches)


def delete_todo(
    query
):
    return delete_matching_item(
        "todos",
        "item",
        query
    )


# ============================================================
# SHOPPING LIST
# ============================================================

def add_shopping_item(
    item
):
    item = (
        item.strip(" ,.!?")
    )

    if not item:
        return False

    current_items = database_get_all(
        "shopping",
        "item"
    )

    for row in current_items:
        if (
            row[1].lower()
            == item.lower()
        ):
            return False

    database_insert(
        "shopping",
        "item",
        item
    )

    return True


def get_shopping_list():
    return database_get_all(
        "shopping",
        "item"
    )


def remove_shopping_item(
    item
):
    return delete_matching_item(
        "shopping",
        "item",
        item
    )


# ============================================================
# SPLIT LIST ITEMS
# ============================================================

def split_list_items(
    text
):
    text = (
        text.strip(" ,.!?")
    )

    text = re.sub(
        r"\s*,\s*",
        " and ",
        text
    )

    pieces = re.split(
        r"\s+and\s+",
        text,
        flags=re.IGNORECASE
    )

    return [
        piece.strip(" ,.!?")
        for piece in pieces
        if piece.strip(" ,.!?")
    ]


# ============================================================
# NOTES / TODO / SHOPPING COMMANDS
# ============================================================

def handle_notes_and_lists(
    command
):
    cleaned = (
        command.lower()
        .strip(" ,.!?")
    )

    note_prefixes = [
        "make a note that ",
        "make a note ",
        "create a note that ",
        "create a note ",
        "write a note that ",
        "write a note ",
        "note that "
    ]

    for prefix in note_prefixes:
        if cleaned.startswith(
            prefix
        ):
            note = (
                command[
                    len(prefix):
                ]
                .strip(" ,.!?")
            )

            if not note:
                speak(
                    "What would you like "
                    "me to write down?"
                )
                return True

            add_note(
                note
            )

            speak(
                "Note saved."
            )

            return True


    if cleaned in [
        "read my notes",
        "what are my notes",
        "what notes do i have",
        "show my notes",
        "list my notes"
    ]:
        notes = get_notes()

        if not notes:
            speak(
                "You don't have "
                "any saved notes."
            )
            return True

        text = "; ".join(
            note[1]
            for note in notes[-10:]
        )

        speak(
            "Your notes are: "
            + text
        )

        return True


    todo_match = re.match(
        r"(?:add|put)\s+(.+?)\s+"
        r"(?:to|on)\s+my\s+"
        r"(?:to do|todo|task)\s+list$",
        cleaned
    )

    if todo_match:
        items = split_list_items(
            todo_match.group(1)
        )

        added = 0

        for item in items:
            if add_todo(
                item
            ):
                added += 1

        if added == 1:
            speak(
                "Added to your "
                "to-do list."
            )

        elif added > 1:
            speak(
                f"Added {added} items "
                f"to your to-do list."
            )

        else:
            speak(
                "Those items are already "
                "on your to-do list."
            )

        return True


    if cleaned in [
        "what's on my to do list",
        "whats on my to do list",
        "what is on my to do list",
        "what's on my todo list",
        "whats on my todo list",
        "read my to do list",
        "read my todo list",
        "show my to do list",
        "show my todo list",
        "what tasks do i have"
    ]:
        todos = get_todos()

        if not todos:
            speak(
                "Your to-do list "
                "is empty."
            )

            return True

        text = "; ".join(
            todo[1]
            for todo in todos
        )

        speak(
            "Your to-do list has: "
            + text
        )

        return True


    shopping_match = re.match(
        r"(?:add|put)\s+(.+?)\s+"
        r"(?:to|on)\s+my\s+"
        r"(?:shopping|grocery)\s+list$",
        cleaned
    )

    if shopping_match:
        items = split_list_items(
            shopping_match.group(1)
        )

        added = 0

        for item in items:
            if add_shopping_item(
                item
            ):
                added += 1

        if added == 1:
            speak(
                "Added to your "
                "shopping list."
            )

        elif added > 1:
            speak(
                f"Added {added} items "
                f"to your shopping list."
            )

        else:
            speak(
                "Those items are already "
                "on your shopping list."
            )

        return True


    if cleaned in [
        "what's on my shopping list",
        "whats on my shopping list",
        "what is on my shopping list",
        "read my shopping list",
        "show my shopping list",
        "what's on my grocery list",
        "whats on my grocery list",
        "read my grocery list",
        "what do i need to buy"
    ]:
        items = get_shopping_list()

        if not items:
            speak(
                "Your shopping list "
                "is empty."
            )

            return True

        text = "; ".join(
            item[1]
            for item in items
        )

        speak(
            "Your shopping list has: "
            + text
        )

        return True

    return False


# ============================================================
# REMINDER DATABASE FUNCTIONS
# ============================================================

def add_reminder(
    reminder_text,
    due_time
):
    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO reminders (
            reminder,
            due_at,
            completed,
            created_at
        )
        VALUES (?, ?, 0, ?)
        """,
        (
            reminder_text,
            due_time.isoformat(),
            datetime.now().isoformat()
        )
    )

    connection.commit()
    connection.close()


def get_active_reminders():
    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            id,
            reminder,
            due_at,
            completed,
            created_at
        FROM reminders
        WHERE completed = 0
        ORDER BY due_at ASC
        """
    )

    rows = cursor.fetchall()

    connection.close()

    return rows


def mark_reminder_complete(
    reminder_id
):
    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        UPDATE reminders
        SET completed = 1
        WHERE id = ?
        """,
        (
            reminder_id,
        )
    )

    connection.commit()
    connection.close()


def cancel_reminder(
    query
):
    query = (
        query.lower()
        .strip(" ,.!?")
    )

    reminders = (
        get_active_reminders()
    )

    matches = []

    for reminder in reminders:
        if query in reminder[1].lower():
            matches.append(
                reminder[0]
            )

    if not matches:
        return 0

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    for reminder_id in matches:
        cursor.execute(
            """
            UPDATE reminders
            SET completed = 1
            WHERE id = ?
            """,
            (
                reminder_id,
            )
        )

    connection.commit()
    connection.close()

    return len(matches)


# ============================================================
# REMINDER TIME PARSING
# ============================================================

def parse_clock_time(
    hour,
    minute,
    ampm
):
    hour = int(
        hour
    )

    minute = int(
        minute or 0
    )

    if ampm:
        ampm = ampm.lower()

        if (
            ampm == "pm"
            and hour != 12
        ):
            hour += 12

        if (
            ampm == "am"
            and hour == 12
        ):
            hour = 0

    return hour, minute


def parse_reminder(
    command
):
    cleaned = (
        command.lower()
        .strip(" ,.!?")
    )

    now = datetime.now()


    # --------------------------------------------------------
    # IN X MINUTES / HOURS
    # --------------------------------------------------------

    relative_match = re.search(
        r"remind me in "
        r"(\d+)\s*"
        r"(minute|minutes|hour|hours|second|seconds)"
        r"\s+to\s+(.+)",
        cleaned
    )

    if relative_match:
        amount = int(
            relative_match.group(1)
        )

        unit = (
            relative_match.group(2)
        )

        reminder_text = (
            relative_match.group(3)
            .strip()
        )

        if unit.startswith(
            "second"
        ):
            due_time = (
                now
                + timedelta(
                    seconds=amount
                )
            )

        elif unit.startswith(
            "minute"
        ):
            due_time = (
                now
                + timedelta(
                    minutes=amount
                )
            )

        else:
            due_time = (
                now
                + timedelta(
                    hours=amount
                )
            )

        return (
            reminder_text,
            due_time
        )


    # --------------------------------------------------------
    # TOMORROW AT 9 AM
    # --------------------------------------------------------

    tomorrow_match = re.search(
        r"remind me tomorrow at "
        r"(\d{1,2})"
        r"(?::(\d{2}))?"
        r"\s*(am|pm)?"
        r"\s+to\s+(.+)",
        cleaned
    )

    if tomorrow_match:
        hour, minute = parse_clock_time(
            tomorrow_match.group(1),
            tomorrow_match.group(2),
            tomorrow_match.group(3)
        )

        reminder_text = (
            tomorrow_match.group(4)
            .strip()
        )

        tomorrow = (
            now.date()
            + timedelta(
                days=1
            )
        )

        due_time = datetime.combine(
            tomorrow,
            datetime.min.time()
        ).replace(
            hour=hour,
            minute=minute
        )

        return (
            reminder_text,
            due_time
        )


    # --------------------------------------------------------
    # AT 7 PM
    # --------------------------------------------------------

    today_match = re.search(
        r"remind me at "
        r"(\d{1,2})"
        r"(?::(\d{2}))?"
        r"\s*(am|pm)?"
        r"\s+to\s+(.+)",
        cleaned
    )

    if today_match:
        hour, minute = parse_clock_time(
            today_match.group(1),
            today_match.group(2),
            today_match.group(3)
        )

        reminder_text = (
            today_match.group(4)
            .strip()
        )

        due_time = now.replace(
            hour=hour,
            minute=minute,
            second=0,
            microsecond=0
        )

        # If that time already passed,
        # interpret it as tomorrow.

        if due_time <= now:
            due_time += timedelta(
                days=1
            )

        return (
            reminder_text,
            due_time
        )


    return None, None


# ============================================================
# REMINDER COMMAND HANDLER
# ============================================================

def handle_reminder_command(
    command
):
    cleaned = (
        command.lower()
        .strip(" ,.!?")
    )

    if cleaned.startswith(
        "remind me "
    ):
        reminder_text, due_time = (
            parse_reminder(
                command
            )
        )

        if (
            reminder_text
            and due_time
        ):
            add_reminder(
                reminder_text,
                due_time
            )

            speak(
                "Okay. I'll remind you "
                + due_time.strftime(
                    "at %I:%M %p."
                )
            )

            return True

        speak(
            "I heard the reminder request, "
            "but I couldn't understand "
            "the time."
        )

        return True


    if cleaned in [
        "what reminders do i have",
        "what are my reminders",
        "list my reminders",
        "show my reminders"
    ]:
        reminders = (
            get_active_reminders()
        )

        if not reminders:
            speak(
                "You don't have any "
                "active reminders."
            )
            return True

        parts = []

        for reminder in reminders[:10]:
            due_time = datetime.fromisoformat(
                reminder[2]
            )

            parts.append(
                f"{reminder[1]} "
                f"at "
                f"{due_time.strftime('%I:%M %p')}"
            )

        speak(
            "Your reminders are: "
            + "; ".join(
                parts
            )
        )

        return True


    cancel_match = re.match(
        r"(?:cancel|delete|remove)\s+"
        r"(?:my\s+|the\s+)?"
        r"(.+?)\s+reminder$",
        cleaned
    )

    if cancel_match:
        query = (
            cancel_match.group(1)
            .strip()
        )

        count = cancel_reminder(
            query
        )

        if count:
            speak(
                "Reminder cancelled."
            )

        else:
            speak(
                "I couldn't find "
                "that reminder."
            )

        return True


    return False


# ============================================================
# REMINDER WATCHER
# ============================================================

def reminder_watcher():
    while True:
        try:
            reminders = (
                get_active_reminders()
            )

            now = datetime.now()

            for reminder in reminders:
                reminder_id = (
                    reminder[0]
                )

                reminder_text = (
                    reminder[1]
                )

                due_time = datetime.fromisoformat(
                    reminder[2]
                )

                if due_time <= now:
                    mark_reminder_complete(
                        reminder_id
                    )

                    speak(
                        f"Reminder. "
                        f"{reminder_text}."
                    )

        except Exception as error:
            print(
                "Reminder error:",
                error
            )

        time.sleep(
            1
        )


reminder_thread = threading.Thread(
    target=reminder_watcher,
    daemon=True
)

reminder_thread.start()


# ============================================================
# SMART APP INDEX
# ============================================================

def build_app_index():
    app_index = {}

    locations = [
        os.path.join(
            os.environ.get(
                "APPDATA",
                ""
            ),
            "Microsoft",
            "Windows",
            "Start Menu",
            "Programs"
        ),

        os.path.join(
            os.environ.get(
                "PROGRAMDATA",
                ""
            ),
            "Microsoft",
            "Windows",
            "Start Menu",
            "Programs"
        )
    ]

    for location in locations:
        if not os.path.exists(
            location
        ):
            continue

        for root, dirs, files in os.walk(
            location
        ):
            for filename in files:
                if filename.lower().endswith(
                    (
                        ".lnk",
                        ".url"
                    )
                ):
                    name = (
                        os.path.splitext(
                            filename
                        )[0]
                    )

                    app_index[
                        name.lower()
                    ] = os.path.join(
                        root,
                        filename
                    )

    print(
        f"Found {len(app_index)} "
        f"Start Menu apps."
    )

    return app_index


APP_INDEX = build_app_index()


def find_app(
    app_name
):
    search = (
        app_name.lower()
        .strip()
    )

    if search in APP_INDEX:
        return (
            search,
            APP_INDEX[search]
        )

    matches = []

    for name, path in APP_INDEX.items():
        if search in name:
            matches.append(
                (
                    name,
                    path
                )
            )

    if matches:
        matches.sort(
            key=lambda x:
            len(x[0])
        )

        return matches[0]

    return None, None


def open_installed_app(
    app_name
):
    name, path = find_app(
        app_name
    )

    if path is None:
        return False

    try:
        os.startfile(
            path
        )

        return True

    except Exception as error:
        print(
            "App launch error:",
            error
        )

        return False


# ============================================================
# WINDOWS BUILT-IN APPS
# ============================================================

def open_builtin_app(
    app
):
    app = (
        app.lower()
        .strip()
    )

    try:
        builtins = {
            "notepad":
            lambda:
            subprocess.Popen(
                ["notepad.exe"]
            ),

            "note pad":
            lambda:
            subprocess.Popen(
                ["notepad.exe"]
            ),

            "settings":
            lambda:
            os.startfile(
                "ms-settings:"
            ),

            "windows settings":
            lambda:
            os.startfile(
                "ms-settings:"
            ),

            "file explorer":
            lambda:
            subprocess.Popen(
                ["explorer.exe"]
            ),

            "explorer":
            lambda:
            subprocess.Popen(
                ["explorer.exe"]
            ),

            "task manager":
            lambda:
            subprocess.Popen(
                ["taskmgr.exe"]
            ),

            "calculator":
            lambda:
            subprocess.Popen(
                ["calc.exe"]
            ),

            "paint":
            lambda:
            subprocess.Popen(
                ["mspaint.exe"]
            ),

            "command prompt":
            lambda:
            subprocess.Popen(
                ["cmd.exe"]
            ),

            "powershell":
            lambda:
            subprocess.Popen(
                ["powershell.exe"]
            ),

            "control panel":
            lambda:
            subprocess.Popen(
                ["control.exe"]
            )
        }

        if app in builtins:
            builtins[app]()
            return True

        return False

    except Exception as error:
        print(
            "Windows app error:",
            error
        )

        return False


# ============================================================
# LOAD WHISPER
# ============================================================

print(
    "\nLoading Atlas..."
)

whisper = WhisperModel(
    WHISPER_MODEL,
    device="cpu",
    compute_type="int8"
)

print(
    "Speech recognition loaded."
)


# ============================================================
# SPEAK
# ============================================================

def speak(
    text
):
    # Phone API requests use the same Atlas command handlers,
    # but their spoken response should be returned to the phone
    # instead of playing through the PC speakers.
    if getattr(
        atlas_request_context,
        "phone_mode",
        False
    ):
        print(
            f"\nAtlas phone response: {text}"
        )

        if not hasattr(
            atlas_request_context,
            "responses"
        ):
            atlas_request_context.responses = []

        atlas_request_context.responses.append(
            str(text)
        )
        return

    if atlas_muted:
        print(
            f"\nAtlas (muted): {text}"
        )
        return

    with speech_lock:
        print(
            f"\nAtlas: {text}"
        )

        output_file = os.path.join(
            BASE_DIR,
            f"atlas_voice_"
            f"{threading.get_ident()}.wav"
        )

        try:
            startupinfo = None
            creationflags = 0

            if os.name == "nt":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
                creationflags = subprocess.CREATE_NO_WINDOW

            process = subprocess.Popen(
                [
                    PIPER_EXE,
                    "--model",
                    PIPER_VOICE,
                    "--output_file",
                    output_file
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                startupinfo=startupinfo,
                creationflags=creationflags
            )

            _, stderr_text = process.communicate(
                input=str(text)
            )

            if process.returncode != 0:
                print(
                    "Piper speech failed:",
                    stderr_text
                )
                return

            if not os.path.isfile(
                output_file
            ):
                print(
                    "Piper did not create the WAV file."
                )
                return

            # Play WAV directly through the Windows multimedia API.
            # This avoids spawning PowerShell from inside Atlas.exe.
            import ctypes

            winmm = ctypes.windll.winmm

            SND_SYNC = 0x0000
            SND_NODEFAULT = 0x0002
            SND_FILENAME = 0x00020000

            result = winmm.PlaySoundW(
                output_file,
                None,
                SND_FILENAME
                | SND_SYNC
                | SND_NODEFAULT
            )

            if result == 0:
                print(
                    "Windows could not play the Atlas WAV file."
                )

        except Exception as error:
            print(
                "Voice error:",
                repr(error)
            )

        finally:
            try:
                if os.path.exists(
                    output_file
                ):
                    os.remove(
                        output_file
                    )
            except Exception:
                pass


# ============================================================
# MICROPHONE RECORDING
# ============================================================

def record_until_silence(abort_event=None):
    frames = []

    recording = False
    silence_start = None
    record_start = None

    blocksize = int(
        SAMPLE_RATE
        * BLOCK_SECONDS
    )

    try:
        # Only one Atlas feature can own the microphone at a time.
        with microphone_lock:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                device=MIC_DEVICE,
                blocksize=blocksize
            ) as stream:

                while True:
                    if (
                        abort_event is not None
                        and abort_event.is_set()
                    ):
                        return None

                    data, overflowed = (
                        stream.read(
                            blocksize
                        )
                    )

                    audio_float = (
                        data.astype(
                            np.float32
                        )
                        / 32768.0
                    )

                    volume = np.max(
                        np.abs(
                            audio_float
                        )
                    )

                    if not recording:
                        if volume > START_THRESHOLD:
                            recording = True
                            record_start = time.time()

                            frames.append(
                                data.copy()
                            )

                    else:
                        frames.append(
                            data.copy()
                        )

                        if volume < STOP_THRESHOLD:
                            if silence_start is None:
                                silence_start = time.time()

                            elif (
                                time.time()
                                - silence_start
                                >= SILENCE_SECONDS
                            ):
                                break

                        else:
                            silence_start = None

                        if (
                            time.time()
                            - record_start
                            >= MAX_RECORD_SECONDS
                        ):
                            break

    except Exception as error:
        print(
            "Microphone error:",
            repr(error)
        )

        time.sleep(0.5)

        return None

    if not frames:
        return None

    return np.concatenate(
        frames,
        axis=0
    )


# ============================================================
# SPEECH TO TEXT
# ============================================================

def transcribe_audio(
    audio
):
    filename = os.path.join(
        BASE_DIR,
        "atlas_input.wav"
    )

    try:
        with wave.open(
            filename,
            "wb"
        ) as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(
                SAMPLE_RATE
            )
            f.writeframes(
                audio.tobytes()
            )

        segments, info = (
            whisper.transcribe(
                filename,
                language="en",
                beam_size=1,
                vad_filter=True,
                vad_parameters={
                    "min_silence_duration_ms":
                    500
                },
                condition_on_previous_text=False,
                initial_prompt=(
                    "The speaker is talking "
                    "to an assistant named Atlas. "
                    "Commands can involve notes, "
                    "shopping lists, to-do lists, "
                    "timers, reminders, applications, "
                    "and normal conversation."
                )
            )
        )

        return " ".join(
            segment.text
            for segment in segments
        ).strip()

    except Exception as error:
        print(
            "Speech recognition error:",
            error
        )

        return ""

    finally:
        try:
            if os.path.exists(
                filename
            ):
                os.remove(
                    filename
                )
        except:
            pass


def listen(
    abort_event=None
):
    audio = (
        record_until_silence(
            abort_event
        )
    )

    if audio is None:
        return ""

    print(
        "Processing speech..."
    )

    return transcribe_audio(
        audio
    )


# ============================================================
# WAKE WORD
# ============================================================

def find_wake_phrase(
    text
):
    text = text.lower()

    for phrase in sorted(
        WAKE_PHRASES,
        key=len,
        reverse=True
    ):
        if phrase in text:
            return phrase

    return None


# ============================================================
# TIMER SYSTEM
# ============================================================

timers = {}
timer_counter = 0


def format_duration(
    seconds
):
    seconds = max(
        0,
        int(round(seconds))
    )

    hours = (
        seconds // 3600
    )

    minutes = (
        seconds % 3600
    ) // 60

    secs = (
        seconds % 60
    )

    parts = []

    if hours:
        parts.append(
            f"{hours} "
            + (
                "hour"
                if hours == 1
                else "hours"
            )
        )

    if minutes:
        parts.append(
            f"{minutes} "
            + (
                "minute"
                if minutes == 1
                else "minutes"
            )
        )

    if secs or not parts:
        parts.append(
            f"{secs} "
            + (
                "second"
                if secs == 1
                else "seconds"
            )
        )

    return " and ".join(
        parts
    )


def parse_duration(
    text
):
    text = text.lower()

    pattern = (
        r"(\d+)\s*"
        r"(seconds?|minutes?|hours?)"
    )

    matches = re.findall(
        pattern,
        text
    )

    if not matches:
        return None

    seconds = 0

    for amount, unit in matches:
        amount = int(
            amount
        )

        if unit.startswith(
            "second"
        ):
            seconds += amount

        elif unit.startswith(
            "minute"
        ):
            seconds += (
                amount * 60
            )

        elif unit.startswith(
            "hour"
        ):
            seconds += (
                amount * 3600
            )

    return (
        seconds
        if seconds > 0
        else None
    )


def timer_finished(
    timer_id
):
    with timer_lock:
        timer = timers.pop(
            timer_id,
            None
        )

    if not timer:
        return

    name = timer["name"]

    if name:
        speak(
            f"Your {name} timer "
            f"is finished."
        )

    else:
        speak(
            "Your timer is finished."
        )


def create_timer(
    duration,
    name=None
):
    global timer_counter

    with timer_lock:
        timer_counter += 1

        timer_id = (
            timer_counter
        )

        timer = threading.Timer(
            duration,
            timer_finished,
            args=(
                timer_id,
            )
        )

        timer.daemon = True

        timers[
            timer_id
        ] = {
            "name":
            name,

            "duration":
            duration,

            "ends":
            time.time()
            + duration,

            "thread":
            timer
        }

        timer.start()


def handle_timer_command(
    command
):
    cleaned = (
        command.lower()
        .strip(" ,.!?")
    )

    if (
        "set" in cleaned
        and "timer" in cleaned
    ):
        duration = (
            parse_duration(
                cleaned
            )
        )

        if duration is None:
            speak(
                "I couldn't understand "
                "the timer duration."
            )
            return True

        match = re.search(
            r"set\s+(?:a\s+)?"
            r"(.+?)\s+timer\s+for",
            cleaned
        )

        name = None

        if match:
            possible_name = (
                match.group(1)
                .strip()
            )

            if possible_name not in [
                "a",
                "the"
            ]:
                name = (
                    possible_name
                )

        create_timer(
            duration,
            name
        )

        if name:
            speak(
                f"Your {name} timer "
                f"is set for "
                f"{format_duration(duration)}."
            )

        else:
            speak(
                f"Timer set for "
                f"{format_duration(duration)}."
            )

        return True

    return False


# ============================================================
# MEMORY COMMAND HANDLER
# ============================================================

def handle_memory_command(
    command
):
    cleaned = (
        command.lower()
        .strip(" ,.!?")
    )

    prefixes = [
        "remember that ",
        "remember "
    ]

    for prefix in prefixes:
        if cleaned.startswith(
            prefix
        ):
            memory = (
                command[
                    len(prefix):
                ]
                .strip(" ,.!?")
            )

            save_memory(
                memory
            )

            speak(
                "I'll remember that."
            )

            return True

    if cleaned in [
        "what do you remember",
        "what do you remember about me",
        "tell me what you remember"
    ]:
        memories = (
            get_all_memories()
        )

        if not memories:
            speak(
                "I don't have any "
                "saved memories."
            )

        else:
            text = "; ".join(
                memory[1]
                for memory
                in memories[-10:]
            )

            speak(
                "I remember: "
                + text
            )

        return True

    if cleaned.startswith(
        "what do you remember about "
    ):
        query = cleaned.replace(
            "what do you remember about ",
            "",
            1
        )

        matches = search_memories(
            query
        )

        if matches:
            speak(
                "I remember: "
                + "; ".join(
                    row[1]
                    for row
                    in matches[:5]
                )
            )

        else:
            speak(
                "I don't have anything "
                "saved about that."
            )

        return True

    return False


# ============================================================
# PC CONTROLS
# ============================================================

def handle_local_command(
    command
):
    cleaned = (
        command.lower()
        .strip(" ,.!?")
    )

    if "what time is it" in cleaned:
        speak(
            datetime.now()
            .strftime(
                "It is %I:%M %p."
            )
        )
        return True

    if "battery" in cleaned:
        battery = (
            psutil.sensors_battery()
        )

        if battery:
            speak(
                f"Your battery is at "
                f"{round(battery.percent)} "
                f"percent."
            )
        return True

    if "cpu" in cleaned:
        speak(
            f"CPU usage is "
            f"{round(psutil.cpu_percent(0.5))} "
            f"percent."
        )
        return True

    if (
        "ram" in cleaned
        or "memory usage" in cleaned
    ):
        speak(
            f"Memory usage is "
            f"{round(psutil.virtual_memory().percent)} "
            f"percent."
        )
        return True

    if "volume up" in cleaned:
        pyautogui.press(
            "volumeup",
            presses=3
        )
        speak(
            "Volume up."
        )
        return True

    if "volume down" in cleaned:
        pyautogui.press(
            "volumedown",
            presses=3
        )
        speak(
            "Volume down."
        )
        return True

    if cleaned == "mute":
        pyautogui.press(
            "volumemute"
        )
        speak(
            "Muted."
        )
        return True

    if cleaned in [
        "pause",
        "play",
        "play music",
        "pause music"
    ]:
        pyautogui.press(
            "playpause"
        )
        speak(
            "Done."
        )
        return True

    if (
        "next song"
        in cleaned
    ):
        pyautogui.press(
            "nexttrack"
        )
        speak(
            "Skipping."
        )
        return True

    if (
        "previous song"
        in cleaned
    ):
        pyautogui.press(
            "prevtrack"
        )
        speak(
            "Going back."
        )
        return True

    open_prefixes = [
        "open ",
        "launch ",
        "start "
    ]

    for prefix in open_prefixes:
        if cleaned.startswith(
            prefix
        ):
            app = (
                cleaned[
                    len(prefix):
                ]
                .strip()
            )

            if open_builtin_app(
                app
            ):
                speak(
                    f"Opening {app}."
                )
                return True

            if open_installed_app(
                app
            ):
                speak(
                    f"Opening {app}."
                )
                return True

            speak(
                f"I couldn't find "
                f"{app}."
            )

            return True

    return False


# ============================================================
# WEB
# ============================================================

def search_web(
    query
):
    print(
        f"Searching web: {query}"
    )

    try:
        results = list(
            DDGS().text(
                query,
                max_results=5
            )
        )

        context = ""

        for result in results:
            context += (
                f"\nTitle: "
                f"{result.get('title','')}"
                f"\nInformation: "
                f"{result.get('body','')}"
            )

        return context

    except Exception as error:
        print(
            "Web error:",
            error
        )

        return ""


def needs_web_search(
    command
):
    text = (
        command.lower()
    )

    live_words = [
        "today",
        "latest",
        "current",
        "right now",
        "weather",
        "forecast",
        "news",
        "score",
        "stock price",
        "search the web",
        "look up",
        "online",
        "recent"
    ]

    return any(
        word in text
        for word
        in live_words
    )


# ============================================================
# ASK OLLAMA
# ============================================================

def ask_atlas(
    message
):
    global conversation

    memory_context = (
        search_memories(
            message
        )
    )

    memory_text = ""

    if memory_context:
        memory_text = (
            "\nRelevant saved memory:\n"
            + "\n".join(
                row[1]
                for row
                in memory_context[:5]
            )
        )

    conversation.append(
        {
            "role":
            "user",

            "content":
            message
            + memory_text
        }
    )

    if len(conversation) > 21:
        conversation = (
            [conversation[0]]
            + conversation[-20:]
        )

    try:
        response = requests.post(
            "http://localhost:11434/api/chat",

            json={
                "model":
                AI_MODEL,

                "messages":
                conversation,

                "stream":
                False
            },

            timeout=120
        )

        response.raise_for_status()

        answer = (
            response.json()
            ["message"]
            ["content"]
        )

        conversation.append(
            {
                "role":
                "assistant",

                "content":
                answer
            }
        )

        return answer

    except Exception as error:
        print(
            "AI error:",
            error
        )

        return (
            "I'm having trouble "
            "connecting to my brain."
        )


def ask_atlas_with_web(
    message
):
    web = search_web(
        message
    )

    if not web:
        return (
            "I couldn't get live "
            "information right now."
        )

    prompt = f"""
The user asked:

{message}

Current search results:

{web}

Answer concisely using the current
search results.

Do not invent information.
Do not read URLs aloud.
"""

    return ask_atlas(
        prompt
    )


# ============================================================
# CLASS RECORDER
# ============================================================

class_recording_active = False
class_recording_stop_event = threading.Event()
class_recording_thread = None
class_recording_folder = None
class_recording_name = None


def ask_class_ai(prompt):
    try:
        response = requests.post("http://localhost:11434/api/chat", json={"model": AI_MODEL, "messages": [{"role": "system", "content": "You are an expert academic note-taking assistant. Create accurate, detailed study materials from lecture transcripts. Never invent information not supported by the lecture."}, {"role": "user", "content": prompt}], "stream": False}, timeout=600)
        response.raise_for_status()
        return response.json()["message"]["content"]
    except Exception as error:
        print("Class AI error:", error)
        return ""


def record_class_audio(wav_path, stop_event):
    blocksize = int(SAMPLE_RATE * 0.5)
    try:
        with wave.open(wav_path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(SAMPLE_RATE)
            with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=blocksize, device=MIC_DEVICE, channels=1, dtype="int16") as stream:
                print("\nCLASS RECORDING ACTIVE - Press Ctrl+Shift+S to stop.")
                while not stop_event.is_set():
                    data, overflowed = stream.read(blocksize)
                    wav_file.writeframes(bytes(data))
    except Exception as error:
        print("Class recording error:", error)


def transcribe_class_audio(wav_path, transcript_path):
    print("\nTranscribing class...")
    segments, info = whisper.transcribe(wav_path, language="en", beam_size=1, vad_filter=True, vad_parameters={"min_silence_duration_ms": 500}, condition_on_previous_text=True)
    transcript = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
    with open(transcript_path, "w", encoding="utf-8") as file:
        file.write(transcript)
    return transcript


def chunk_class_text(text, chunk_size=8000):
    words = text.split()
    chunks, current, size = [], [], 0
    for word in words:
        current.append(word); size += len(word) + 1
        if size >= chunk_size:
            chunks.append(" ".join(current)); current, size = [], 0
    if current:
        chunks.append(" ".join(current))
    return chunks


def create_class_summary(transcript):
    chunks = chunk_class_text(transcript)
    section_notes = []
    print(f"\nProcessing {len(chunks)} class section(s)...")
    for index, chunk in enumerate(chunks, 1):
        print(f"Summarizing section {index} of {len(chunks)}...")
        prompt = f"""Create detailed academic notes from this lecture section. Capture main concepts, supporting details, definitions, vocabulary, names, dates, formulas, numbers, examples, processes, comparisons, cause-and-effect relationships, assignments, questions, instructor emphasis, and likely test material. Preserve useful detail and do not invent anything.\n\nLECTURE SECTION:\n{chunk}"""
        notes = ask_class_ai(prompt)
        if notes:
            section_notes.append(notes)
    if not section_notes:
        return ""
    combined = "\n\n".join(section_notes)
    return ask_class_ai(f"""Combine these consecutive lecture notes into one detailed organized set of class notes. Include lecture overview, main topics, detailed topic sections, definitions, vocabulary, people, dates, numbers, formulas, examples, processes, assignments or deadlines, instructor emphasis, likely exam material, and a concise recap. Remove duplication but preserve detail. Do not invent information.\n\nSECTION NOTES:\n{combined}""")


def create_class_study_guide(summary):
    return ask_class_ai(f"""Create a study guide using only these class notes. Include key concepts, vocabulary and definitions, important facts/names/dates/numbers/formulas, processes, instructor emphasis, 10 review questions with answers, 5 possible test questions with answers, and a final section titled Things I Absolutely Need to Know. Do not invent information.\n\nCLASS NOTES:\n{summary}""")


def process_finished_class(class_folder, class_name):
    wav_path = os.path.join(class_folder, "recording.wav")
    transcript_path = os.path.join(class_folder, "transcript.txt")
    summary_path = os.path.join(class_folder, "summary.txt")
    study_path = os.path.join(class_folder, "study_guide.txt")
    try:
        speak("Recording stopped. I'm processing your class now.")
        transcript = transcribe_class_audio(wav_path, transcript_path)
        if not transcript.strip():
            speak("I saved the recording, but I couldn't detect enough speech to create notes.")
            return
        summary = create_class_summary(transcript)
        if not summary:
            speak("The transcript is saved, but I had trouble generating the summary.")
            return
        with open(summary_path, "w", encoding="utf-8") as file:
            file.write(summary)
        study = create_class_study_guide(summary)
        if study:
            with open(study_path, "w", encoding="utf-8") as file:
                file.write(study)
        print(f"\nClass files saved to:\n{class_folder}")
        speak(f"Your {class_name} class is finished. I saved the recording, transcript, detailed notes, and study guide.")
        try:
            os.startfile(class_folder)
        except Exception:
            pass
    except Exception as error:
        print("Class processing error:", error)
        speak("I saved what I could, but there was an error processing the class.")


def stop_class_recording():
    global class_recording_active, class_recording_thread
    if not class_recording_active:
        return
    print("\nStopping class recording...")
    class_recording_active = False
    class_recording_stop_event.set()
    if class_recording_thread and class_recording_thread.is_alive():
        class_recording_thread.join(timeout=5)
    threading.Thread(target=process_finished_class, args=(class_recording_folder, class_recording_name), daemon=True).start()


def start_class_recording(class_name):
    global class_recording_active, class_recording_thread, class_recording_folder, class_recording_name
    if class_recording_active:
        speak("A class recording is already running.")
        return False
    class_name = class_name.strip(" ,.!?") or "Class"
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", class_name).strip("_") or "Class"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    class_recording_folder = os.path.join(CLASSES_DIR, f"{timestamp}_{safe_name}")
    os.makedirs(class_recording_folder, exist_ok=True)
    wav_path = os.path.join(class_recording_folder, "recording.wav")
    class_recording_name = class_name
    class_recording_stop_event.clear()
    speak(f"Starting {class_name} class recording. Press Control Shift S when class is over.")
    time.sleep(0.3)
    class_recording_active = True
    class_recording_thread = threading.Thread(target=record_class_audio, args=(wav_path, class_recording_stop_event), daemon=True)
    class_recording_thread.start()
    return True


def handle_class_recording_command(command):
    cleaned = command.lower().strip(" ,.!?")
    patterns = [r"start recording my (.+?) class$", r"start recording (.+?) class$", r"record my (.+?) class$", r"record (.+?) class$", r"start class recording for (.+)$"]
    for pattern in patterns:
        match = re.match(pattern, cleaned)
        if match:
            start_class_recording(match.group(1).strip())
            return True
    if cleaned in ["start class recording", "start recording class", "record my class"]:
        start_class_recording("Class")
        return True
    return False


keyboard.add_hotkey("ctrl+shift+s", stop_class_recording)



# ============================================================
# CLASS RECALL / STUDY / QUIZ
# ============================================================

class_quiz_active = False
class_quiz_questions = []
class_quiz_index = 0
class_quiz_source = None


def get_saved_classes():
    classes = []

    if not os.path.exists(CLASSES_DIR):
        print(f"Classes folder not found: {CLASSES_DIR}")
        return classes

    for folder_name in os.listdir(CLASSES_DIR):
        folder_path = os.path.join(
            CLASSES_DIR,
            folder_name
        )

        if not os.path.isdir(folder_path):
            continue

        summary_path = os.path.join(
            folder_path,
            "summary.txt"
        )

        transcript_path = os.path.join(
            folder_path,
            "transcript.txt"
        )

        study_path = os.path.join(
            folder_path,
            "study_guide.txt"
        )

        # Ignore folders that do not contain any usable class text.
        usable_files = []

        for candidate in [
            summary_path,
            study_path,
            transcript_path
        ]:
            try:
                if (
                    os.path.isfile(candidate)
                    and os.path.getsize(candidate) > 10
                ):
                    usable_files.append(candidate)
            except OSError:
                pass

        if not usable_files:
            print(
                f"Skipping class folder with no usable notes: "
                f"{folder_path}"
            )
            continue

        # Folder format:
        # YYYY-MM-DD_HH-MM-SS_Class_Name
        match = re.match(
            r"(\d{4}-\d{2}-\d{2})_"
            r"(\d{2}-\d{2}-\d{2})_"
            r"(.+)",
            folder_name
        )

        if match:
            date_text = match.group(1)
            time_text = match.group(2)
            class_name = (
                match.group(3)
                .replace("_", " ")
            )

            try:
                recorded_at = datetime.strptime(
                    date_text + "_" + time_text,
                    "%Y-%m-%d_%H-%M-%S"
                )
            except ValueError:
                recorded_at = datetime.fromtimestamp(
                    os.path.getmtime(folder_path)
                )
        else:
            class_name = (
                folder_name
                .replace("_", " ")
            )

            recorded_at = datetime.fromtimestamp(
                os.path.getmtime(folder_path)
            )

        classes.append(
            {
                "name": class_name,
                "folder": folder_path,
                "recorded_at": recorded_at,
                "summary": summary_path,
                "transcript": transcript_path,
                "study_guide": study_path,
                "usable_files": usable_files
            }
        )

    classes.sort(
        key=lambda item: item["recorded_at"],
        reverse=True
    )

    print(
        f"Found {len(classes)} usable saved class(es)."
    )

    return classes


def find_saved_class(
    query=""
):
    classes = get_saved_classes()

    if not classes:
        return None

    cleaned = (
        query.lower()
        .strip(" ,.!?")
    )

    today = datetime.now().date()
    target_date = None

    if "yesterday" in cleaned:
        target_date = (
            today
            - timedelta(days=1)
        )

    elif "today" in cleaned:
        target_date = today

    ignored = {
        "what", "did", "we", "cover", "learn",
        "in", "my", "last", "class", "lecture",
        "from", "today", "yesterday", "summarize",
        "summary", "quiz", "me", "on", "about",
        "study", "review", "the", "most", "important",
        "things", "were", "give", "practice",
        "questions"
    }

    query_words = [
        word
        for word in re.findall(
            r"[a-z0-9]+",
            cleaned
        )
        if word not in ignored
    ]

    candidates = classes

    if target_date is not None:
        dated = [
            item
            for item in candidates
            if item["recorded_at"].date()
            == target_date
        ]

        if dated:
            candidates = dated

    if query_words:
        scored = []

        for item in candidates:
            name = item["name"].lower()

            score = sum(
                1
                for word in query_words
                if word in name
            )

            if score:
                scored.append(
                    (score, item)
                )

        if scored:
            scored.sort(
                key=lambda pair: (
                    pair[0],
                    pair[1]["recorded_at"]
                ),
                reverse=True
            )

            return scored[0][1]

    return candidates[0]


def read_class_material(
    class_info,
    prefer_study=False
):
    if prefer_study:
        paths = [
            class_info["study_guide"],
            class_info["summary"],
            class_info["transcript"]
        ]
    else:
        paths = [
            class_info["summary"],
            class_info["study_guide"],
            class_info["transcript"]
        ]

    for path in paths:
        print(
            f"Checking saved class file: {path}"
        )

        if not os.path.isfile(path):
            print(
                "  File does not exist."
            )
            continue

        try:
            # utf-8-sig also handles ordinary UTF-8 and strips a BOM if present.
            with open(
                path,
                "r",
                encoding="utf-8-sig",
                errors="replace"
            ) as file:
                content = file.read().strip()

            print(
                f"  Read {len(content)} characters."
            )

            if content:
                return content

        except Exception as error:
            print(
                "Class file read error:",
                repr(error)
            )

    print(
        "No readable summary, study guide, or transcript "
        f"found in: {class_info['folder']}"
    )

    return ""


def answer_from_saved_class(
    command,
    class_info
):
    material = read_class_material(
        class_info
    )

    if not material:
        return (
            "I found that class, but I couldn't "
            "read its saved notes."
        )

    # Keep local-model context manageable.
    if len(material) > 24000:
        material = material[:24000]

    prompt = f"""
The user is asking about a previously recorded class.

Class name: {class_info["name"]}
Recorded: {class_info["recorded_at"].strftime("%B %d, %Y at %I:%M %p")}

User question:
{command}

Use ONLY the saved class material below.
Do not add outside facts.
If the saved material does not contain the answer,
say that it was not covered in the saved class notes.

Answer naturally and concisely for spoken output,
unless the user asks for detail.

SAVED CLASS MATERIAL:

{material}
"""

    return ask_class_ai(
        prompt
    )


def generate_quiz_questions(
    class_info,
    question_count=10
):
    material = read_class_material(
        class_info,
        prefer_study=True
    )

    if not material:
        return []

    if len(material) > 24000:
        material = material[:24000]

    prompt = f"""
Create exactly {question_count} study questions
using ONLY the saved class material below.

Return them in this exact machine-readable format:

QUESTION: question text
ANSWER: answer text
QUESTION: question text
ANSWER: answer text

Do not number them.
Do not include an introduction or conclusion.
Keep each answer concise but sufficient to judge
whether the student's response is correct.
Do not invent information.

CLASS MATERIAL:

{material}
"""

    raw = ask_class_ai(
        prompt
    )

    questions = []

    pattern = re.compile(
        r"QUESTION:\s*(.*?)\s*"
        r"ANSWER:\s*(.*?)"
        r"(?=\s*QUESTION:|\Z)",
        re.IGNORECASE | re.DOTALL
    )

    for match in pattern.finditer(raw):
        question = match.group(1).strip()
        answer = match.group(2).strip()

        if question and answer:
            questions.append(
                {
                    "question": question,
                    "answer": answer
                }
            )

    return questions[:question_count]


def start_class_quiz(
    command
):
    global class_quiz_active
    global class_quiz_questions
    global class_quiz_index
    global class_quiz_source

    class_info = find_saved_class(
        command
    )

    if not class_info:
        speak(
            "I couldn't find any saved classes yet."
        )
        return True

    speak(
        f"Okay. I'll quiz you on your "
        f"{class_info['name']} class."
    )

    questions = generate_quiz_questions(
        class_info,
        10
    )

    if not questions:
        speak(
            "I found the class, but I couldn't "
            "generate quiz questions from it."
        )
        return True

    class_quiz_active = True
    class_quiz_questions = questions
    class_quiz_index = 0
    class_quiz_source = class_info

    speak(
        f"Question one. "
        f"{questions[0]['question']}"
    )

    return True


def handle_quiz_answer(
    command
):
    global class_quiz_active
    global class_quiz_index

    if not class_quiz_active:
        return False

    cleaned = (
        command.lower()
        .strip(" ,.!?")
    )

    if cleaned in [
        "stop quiz",
        "end quiz",
        "quit quiz",
        "stop the quiz"
    ]:
        class_quiz_active = False
        speak(
            "Okay. Quiz ended."
        )
        return True

    current = class_quiz_questions[
        class_quiz_index
    ]

    prompt = f"""
You are grading one spoken study-quiz answer.

Question:
{current["question"]}

Expected answer based on the class:
{current["answer"]}

Student answer:
{command}

Judge the answer using ONLY the expected answer.
Allow different wording if the meaning is correct.

Reply in no more than two short spoken sentences.
Start with Correct, Mostly correct, or Not quite.
Briefly explain the key point.
Do not ask the next question.
"""

    feedback = ask_class_ai(
        prompt
    )

    speak(
        feedback
    )

    class_quiz_index += 1

    if class_quiz_index >= len(
        class_quiz_questions
    ):
        class_quiz_active = False

        speak(
            "That's the end of the quiz. "
            "Nice work."
        )

        return True

    next_number = (
        class_quiz_index + 1
    )

    speak(
        f"Question {next_number}. "
        f"{class_quiz_questions[class_quiz_index]['question']}"
    )

    return True


def handle_class_recall_command(
    command
):
    cleaned = (
        command.lower()
        .strip(" ,.!?")
    )

    quiz_phrases = [
        "quiz me",
        "test me",
        "practice questions"
    ]

    if any(
        phrase in cleaned
        for phrase in quiz_phrases
    ) and (
        "class" in cleaned
        or "lecture" in cleaned
        or "biology" in cleaned
        or "chemistry" in cleaned
        or "history" in cleaned
        or "math" in cleaned
        or "science" in cleaned
    ):
        return start_class_quiz(
            command
        )

    recall_phrases = [
        "what did we cover",
        "what did we learn",
        "summarize my last class",
        "summarize my class",
        "summarize the class",
        "summary of my class",
        "important things from my last class",
        "important things from class",
        "review my last class",
        "review my class",
        "what was in my last class",
        "what were the important terms"
    ]

    if any(
        phrase in cleaned
        for phrase in recall_phrases
    ):
        class_info = find_saved_class(
            command
        )

        if not class_info:
            speak(
                "I couldn't find any saved classes yet."
            )
            return True

        response = answer_from_saved_class(
            command,
            class_info
        )

        speak(
            response
        )

        return True

    # Natural questions that explicitly mention a saved class.
    if (
        "class" in cleaned
        or "lecture" in cleaned
    ) and any(
        word in cleaned
        for word in [
            "last",
            "today",
            "yesterday",
            "biology",
            "chemistry",
            "history",
            "math",
            "science"
        ]
    ):
        class_info = find_saved_class(
            command
        )

        if class_info:
            response = answer_from_saved_class(
                command,
                class_info
            )

            speak(
                response
            )

            return True

    return False




# ============================================================
# PHONE API - SHARES THE REAL ATLAS BACKEND
# ============================================================

ATLAS_API_HOST = "127.0.0.1"
ATLAS_API_PORT = 5051

atlas_api = Flask(
    "atlas_local_api"
)


def process_phone_command(
    command
):
    """
    Run a phone command through the same handlers/data/state
    used by desktop Atlas.

    Responses sent through speak() are captured and returned
    to the phone instead of being played through the PC.
    """

    global class_quiz_active

    command = (
        command.strip()
    )

    if not command:
        return (
            "I didn't receive a command."
        )

    atlas_request_context.phone_mode = True
    atlas_request_context.responses = []

    try:
        cleaned = (
            command.lower()
            .strip(" ,.!?")
        )

        print(
            f"\nPHONE COMMAND: {command}"
        )

        # Continue an interactive class quiz from the phone.
        if class_quiz_active:
            if handle_quiz_answer(
                command
            ):
                pass

        # Do not let a remote phone command terminate Atlas.
        elif cleaned in [
            "go offline",
            "shutdown atlas",
            "shut down atlas",
            "exit atlas"
        ]:
            speak(
                "For safety, shutdown commands are "
                "disabled from the phone."
            )

        # Recording here would record the PC microphone, not
        # the iPhone microphone, so keep recording on desktop.
        elif (
            "record" in cleaned
            and "class" in cleaned
        ):
            speak(
                "Class recording currently uses the PC "
                "microphone. Start that from desktop Atlas."
            )

        elif handle_class_recall_command(
            command
        ):
            pass

        elif handle_reminder_command(
            command
        ):
            pass

        elif handle_timer_command(
            command
        ):
            pass

        elif handle_notes_and_lists(
            command
        ):
            pass

        elif handle_memory_command(
            command
        ):
            pass

        elif handle_local_command(
            command
        ):
            pass

        else:
            if needs_web_search(
                command
            ):
                response = (
                    ask_atlas_with_web(
                        command
                    )
                )
            else:
                response = (
                    ask_atlas(
                        command
                    )
                )

            speak(
                response
            )

        responses = getattr(
            atlas_request_context,
            "responses",
            []
        )

        if responses:
            return " ".join(
                responses
            )

        return "Done."

    except Exception as error:
        print(
            "Phone command error:",
            error
        )

        return (
            "I ran into an error while "
            "processing that command."
        )

    finally:
        atlas_request_context.phone_mode = False
        atlas_request_context.responses = []


@atlas_api.route(
    "/api/command",
    methods=["POST"]
)
def atlas_api_command():
    data = request.get_json(
        silent=True
    ) or {}

    command = data.get(
        "message",
        ""
    )

    response = process_phone_command(
        command
    )

    return jsonify(
        {
            "response": response
        }
    )


@atlas_api.route(
    "/api/status",
    methods=["GET"]
)
def atlas_api_status():
    return jsonify(
        {
            "online": True,
            "name": "Atlas"
        }
    )


def run_atlas_api():
    print(
        f"\nAtlas phone API listening on "
        f"http://{ATLAS_API_HOST}:"
        f"{ATLAS_API_PORT}"
    )

    atlas_api.run(
        host=ATLAS_API_HOST,
        port=ATLAS_API_PORT,
        debug=False,
        use_reloader=False,
        threaded=True
    )


atlas_api_thread = threading.Thread(
    target=run_atlas_api,
    daemon=True
)

atlas_api_thread.start()





# ============================================================
# UPDATE CHECKING
# ============================================================

def parse_version_number(version_text):
    parts = []

    for piece in str(version_text).strip().lstrip("vV").split("."):
        try:
            parts.append(int(piece))
        except ValueError:
            number = re.match(r"(\d+)", piece)
            parts.append(
                int(number.group(1))
                if number
                else 0
            )

    while len(parts) < 3:
        parts.append(0)

    return tuple(parts[:3])


def check_for_updates():
    """
    Check a hosted JSON manifest for a newer Atlas version.
    Runs in a background thread so the desktop app does not freeze.
    """

    if atlas_desktop_root is None:
        return

    if not UPDATE_MANIFEST_URL.strip():
        atlas_desktop_root.after(
            0,
            lambda: messagebox.showinfo(
                "Atlas Updates",
                (
                    f"You are running Atlas v{ATLAS_VERSION}.\n\n"
                    "Automatic update checking is not configured yet.\n\n"
                    "Once we give Atlas an online update manifest, "
                    "this button will check for new releases automatically."
                )
            )
        )
        return

    def worker():
        try:
            response = requests.get(
                UPDATE_MANIFEST_URL,
                timeout=15
            )

            response.raise_for_status()

            manifest = response.json()

            newest_version = str(
                manifest.get(
                    "version",
                    ""
                )
            ).strip()

            download_url = str(
                manifest.get(
                    "download_url",
                    ""
                )
            ).strip()

            notes = str(
                manifest.get(
                    "notes",
                    ""
                )
            ).strip()

            if not newest_version:
                raise ValueError(
                    "The update manifest did not contain a version."
                )

            current = parse_version_number(
                ATLAS_VERSION
            )

            newest = parse_version_number(
                newest_version
            )

            if newest > current:
                def show_available():
                    details = (
                        f"Atlas v{newest_version} is available.\n\n"
                        f"You currently have v{ATLAS_VERSION}."
                    )

                    if notes:
                        details += (
                            "\n\nWhat's new:\n"
                            + notes
                        )

                    if download_url:
                        details += (
                            "\n\nWould you like to open the download page?"
                        )

                        should_open = messagebox.askyesno(
                            "Atlas Update Available",
                            details
                        )

                        if should_open:
                            webbrowser.open(
                                download_url
                            )
                    else:
                        messagebox.showinfo(
                            "Atlas Update Available",
                            details
                        )

                atlas_desktop_root.after(
                    0,
                    show_available
                )

            else:
                atlas_desktop_root.after(
                    0,
                    lambda: messagebox.showinfo(
                        "Atlas Updates",
                        (
                            f"Atlas is up to date.\n\n"
                            f"Current version: v{ATLAS_VERSION}"
                        )
                    )
                )

        except Exception as error:
            print(
                "Update check error:",
                repr(error)
            )

            atlas_desktop_root.after(
                0,
                lambda: messagebox.showerror(
                    "Atlas Updates",
                    (
                        "Atlas could not check for updates.\n\n"
                        f"{error}"
                    )
                )
            )

    threading.Thread(
        target=worker,
        daemon=True
    ).start()


# ============================================================
# ATLAS DESKTOP APP
# ============================================================

def record_desktop_talk(seconds=6):
    """
    Record a fixed-length desktop utterance.
    This avoids the voice-activity threshold missing quiet speech.
    """
    frames = int(
        SAMPLE_RATE * seconds
    )

    try:
        with microphone_lock:
            audio = sd.rec(
                frames,
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                device=MIC_DEVICE
            )

            sd.wait()

        return audio

    except Exception as error:
        print(
            "Desktop mic recording error:",
            repr(error)
        )
        return None


def desktop_push_to_talk():
    if atlas_desktop_root is None:
        return

    def worker():
        try:
            # Pause the normal wake-word listener so it cannot
            # compete with the desktop Talk button for the mic.
            atlas_manual_talk_event.set()

            atlas_desktop_root.after(
                0,
                lambda: desktop_status_var.set(
                    "LISTENING... SPEAK NOW"
                )
            )

            time.sleep(
                0.7
            )

            audio = record_desktop_talk(
                seconds=6
            )

            if audio is None:
                atlas_desktop_root.after(
                    0,
                    lambda: desktop_add_message(
                        "Atlas",
                        "I couldn't access the microphone."
                    )
                )
                return

            atlas_desktop_root.after(
                0,
                lambda: desktop_status_var.set(
                    "PROCESSING..."
                )
            )

            heard = transcribe_audio(
                audio
            )

            if heard:
                atlas_desktop_root.after(
                    0,
                    lambda h=heard: desktop_run_command(
                        h
                    )
                )
            else:
                atlas_desktop_root.after(
                    0,
                    lambda: desktop_add_message(
                        "Atlas",
                        "I recorded the microphone, but couldn't understand the speech."
                    )
                )

        except Exception as error:
            print(
                "Desktop push-to-talk error:",
                repr(error)
            )

            atlas_desktop_root.after(
                0,
                lambda: desktop_add_message(
                    "Atlas",
                    "There was a microphone error."
                )
            )

        finally:
            atlas_manual_talk_event.clear()

            if atlas_desktop_root is not None:
                atlas_desktop_root.after(
                    250,
                    desktop_refresh_home
                )

    threading.Thread(
        target=worker,
        daemon=True
    ).start()


def desktop_run_command(command):
    command = command.strip()

    if not command:
        return

    def worker():
        response = process_phone_command(command)

        # process_phone_command captures the text response so it can
        # be shown in the desktop chat. Speak it normally afterward.
        if response:
            speak(
                response
            )

        if atlas_desktop_root is not None:
            atlas_desktop_root.after(
                0,
                lambda: desktop_add_message(
                    "Atlas",
                    response
                )
            )

    desktop_add_message(
        "You",
        command
    )

    desktop_command_var.set("")

    threading.Thread(
        target=worker,
        daemon=True
    ).start()


def desktop_add_message(sender, message):
    if "desktop_chat_text" not in globals():
        return

    desktop_chat_text.configure(
        state="normal"
    )

    desktop_chat_text.insert(
        "end",
        f"{sender}: {message}\n\n"
    )

    desktop_chat_text.see(
        "end"
    )

    desktop_chat_text.configure(
        state="disabled"
    )


def desktop_refresh_home():
    if atlas_desktop_root is None:
        return

    try:
        todos = get_todos()
        notes = get_notes()
        shopping = get_shopping_list()
        reminders = get_active_reminders()
        classes = get_saved_classes()

        desktop_stats_var.set(
            f"{len(todos)} tasks   •   "
            f"{len(reminders)} reminders   •   "
            f"{len(classes)} classes   •   "
            f"{len(shopping)} shopping items"
        )

        desktop_status_var.set(
            "ATLAS ONLINE"
            + ("  •  MUTED" if atlas_muted else "  •  VOICE ON")
        )

        desktop_list_text.configure(
            state="normal"
        )
        desktop_list_text.delete(
            "1.0",
            "end"
        )

        if todos:
            desktop_list_text.insert(
                "end",
                "TO-DO\n"
            )
            for todo in todos[:6]:
                desktop_list_text.insert(
                    "end",
                    f"  • {todo[1]}\n"
                )
            desktop_list_text.insert(
                "end",
                "\n"
            )

        if reminders:
            desktop_list_text.insert(
                "end",
                "REMINDERS\n"
            )
            for reminder in reminders[:6]:
                due = datetime.fromisoformat(
                    reminder[2]
                )
                desktop_list_text.insert(
                    "end",
                    f"  • {reminder[1]} — "
                    f"{due.strftime('%b %d, %I:%M %p')}\n"
                )
            desktop_list_text.insert(
                "end",
                "\n"
            )

        if classes:
            desktop_list_text.insert(
                "end",
                "RECENT CLASSES\n"
            )
            for item in classes[:5]:
                desktop_list_text.insert(
                    "end",
                    f"  • {item['name']} — "
                    f"{item['recorded_at'].strftime('%b %d')}\n"
                )

        if not todos and not reminders and not classes:
            desktop_list_text.insert(
                "end",
                "Atlas is ready."
            )

        desktop_list_text.configure(
            state="disabled"
        )

    except Exception as error:
        print(
            "Desktop refresh error:",
            repr(error)
        )


def desktop_toggle_mute():
    global atlas_muted

    atlas_muted = not atlas_muted

    try:
        if tray_icon is not None:
            tray_icon.update_menu()
    except Exception:
        pass

    desktop_refresh_home()


def desktop_show_page(page_name):
    for name, frame in desktop_pages.items():
        if name == page_name:
            frame.tkraise()

    desktop_page_title_var.set(
        page_name
    )

    if page_name == "Home":
        desktop_refresh_home()


def desktop_hide_window():
    if atlas_desktop_root is not None:
        atlas_desktop_root.withdraw()


def show_atlas_desktop():
    if atlas_desktop_root is None:
        return

    atlas_desktop_root.after(
        0,
        lambda: (
            atlas_desktop_root.deiconify(),
            atlas_desktop_root.lift(),
            atlas_desktop_root.focus_force(),
            desktop_refresh_home()
        )
    )


def run_atlas_desktop():
    global atlas_desktop_root
    global desktop_pages
    global desktop_command_var
    global desktop_chat_text
    global desktop_stats_var
    global desktop_status_var
    global desktop_list_text
    global desktop_page_title_var

    root = tk.Tk()
    atlas_desktop_root = root

    root.title(
        "Atlas"
    )
    root.geometry(
        "980x640"
    )
    root.minsize(
        760,
        500
    )

    root.configure(
        bg="#0b1018"
    )

    root.protocol(
        "WM_DELETE_WINDOW",
        desktop_hide_window
    )

    style = ttk.Style()
    try:
        style.theme_use(
            "clam"
        )
    except Exception:
        pass

    style.configure(
        "Atlas.TButton",
        font=("Segoe UI", 11),
        padding=10
    )

    shell = tk.Frame(
        root,
        bg="#0b1018"
    )
    shell.pack(
        fill="both",
        expand=True
    )

    sidebar = tk.Frame(
        shell,
        bg="#111a26",
        width=180
    )
    sidebar.pack(
        side="left",
        fill="y"
    )
    sidebar.pack_propagate(
        False
    )

    brand = tk.Label(
        sidebar,
        text="ATLAS",
        fg="#8bd5ff",
        bg="#111a26",
        font=("Segoe UI Semibold", 24)
    )
    brand.pack(
        anchor="w",
        padx=22,
        pady=(24, 4)
    )

    subtitle = tk.Label(
        sidebar,
        text=f"Personal AI Assistant  •  v{ATLAS_VERSION}",
        fg="#8391a7",
        bg="#111a26",
        font=("Segoe UI", 9)
    )
    subtitle.pack(
        anchor="w",
        padx=23,
        pady=(0, 24)
    )

    content = tk.Frame(
        shell,
        bg="#0b1018"
    )
    content.pack(
        side="left",
        fill="both",
        expand=True
    )

    topbar = tk.Frame(
        content,
        bg="#0b1018",
        height=76
    )
    topbar.pack(
        fill="x"
    )

    desktop_page_title_var = tk.StringVar(
        value="Home"
    )

    page_title = tk.Label(
        topbar,
        textvariable=desktop_page_title_var,
        fg="white",
        bg="#0b1018",
        font=("Segoe UI Semibold", 22)
    )
    page_title.pack(
        side="left",
        padx=28,
        pady=20
    )

    desktop_status_var = tk.StringVar(
        value="ATLAS ONLINE"
    )

    status = tk.Label(
        topbar,
        textvariable=desktop_status_var,
        fg="#8bd5ff",
        bg="#0b1018",
        font=("Segoe UI Semibold", 10)
    )
    status.pack(
        side="right",
        padx=28
    )

    page_host = tk.Frame(
        content,
        bg="#0b1018"
    )
    page_host.pack(
        fill="both",
        expand=True,
        padx=28,
        pady=(0, 28)
    )

    desktop_pages = {}

    for name in [
        "Home",
        "Talk",
        "Classes",
        "Tasks",
        "Shopping",
        "Notes",
        "Settings"
    ]:
        frame = tk.Frame(
            page_host,
            bg="#0b1018"
        )
        frame.place(
            relx=0,
            rely=0,
            relwidth=1,
            relheight=1
        )
        desktop_pages[name] = frame

        button = tk.Button(
            sidebar,
            text=name,
            anchor="w",
            relief="flat",
            bd=0,
            bg="#111a26",
            fg="#d8e2ef",
            activebackground="#1c2a3c",
            activeforeground="white",
            font=("Segoe UI", 11),
            padx=22,
            pady=11,
            command=lambda n=name: desktop_show_page(n)
        )
        button.pack(
            fill="x",
            padx=8,
            pady=2
        )

    # ---------------- HOME ----------------

    home = desktop_pages["Home"]

    hero = tk.Frame(
        home,
        bg="#131d2a",
        highlightbackground="#223249",
        highlightthickness=1
    )
    hero.pack(
        fill="x",
        pady=(0, 18)
    )

    tk.Label(
        hero,
        text="Atlas is ready.",
        fg="white",
        bg="#131d2a",
        font=("Segoe UI Semibold", 25)
    ).pack(
        anchor="w",
        padx=24,
        pady=(22, 5)
    )

    desktop_stats_var = tk.StringVar(
        value="Loading..."
    )

    tk.Label(
        hero,
        textvariable=desktop_stats_var,
        fg="#9aabc0",
        bg="#131d2a",
        font=("Segoe UI", 11)
    ).pack(
        anchor="w",
        padx=24,
        pady=(0, 22)
    )

    quick = tk.Frame(
        home,
        bg="#0b1018"
    )
    quick.pack(
        fill="x",
        pady=(0, 18)
    )

    for label, page in [
        ("Talk to Atlas", "Talk"),
        ("Classes", "Classes"),
        ("Tasks", "Tasks"),
        ("Notes", "Notes")
    ]:
        tk.Button(
            quick,
            text=label,
            command=lambda p=page: desktop_show_page(p),
            bg="#182638",
            fg="white",
            activebackground="#24374e",
            activeforeground="white",
            relief="flat",
            font=("Segoe UI Semibold", 10),
            padx=16,
            pady=12
        ).pack(
            side="left",
            padx=(0, 10)
        )

    desktop_list_text = tk.Text(
        home,
        bg="#111a26",
        fg="#dce7f4",
        insertbackground="white",
        relief="flat",
        font=("Segoe UI", 11),
        padx=20,
        pady=18,
        wrap="word"
    )
    desktop_list_text.pack(
        fill="both",
        expand=True
    )
    desktop_list_text.configure(
        state="disabled"
    )

    # ---------------- TALK ----------------

    talk = desktop_pages["Talk"]

    # Use grid so the chat expands/shrinks while the command bar
    # always stays pinned at the bottom of the Talk page.
    talk.grid_rowconfigure(
        0,
        weight=1
    )

    talk.grid_rowconfigure(
        1,
        weight=0
    )

    talk.grid_columnconfigure(
        0,
        weight=1
    )

    desktop_chat_text = tk.Text(
        talk,
        bg="#111a26",
        fg="#e8eef7",
        insertbackground="white",
        relief="flat",
        font=("Segoe UI", 11),
        padx=18,
        pady=18,
        wrap="word"
    )

    desktop_chat_text.grid(
        row=0,
        column=0,
        sticky="nsew",
        pady=(0, 12)
    )

    desktop_chat_text.configure(
        state="disabled"
    )

    command_bar = tk.Frame(
        talk,
        bg="#0b1018"
    )

    command_bar.grid(
        row=1,
        column=0,
        sticky="ew"
    )

    command_bar.grid_columnconfigure(
        0,
        weight=1
    )

    desktop_command_var = tk.StringVar()

    command_entry = tk.Entry(
        command_bar,
        textvariable=desktop_command_var,
        bg="#182638",
        fg="white",
        insertbackground="white",
        relief="flat",
        font=("Segoe UI", 12)
    )

    command_entry.grid(
        row=0,
        column=0,
        sticky="ew",
        ipady=12
    )

    command_entry.bind(
        "<Return>",
        lambda event: desktop_run_command(
            desktop_command_var.get()
        )
    )

    tk.Button(
        command_bar,
        text="🎙 Talk",
        command=desktop_push_to_talk,
        bg="#1f6f8b",
        fg="white",
        activebackground="#2d8aae",
        activeforeground="white",
        relief="flat",
        font=("Segoe UI Semibold", 11),
        padx=18,
        pady=11
    ).grid(
        row=0,
        column=1,
        padx=(10, 0)
    )

    tk.Button(
        command_bar,
        text="Send",
        command=lambda: desktop_run_command(
            desktop_command_var.get()
        ),
        bg="#2a80b9",
        fg="white",
        activebackground="#3697d3",
        activeforeground="white",
        relief="flat",
        font=("Segoe UI Semibold", 11),
        padx=20,
        pady=11
    ).grid(
        row=0,
        column=2,
        padx=(10, 0)
    )

    # ---------------- DATA PAGES ----------------

    def create_data_page(
        page,
        loader,
        formatter
    ):
        box = tk.Text(
            page,
            bg="#111a26",
            fg="#dce7f4",
            relief="flat",
            font=("Segoe UI", 11),
            padx=20,
            pady=18,
            wrap="word"
        )
        box.pack(
            fill="both",
            expand=True
        )

        def refresh():
            box.configure(
                state="normal"
            )
            box.delete(
                "1.0",
                "end"
            )

            try:
                rows = loader()

                if not rows:
                    box.insert(
                        "end",
                        "Nothing here yet."
                    )
                else:
                    for row in rows:
                        box.insert(
                            "end",
                            formatter(row)
                            + "\n\n"
                        )
            except Exception as error:
                box.insert(
                    "end",
                    f"Could not load this section: {error}"
                )

            box.configure(
                state="disabled"
            )

        tk.Button(
            page,
            text="Refresh",
            command=refresh,
            bg="#182638",
            fg="white",
            activebackground="#24374e",
            activeforeground="white",
            relief="flat",
            padx=16,
            pady=8
        ).pack(
            anchor="e",
            pady=(0, 10)
        )

        refresh()

    create_data_page(
        desktop_pages["Classes"],
        get_saved_classes,
        lambda item:
            f"{item['name']}\n"
            f"{item['recorded_at'].strftime('%B %d, %Y at %I:%M %p')}"
    )

    create_data_page(
        desktop_pages["Tasks"],
        get_todos,
        lambda row:
            f"• {row[1]}"
    )

    create_data_page(
        desktop_pages["Shopping"],
        get_shopping_list,
        lambda row:
            f"• {row[1]}"
    )

    create_data_page(
        desktop_pages["Notes"],
        get_notes,
        lambda row:
            f"• {row[1]}"
    )

    # ---------------- SETTINGS ----------------

    settings = desktop_pages["Settings"]

    tk.Label(
        settings,
        text="Atlas Settings",
        fg="white",
        bg="#0b1018",
        font=("Segoe UI Semibold", 18)
    ).pack(
        anchor="w",
        pady=(0, 18)
    )

    tk.Button(
        settings,
        text="Mute / Unmute Atlas",
        command=desktop_toggle_mute,
        bg="#182638",
        fg="white",
        activebackground="#24374e",
        activeforeground="white",
        relief="flat",
        font=("Segoe UI", 11),
        padx=18,
        pady=11
    ).pack(
        anchor="w",
        pady=5
    )

    tk.Button(
        settings,
        text="Check for Updates",
        command=check_for_updates,
        bg="#182638",
        fg="white",
        activebackground="#24374e",
        activeforeground="white",
        relief="flat",
        font=("Segoe UI", 11),
        padx=18,
        pady=11
    ).pack(
        anchor="w",
        pady=5
    )

    tk.Button(
        settings,
        text="Hide Atlas to System Tray",
        command=desktop_hide_window,
        bg="#182638",
        fg="white",
        activebackground="#24374e",
        activeforeground="white",
        relief="flat",
        font=("Segoe UI", 11),
        padx=18,
        pady=11
    ).pack(
        anchor="w",
        pady=5
    )

    tk.Label(
        settings,
        text=(
            "Closing this window keeps Atlas running. "
            "Use Exit Atlas from the system tray to shut it down."
        ),
        fg="#8e9caf",
        bg="#0b1018",
        font=("Segoe UI", 10),
        wraplength=600,
        justify="left"
    ).pack(
        anchor="w",
        pady=(20, 0)
    )

    desktop_show_page(
        "Home"
    )

    root.mainloop()


def start_atlas_desktop():
    # Tkinter is most reliable when its mainloop runs on the
    # main Python thread. The Atlas listening loop runs in a
    # background thread instead.
    run_atlas_desktop()


# ============================================================
# WINDOWS SYSTEM TRAY
# ============================================================

def create_atlas_tray_image():
    """
    Create a simple built-in Atlas icon so the tray works even
    before a custom atlas.ico file is bundled.
    """
    size = 64
    image = Image.new(
        "RGBA",
        (size, size),
        (0, 0, 0, 0)
    )

    draw = ImageDraw.Draw(image)

    draw.ellipse(
        (4, 4, 60, 60),
        fill=(20, 30, 45, 255),
        outline=(90, 175, 255, 255),
        width=4
    )

    draw.polygon(
        [
            (32, 13),
            (48, 49),
            (40, 49),
            (36, 40),
            (28, 40),
            (24, 49),
            (16, 49)
        ],
        fill=(120, 200, 255, 255)
    )

    return image


def open_atlas_dashboard(
    icon=None,
    item=None
):
    if atlas_desktop_root is not None:
        show_atlas_desktop()


def tray_mute_label(
    item
):
    if atlas_muted:
        return "Unmute Atlas"

    return "Mute Atlas"


def toggle_atlas_mute(
    icon,
    item
):
    global atlas_muted

    atlas_muted = not atlas_muted

    print(
        "Atlas muted."
        if atlas_muted
        else "Atlas unmuted."
    )

    try:
        icon.update_menu()
    except Exception:
        pass


def restart_atlas(
    icon,
    item
):
    """
    Start a fresh Atlas process, then close this one.
    Works for both the PyInstaller EXE and normal Python runs.
    """
    try:
        if getattr(
            sys,
            "frozen",
            False
        ):
            subprocess.Popen(
                [sys.executable],
                cwd=os.path.dirname(
                    sys.executable
                )
            )
        else:
            subprocess.Popen(
                [
                    sys.executable,
                    os.path.abspath(
                        __file__
                    )
                ],
                cwd=BASE_DIR
            )

    except Exception as error:
        print(
            "Atlas restart error:",
            repr(error)
        )
        return

    exit_atlas(
        icon,
        item
    )


def exit_atlas(
    icon=None,
    item=None
):
    global class_recording_active

    print(
        "Shutting down Atlas..."
    )

    atlas_shutdown_event.set()

    if class_recording_active:
        try:
            stop_class_recording()
        except Exception:
            pass

    try:
        keyboard.unhook_all_hotkeys()
    except Exception:
        pass

    if icon is not None:
        try:
            icon.stop()
        except Exception:
            pass

    # os._exit is intentional here. Atlas has daemon threads,
    # microphone streams, Flask, and keyboard hooks that can keep
    # a windowed PyInstaller process alive after the main loop ends.
    os._exit(0)


def run_atlas_tray():
    global tray_icon

    menu = pystray.Menu(
        pystray.MenuItem(
            "Open Atlas",
            open_atlas_dashboard,
            default=True
        ),
        pystray.MenuItem(
            tray_mute_label,
            toggle_atlas_mute
        ),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem(
            "Restart Atlas",
            restart_atlas
        ),
        pystray.MenuItem(
            "Exit Atlas",
            exit_atlas
        )
    )

    tray_icon = pystray.Icon(
        "Atlas",
        create_atlas_tray_image(),
        "Atlas",
        menu
    )

    tray_icon.run()


tray_thread = threading.Thread(
    target=run_atlas_tray,
    daemon=True
)

tray_thread.start()


# ============================================================
# START ATLAS
# ============================================================

print(
    "\nStarting Atlas..."
)

speak(
    "Atlas online."
)

print(
    "\nATLAS IS ONLINE"
)

print(
    "Say Hey Atlas when you need me."
)


# ============================================================
# MAIN LOOP
# ============================================================

def atlas_voice_loop():
    global conversation_mode
    global last_activity_time

    conversation_mode = False
    last_activity_time = None

    while not atlas_shutdown_event.is_set():

        # --------------------------------------------------------
        # STANDBY
        # --------------------------------------------------------

        if not conversation_mode:
            if atlas_manual_talk_event.is_set():
                time.sleep(0.1)
                continue

            print(
                "\nWaiting for wake word..."
            )

            heard = listen(
                atlas_manual_talk_event
            )

            if atlas_manual_talk_event.is_set():
                continue

            if not heard:
                continue

            print(
                f"Heard: {heard}"
            )

            wake_phrase = (
                find_wake_phrase(
                    heard
                )
            )

            if wake_phrase is None:
                continue

            position = (
                heard.lower()
                .find(
                    wake_phrase
                )
            )

            command = (
                heard[
                    position
                    + len(wake_phrase):
                ]
                .strip(
                    " ,.!?-"
                )
            )

            conversation_mode = True

            last_activity_time = (
                time.time()
            )

            if not command:
                speak(
                    "Yes?"
                )
                continue


        # --------------------------------------------------------
        # CONVERSATION MODE
        # --------------------------------------------------------

        else:
            if atlas_manual_talk_event.is_set():
                time.sleep(0.1)
                continue

            if (
                time.time()
                - last_activity_time
                >= CONVERSATION_TIMEOUT
            ):
                speak(
                    "Standing by."
                )

                conversation_mode = False

                continue

            print(
                "\nListening..."
            )

            command = listen(
                atlas_manual_talk_event
            )

            if atlas_manual_talk_event.is_set():
                continue

            if not command:
                continue

            last_activity_time = (
                time.time()
            )


        print(
            f"\nYou: {command}"
        )

        cleaned = (
            command.lower()
            .strip(" ,.!?")
        )


        # --------------------------------------------------------
        # ACTIVE CLASS QUIZ
        # --------------------------------------------------------

        if class_quiz_active:
            if handle_quiz_answer(
                command
            ):
                continue


        # --------------------------------------------------------
        # END CONVERSATION
        # --------------------------------------------------------

        if any(
            phrase in cleaned
            for phrase
            in END_CONVERSATION_PHRASES
        ):
            speak(
                "Standing by."
            )

            conversation_mode = False

            continue


        # --------------------------------------------------------
        # FULL SHUTDOWN
        # --------------------------------------------------------

        if cleaned in [
            "go offline",
            "shutdown atlas",
            "shut down atlas",
            "exit atlas"
        ]:
            speak(
                "Going offline."
            )

            break


        # --------------------------------------------------------
        # CLASS RECORDING
        # --------------------------------------------------------

        if handle_class_recording_command(
            command
        ):
            conversation_mode = False
            while class_recording_active:
                time.sleep(0.5)
            continue


        # --------------------------------------------------------
        # CLASS RECALL / STUDY / QUIZ
        # --------------------------------------------------------

        if handle_class_recall_command(
            command
        ):
            continue


        # --------------------------------------------------------
        # REMINDERS
        # --------------------------------------------------------

        if handle_reminder_command(
            command
        ):
            continue


        # --------------------------------------------------------
        # TIMERS
        # --------------------------------------------------------

        if handle_timer_command(
            command
        ):
            continue


        # --------------------------------------------------------
        # NOTES / TODO / SHOPPING
        # --------------------------------------------------------

        if handle_notes_and_lists(
            command
        ):
            continue


        # --------------------------------------------------------
        # MEMORY
        # --------------------------------------------------------

        if handle_memory_command(
            command
        ):
            continue


        # --------------------------------------------------------
        # COMPUTER
        # --------------------------------------------------------

        if handle_local_command(
            command
        ):
            continue


        # --------------------------------------------------------
        # AI / WEB
        # --------------------------------------------------------

        if needs_web_search(
            command
        ):
            print(
                "\nSearching live information..."
            )

            response = (
                ask_atlas_with_web(
                    command
                )
            )

        else:
            print(
                "\nThinking..."
            )

            response = (
                ask_atlas(
                    command
                )
            )

        speak(
            response
        )

        last_activity_time = (
            time.time()
        )

# ============================================================
# RUN VOICE + DESKTOP
# ============================================================

voice_thread = threading.Thread(
    target=atlas_voice_loop,
    daemon=True
)

voice_thread.start()

# Keep the desktop UI on the main thread.
start_atlas_desktop()
