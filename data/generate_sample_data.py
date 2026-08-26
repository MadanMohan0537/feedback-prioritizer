"""
Generates a realistic synthetic customer-feedback dataset for a fictional
productivity/SaaS app ("TaskFlow") so the pipeline can be demoed without
depending on a live API or a licensed third-party dataset.

Produces data/sample_reviews.csv with columns: timestamp, text, source, rating

Run: python3 generate_sample_data.py
"""

import csv
import os
import random
from datetime import datetime, timedelta

random.seed(42)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "sample_reviews.csv")

SOURCES = ["app_store", "google_play", "twitter", "support_ticket", "in_app_survey"]

TEMPLATES = {
    "crashes_stability": [
        "The app keeps crashing every time I try to open the dashboard.",
        "TaskFlow crashed twice today and I lost my unsaved notes.",
        "App crashes on launch since the last update, completely unusable.",
        "Constant crashing on my Android phone, please fix this ASAP.",
        "Crashes whenever I try to attach a file to a task.",
        "It froze and then crashed while syncing, very frustrating.",
        "Stability has gotten so much better, no crashes in weeks!",
        "Used to crash constantly, but the latest update fixed it for me.",
    ],
    "billing_payments": [
        "I was billed twice this month for the same subscription.",
        "Overcharged on my credit card, this is unacceptable.",
        "Payment failed but you still charged me, please refund.",
        "Billing is confusing, I can't tell what plan I'm on.",
        "Got double charged after upgrading my plan, need a refund immediately.",
        "The pricing page doesn't match what I was actually billed.",
        "Support fixed my billing issue quickly, thank you.",
        "Cancelled my subscription but still got charged this month.",
    ],
    "performance_speed": [
        "The app is so slow, it takes forever to load my tasks.",
        "Everything lags when I have more than 50 items in a list.",
        "Sync is painfully slow between my phone and laptop.",
        "Performance has improved a lot, loads instantly now.",
        "Really laggy scrolling through the calendar view.",
        "The search feature is slow and often times out.",
        "Loading spinner just spins forever on the reports page.",
        "Snappy and fast after the recent performance update, nice work.",
    ],
    "ui_ux": [
        "The new UI is confusing, I can't find the settings anymore.",
        "Love the redesign, it looks so much cleaner now.",
        "Dark mode please! My eyes hurt using this at night.",
        "The color scheme is hard to read, low contrast text everywhere.",
        "Navigation is unintuitive, took me forever to find my projects.",
        "Really clean and minimal design, easy to use.",
        "Font is too small on mobile, hard to read.",
        "I wish the drag and drop was smoother, it feels clunky.",
    ],
    "feature_requests": [
        "I wish there was a way to export my tasks to PDF.",
        "Please add integration with Google Calendar.",
        "Would be nice to have recurring task templates.",
        "Can you add a dark mode option please?",
        "It would be great if we could assign tasks to multiple people.",
        "Please add support for Slack notifications.",
        "Any plans to add a native mobile widget?",
        "Feature request: bulk editing for tasks would save so much time.",
        "Would love to see a Kanban board view added.",
        "Hope you add offline mode soon, I travel a lot without wifi.",
    ],
    "customer_support": [
        "Support team was super helpful and resolved my issue in minutes.",
        "Waiting three days for a response from support, very disappointed.",
        "The support chat is unresponsive, nobody ever answers.",
        "Great customer service, they walked me through the whole setup.",
        "Still waiting on a reply to my ticket from last week.",
        "Support rep was rude and unhelpful when I asked for a refund.",
        "Quick and friendly support, solved my sync issue right away.",
    ],
    "onboarding": [
        "Onboarding was smooth, got set up in under five minutes.",
        "The tutorial doesn't explain how to invite team members.",
        "Too many steps just to create my first project, simplify it.",
        "Loved the guided walkthrough for new users.",
        "Confusing setup process, I almost gave up before figuring it out.",
    ],
    "data_sync": [
        "My data didn't sync across devices and I lost a week of work.",
        "Sync between desktop and mobile is unreliable, tasks go missing.",
        "Finally sync works flawlessly across all my devices now.",
        "Lost my data after the app crashed during a sync, need this fixed.",
        "Real-time sync is great, changes show up instantly on all my devices.",
    ],
    "general_positive": [
        "This app has completely changed how my team manages projects, love it.",
        "Best productivity app I've used in years, highly recommend.",
        "Simple, reliable, and does exactly what I need.",
        "Five stars, this has saved me so much time every week.",
        "Great value for the price, worth every penny.",
    ],
    "security_login": [
        "Can't log in after resetting my password, keeps saying invalid credentials.",
        "Getting locked out of my account randomly, this is concerning.",
        "Two-factor authentication login is broken on iOS.",
        "Worried about security after seeing this app doesn't support 2FA properly.",
        "Login now works great with Face ID, very convenient.",
    ],
}

SOURCE_PREFIX = {
    "twitter": ["@TaskFlowApp ", "hey @TaskFlowApp ", ""],
    "app_store": ["", ""],
    "google_play": ["", ""],
    "support_ticket": ["Subject: Issue report. ", ""],
    "in_app_survey": ["", ""],
}


def build_dataset(n_rows=650, days_back=14):
    rows = []
    now = datetime.now()
    topics = list(TEMPLATES.keys())

    topic_weights = {
        "crashes_stability": 3.0,
        "billing_payments": 1.6,
        "performance_speed": 1.8,
        "ui_ux": 1.4,
        "feature_requests": 2.2,
        "customer_support": 1.3,
        "onboarding": 1.0,
        "data_sync": 1.5,
        "general_positive": 2.0,
        "security_login": 1.2,
    }

    for i in range(n_rows):
        frac = i / n_rows
        days_ago = days_back * (1 - frac) * random.random()
        ts = now - timedelta(days=days_ago, hours=random.uniform(0, 23), minutes=random.uniform(0, 59))

        if days_ago < 1.0 and random.random() < 0.55:
            topic = "crashes_stability"
        else:
            topic = random.choices(topics, weights=[topic_weights[t] for t in topics])[0]

        text = random.choice(TEMPLATES[topic])
        source = random.choice(SOURCES)
        prefix = random.choice(SOURCE_PREFIX.get(source, [""]))
        text = f"{prefix}{text}"

        negative_topics = {"crashes_stability", "billing_payments", "performance_speed",
                            "data_sync", "security_login"}
        if topic == "general_positive" or "great" in text.lower() or "love" in text.lower():
            rating = random.choice([4, 5, 5])
        elif topic in negative_topics and ("finally" not in text.lower() and "great" not in text.lower()
                                            and "works flawlessly" not in text.lower()
                                            and "improved" not in text.lower()):
            rating = random.choice([1, 1, 2])
        else:
            rating = random.choice([2, 3, 3, 4])

        rows.append({
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "text": text,
            "source": source,
            "rating": rating,
            "true_topic": topic,
        })

    rows.sort(key=lambda r: r["timestamp"])
    return rows


def main():
    rows = build_dataset()
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "text", "source", "rating", "true_topic"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
