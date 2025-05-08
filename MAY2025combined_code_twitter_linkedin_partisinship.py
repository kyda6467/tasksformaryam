from dotenv import load_dotenv
from pathlib import Path
import os
import json
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

load_dotenv(dotenv_path=Path('.') / '.env')
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
client.base_url = "https://openrouter.ai/api/v1"

TWITTER_FOLDER = "/Users/kyliedavis/Desktop/Project for Moshen/10-X-timeline"
LINKEDIN_FOLDER = "/Users/kyliedavis/Desktop/Project for Moshen/09-linkedin-timeline"
TWITTER_OUTPUT = "twitter_post_classification.csv"
LINKEDIN_OUTPUT = "linkedin_post_classification.csv"
PARTISAN_OUTPUT = "user_partisanship_results.csv"
MAX_TWEETS = 1000
MAX_USERS = 1000

FORCE_REPROCESSING = False  # Set to True to bypass skip logic for testing

def load_existing_ids(filepath, id_col):
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        return set(df[id_col].astype(str))
    return set()

def classify_post(text, model="gpt-4.1-mini"):
    prompt = f"""
    You are an AI assistant that must decide whether the following post is political or not.

    Rules:
    - Answer either: "political" or "not political"
    - Do not provide explanations or context
    - If ambiguous, lean toward "not political"

    Post:
    "{text}"

    Your response:
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        timeout=30
    )
    return response.choices[0].message.content.strip().lower()

def classify_partisanship(tweets, model="gpt-4.1-mini", include_explanation=False):
    prompt = f"""
    You are an AI assistant classifying political leaning.

    Rules:
    - Respond exactly with: democrat, republican, or unsure
    - {'Include a brief explanation after your answer.' if include_explanation else 'Do not provide explanations.'}

    Tweets:
    {tweets}

    Your response:
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    output = response.choices[0].message.content.strip().lower()
    if include_explanation:
        first_line = output.split('\n')[0].strip()
        label = first_line.split()[0]
        explanation = output[len(first_line):].strip()
    else:
        label = output
        explanation = ""

    if label not in {"democrat", "republican", "unsure"}:
        print(f"⚠️ Unexpected label: '{label}'\nFull response: {output}\n")
        label = "error"

    return label, explanation

def classify_twitter():
    print("🚀 Starting classify_twitter")  # DEBUG
    existing_post_ids = load_existing_ids(TWITTER_OUTPUT, "post_id")
    print("✅ Loaded existing post IDs:", len(existing_post_ids))  # DEBUG
    records = []

    files = [f for f in os.listdir(TWITTER_FOLDER) if f.endswith(".jsonl")][:MAX_USERS]
    print("🗂️ Found files:", files)  # DEBUG

    for file in tqdm(files, desc="Classifying Twitter posts"):
        username = file.replace(".jsonl", "")
        print("📄 Processing user:", username)  # DEBUG

        with open(os.path.join(TWITTER_FOLDER, file), "r") as f:
            for i, line in enumerate(f):
                if len(records) >= MAX_TWEETS:
                    break
                post = json.loads(line)
                post_id = post.get("tweet_id")
                if not FORCE_REPROCESSING and str(post_id) in existing_post_ids:
                    continue
                post_text = post.get("text", "")
                label = classify_post(post_text) if post_text.strip() else "not political"
                records.append({
                    "linktree_id": username,
                    "platform": "twitter",
                    "user_id": username,
                    "post_id": post_id,
                    "is_political": label
                })

        if len(records) >= MAX_TWEETS:
            break

    if records:
        df = pd.DataFrame(records)
        df.to_csv(TWITTER_OUTPUT, mode='a', header=not os.path.exists(TWITTER_OUTPUT), index=False)
        print("✅ Twitter classifications saved.")  # DEBUG
    else:
        print("⚠️ No new Twitter posts classified.")  # DEBUG

def classify_users_partisanship(include_explanation=False):
    print("🚀 Starting classify_users_partisanship")  # DEBUG
    existing_users = load_existing_ids(PARTISAN_OUTPUT, "username")
    print("✅ Loaded existing usernames:", len(existing_users))  # DEBUG
    records = []

    files = [f for f in os.listdir(TWITTER_FOLDER) if f.endswith(".jsonl")][:MAX_USERS]
    print("🗂️ Found files:", files)  # DEBUG

    for file in tqdm(files, desc="Classifying user partisanship"):
        username = file.replace(".jsonl", "")
        print("📄 Checking user:", username)  # DEBUG
        if not FORCE_REPROCESSING and username in existing_users:
            print("⏩ Skipping (already processed).")  # DEBUG
            continue

        tweets = []
        with open(os.path.join(TWITTER_FOLDER, file), "r") as f:
            for line in f:
                data = json.loads(line)
                tweets.append(data.get("text", "").replace("\n", " "))
                if len(tweets) >= 500:
                    break

        tweet_text = "\n".join(tweets)
        if tweets:
            label, explanation = classify_partisanship(tweet_text, include_explanation=include_explanation)
        else:
            label, explanation = "unsure", ""

        print(f"🧾 Result for {username}: {label} | {explanation[:50]}...")  # DEBUG
        records.append({
            "username": username,
            "partisanship": label,
            "explanation": explanation
        })

    if records:
        df = pd.DataFrame(records)
        df.to_csv(PARTISAN_OUTPUT, mode='a', header=not os.path.exists(PARTISAN_OUTPUT), index=False)
        print("✅ Partisanship results saved.")  # DEBUG
    else:
        print("⚠️ No new users classified.")  # DEBUG

if __name__ == "__main__":
    print("🏁 Starting script")  # DEBUG
    classify_twitter()
    classify_users_partisanship(include_explanation=True)
    print("✅ Script finished")  # DEBUG
