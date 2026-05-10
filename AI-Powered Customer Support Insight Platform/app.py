import pandas as pd

# -----------------------------
# AI-style rule-based + smart scoring
# -----------------------------

def analyze_sentiment(text):
    text = str(text).lower()

    score = 0

    negative_keywords = [
        "refund", "late", "failed", "damaged",
        "poor", "not received", "cancel", "issue",
        "problem", "angry", "worst", "slow"
    ]

    positive_keywords = [
        "good", "fast", "satisfied", "nice", "great"
    ]

    for word in negative_keywords:
        if word in text:
            score -= 1

    for word in positive_keywords:
        if word in text:
            score += 1

    if score < 0:
        return "NEGATIVE"
    elif score > 0:
        return "POSITIVE"
    else:
        return "NEUTRAL"


# -----------------------------
# AI Category Detection
# -----------------------------

def detect_category(text):
    text = str(text).lower()

    categories = {
        "Refund Issue": ["refund", "money back"],
        "Delivery Issue": ["delivery", "late", "shipping"],
        "Payment Issue": ["payment", "failed", "checkout"],
        "Login Issue": ["login", "password", "account"],
        "Product Issue": ["damaged", "broken", "defective"]
    }

    for cat, keywords in categories.items():
        for word in keywords:
            if word in text:
                return cat

    return "General Issue"


# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv("tickets.csv")

# -----------------------------
# AI Processing
# -----------------------------

df["sentiment"] = df["message"].apply(analyze_sentiment)
df["category"] = df["message"].apply(detect_category)

# High risk logic
df["risk_score"] = df["order_value"] * df["sentiment"].apply(
    lambda x: 1 if x == "NEGATIVE" else 0
)

# -----------------------------
# OUTPUT INSIGHTS
# -----------------------------

print("\n==============================")
print("AI CUSTOMER SUPPORT INSIGHTS")
print("==============================\n")

print("Total Tickets:", len(df))
print("\nSentiment Distribution:")
print(df["sentiment"].value_counts())

print("\nTop Issue Categories:")
print(df["category"].value_counts())

print("\nHigh Risk Tickets (Revenue Impact):")
print(df[df["risk_score"] > 3000][["message", "order_value", "sentiment"]])

print("\nSample Processed Data:\n")
print(df.head())

print("\n==============================")
print("AI ANALYSIS COMPLETE")
print("==============================\n")