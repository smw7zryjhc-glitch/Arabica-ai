import os

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = Flask(__name__)

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"]
)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({
            "error": "اكتب رسالة أولاً."
        }), 400

    try:

        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b:groq",
            messages=[
                {
                    "role": "system",
                    "content": """
أنت مساعد ذكاء اصطناعي عربي.
أجب باللغة العربية ما لم يطلب المستخدم لغة أخرى.
اجعل إجاباتك واضحة ومنظمة ومفيدة.
"""
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ]
        )

        answer = completion.choices[0].message.content

        return jsonify({
            "answer": answer
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000
    )
